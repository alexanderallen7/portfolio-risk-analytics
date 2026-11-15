"""Data-fetching layer: prices, risk-free rate, and Fama-French factors.

All pulls hit free, keyless public sources, are wrapped in
retry-with-backoff logic, and are cached to ``data/raw/``. Every function
here records provenance (which source actually supplied each series) so
the report's methodology section reflects what really happened.

After a successful first run the entire analysis works offline from the
cache. Pass ``force_refresh=True`` (or ``--refresh`` on the CLI script)
to re-pull.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from datetime import date, timedelta
from typing import Callable, TypeVar
from urllib.request import Request, urlopen

import pandas as pd

from src import config

T = TypeVar("T")

#: Retry policy for network pulls. Yahoo in particular rate-limits
#: intermittently; three attempts with exponential backoff clears
#: transient failures without stalling the pipeline for minutes.
MAX_RETRIES: int = 3
BACKOFF_BASE_SECONDS: float = 2.0

STOOQ_CSV_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d&d1={start}&d2={end}"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
FF3_ZIP_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)

_HTTP_HEADERS = {"User-Agent": "portfolio-risk-analytics/1.0 (research tool)"}


def _retry(fn: Callable[[], T], label: str) -> T:
    """Run ``fn`` with exponential-backoff retries; re-raise the last error."""
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as err:  # noqa: BLE001 - network errors vary widely
            last_err = err
            if attempt < MAX_RETRIES:
                wait = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                print(f"  [{label}] attempt {attempt} failed ({err}); retrying in {wait:.0f}s")
                time.sleep(wait)
    raise RuntimeError(f"{label}: all {MAX_RETRIES} attempts failed") from last_err


def _http_get(url: str, timeout: float = 30.0) -> bytes:
    """Plain HTTP GET with a browser-ish User-Agent (stooq/FF require one)."""
    req = Request(url, headers=_HTTP_HEADERS)
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


# --- Prices ---------------------------------------------------------------


def fetch_prices_yfinance(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Pull dividend/split-adjusted daily closes from Yahoo via yfinance.

    Returns a wide DataFrame (index = dates, columns = tickers). Columns
    that come back entirely empty are dropped so the caller can route
    them to the stooq fallback.
    """
    import yfinance as yf

    raw = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=True,  # adjusted for splits and dividends
        progress=False,
        group_by="column",
        threads=False,  # threaded pulls can lock yfinance's sqlite cache
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned an empty frame")
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(name=tickers[0])
    closes = closes.dropna(axis=1, how="all")
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    return closes.sort_index()


def fetch_prices_stooq(symbol: str, start: date, end: date) -> pd.Series:
    """Fallback: pull daily closes for one ticker from stooq.com CSV.

    Stooq US symbols take a ``.us`` suffix. NOTE: stooq closes are
    split-adjusted but NOT dividend-adjusted, which understates total
    return; the methodology section flags any ticker served this way.
    """
    url = STOOQ_CSV_URL.format(
        symbol=f"{symbol.lower()}.us",
        start=start.strftime("%Y%m%d"),
        end=end.strftime("%Y%m%d"),
    )
    raw = _http_get(url)
    text = raw.decode("utf-8", errors="replace")
    if not text.lstrip().startswith("Date,"):
        # Stooq answers rate-limit/no-data conditions with a plain-text
        # message instead of a CSV; surface it so the retry logic sees it.
        raise RuntimeError(f"stooq returned non-CSV response for {symbol}: {text[:80]!r}")
    frame = pd.read_csv(io.StringIO(text), parse_dates=["Date"], index_col="Date")
    if frame.empty or "Close" not in frame:
        raise RuntimeError(f"stooq returned no data for {symbol}")
    series = frame["Close"].astype(float).sort_index()
    series.name = symbol
    return series


def fetch_all_prices(
    tickers: list[str], start: date, end: date
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fetch adjusted closes for all tickers, yfinance first, stooq fallback.

    Returns ``(prices, provenance)`` where provenance maps each ticker to
    the source that actually supplied it.
    """
    provenance: dict[str, str] = {}
    try:
        prices = _retry(lambda: fetch_prices_yfinance(tickers, start, end), "yfinance")
    except RuntimeError:
        prices = pd.DataFrame()
    got = set(prices.columns)
    for ticker in tickers:
        if ticker in got:
            provenance[ticker] = "yfinance (Yahoo Finance, dividend-adjusted)"
    missing = [t for t in tickers if t not in got]
    for ticker in missing:
        # Transient per-ticker Yahoo failures (rate limits, cache locks)
        # are common; retry the single name on yfinance before falling
        # back to stooq, whose prices lack dividend adjustment.
        try:
            single = _retry(
                lambda t=ticker: fetch_prices_yfinance([t], start, end)[t],
                f"yfinance:{ticker}",
            )
            provenance[ticker] = "yfinance (Yahoo Finance, dividend-adjusted; single-ticker retry)"
        except RuntimeError:
            single = _retry(lambda t=ticker: fetch_prices_stooq(t, start, end), f"stooq:{ticker}")
            provenance[ticker] = "stooq.com CSV (split-adjusted only, no dividend adjustment)"
        series = single.rename(ticker)
        prices = prices.join(series, how="outer") if not prices.empty else series.to_frame()
    still_missing = [t for t in tickers if t not in prices.columns]
    if still_missing:
        raise RuntimeError(f"No source could supply prices for: {still_missing}")
    return prices[tickers].sort_index(), provenance


# --- Risk-free rate -------------------------------------------------------


def fetch_risk_free(start: date, end: date) -> tuple[pd.Series, str]:
    """Pull the 3-month Treasury yield (FRED DGS3MO, annualized percent).

    Tries pandas-datareader first, then FRED's keyless fredgraph CSV
    endpoint. Returns ``(series, source_label)``. Raises if both fail;
    the caller decides whether to use the hardcoded fallback constant.
    """

    def _via_datareader() -> pd.Series:
        from pandas_datareader import data as pdr

        frame = pdr.DataReader(config.RISK_FREE_SERIES, "fred", start, end)
        return frame[config.RISK_FREE_SERIES].dropna()

    def _via_csv() -> pd.Series:
        raw = _http_get(FRED_CSV_URL.format(series=config.RISK_FREE_SERIES))
        frame = pd.read_csv(io.BytesIO(raw), na_values=["."])
        date_col = frame.columns[0]  # FRED has used both 'DATE' and 'observation_date'
        frame[date_col] = pd.to_datetime(frame[date_col])
        series = frame.set_index(date_col)[config.RISK_FREE_SERIES].dropna()
        return series.loc[str(start) : str(end)]

    try:
        series = _retry(_via_datareader, "FRED via pandas-datareader")
        return series, "FRED DGS3MO via pandas-datareader"
    except Exception:  # noqa: BLE001
        series = _retry(_via_csv, "FRED via fredgraph CSV")
        return series, "FRED DGS3MO via fredgraph.csv (keyless CSV endpoint)"


# --- Fama-French factors --------------------------------------------------


def fetch_ff3_daily(start: date, end: date) -> tuple[pd.DataFrame, str]:
    """Download the Fama-French 3-factor daily file from Ken French's site.

    Tries, in order: pandas-datareader's famafrench reader, a direct zip
    download, and finally the Internet Archive's most recent snapshot of
    the SAME official zip (the Tuck server intermittently resets
    connections from some networks; the Wayback copy is byte-identical
    to the official file as of its snapshot date, which the returned
    source label records).

    Returns ``(frame, source_label)`` with columns Mkt-RF, SMB, HML, RF
    in DECIMAL (the source file quotes percent; we divide by 100 so
    everything downstream works in returns space).
    """

    def _via_datareader() -> pd.DataFrame:
        from pandas_datareader import data as pdr

        tables = pdr.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start, end)
        frame = tables[0].copy()  # table 0 = the daily factor returns
        frame.index = pd.to_datetime(frame.index)
        frame.columns = [c.strip() for c in frame.columns]
        return frame.astype(float) / 100.0  # percent -> decimal

    def _parse_zip(raw: bytes) -> pd.DataFrame:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            name = zf.namelist()[0]
            text = zf.read(name).decode("utf-8", errors="replace")
        lines = text.splitlines()
        # The CSV has a free-text preamble; data rows start with YYYYMMDD.
        start_idx = next(
            i for i, ln in enumerate(lines) if ln[:8].strip().isdigit() and len(ln[:8].strip()) == 8
        )
        header_idx = start_idx - 1
        body = "\n".join([lines[header_idx]] + [
            ln for ln in lines[start_idx:] if ln[:8].strip().isdigit()
        ])
        frame = pd.read_csv(io.StringIO(body))
        frame = frame.rename(columns={frame.columns[0]: "Date"})
        frame["Date"] = pd.to_datetime(frame["Date"], format="%Y%m%d")
        frame = frame.set_index("Date").astype(float) / 100.0  # percent -> decimal
        frame.columns = [c.strip() for c in frame.columns]
        return frame

    def _via_wayback() -> tuple[pd.DataFrame, str]:
        cdx_url = (
            "http://web.archive.org/cdx/search/cdx?url="
            "mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
            "F-F_Research_Data_Factors_daily_CSV.zip"
            "&output=json&limit=-1&fl=timestamp,statuscode"
        )
        rows = json.loads(_http_get(cdx_url, timeout=60.0))
        if len(rows) < 2 or rows[-1][1] != "200":
            raise RuntimeError("no usable Wayback snapshot of the FF3 zip")
        snapshot_ts = rows[-1][0]
        # The 'if_' modifier serves the original archived bytes.
        snap_url = (
            f"https://web.archive.org/web/{snapshot_ts}if_/{FF3_ZIP_URL}"
        )
        raw = _http_get(snap_url, timeout=120.0)
        label = (
            "Ken French Data Library official daily 3-factor zip via Internet Archive "
            f"snapshot {snapshot_ts[:4]}-{snapshot_ts[4:6]}-{snapshot_ts[6:8]} "
            "(Tuck server unreachable at fetch time)"
        )
        return _parse_zip(raw), label

    try:
        frame = _retry(_via_datareader, "FF3 via pandas-datareader")
        label = "Ken French Data Library (Dartmouth Tuck) via pandas-datareader, daily 3-factor file"
    except RuntimeError:
        try:
            frame = _retry(lambda: _parse_zip(_http_get(FF3_ZIP_URL, timeout=60.0)),
                           "Ken French FF3 daily (direct zip)")
            label = "Ken French Data Library (Dartmouth Tuck), direct daily 3-factor CSV zip"
        except RuntimeError:
            frame, label = _retry(_via_wayback, "FF3 via Internet Archive snapshot")
    return frame.loc[str(start) : str(end)], label


# --- Cache orchestration ---------------------------------------------------


def fetch_and_cache_all(force_refresh: bool = False) -> dict[str, str]:
    """Pull everything (prices, risk-free, FF3) and cache to data/raw/.

    Skips the network entirely when a complete cache exists and
    ``force_refresh`` is False. Returns the provenance dict that was
    written to disk.
    """
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_complete = all(
        p.exists()
        for p in (config.PRICES_CACHE, config.RISK_FREE_CACHE, config.FF3_CACHE, config.PROVENANCE_FILE)
    )
    if cache_complete and not force_refresh:
        print("Cache complete; skipping network pulls (use --refresh to force).")
        return json.loads(config.PROVENANCE_FILE.read_text())

    end = date.today()
    start = end - timedelta(days=round(config.LOOKBACK_YEARS * 365.25))
    provenance: dict[str, str] = {}

    print(f"Fetching prices for {len(config.TICKERS) + 1} symbols, {start} to {end} ...")
    all_symbols = config.TICKERS + [config.BENCHMARK]
    prices, price_prov = fetch_all_prices(all_symbols, start, end)
    prices.to_csv(config.PRICES_CACHE, index_label="Date")
    provenance.update(price_prov)
    print(f"  cached {prices.shape[0]} rows x {prices.shape[1]} tickers -> {config.PRICES_CACHE.name}")

    print("Fetching risk-free rate (FRED DGS3MO) ...")
    try:
        rf, rf_source = fetch_risk_free(start, end)
        rf.to_frame(name=config.RISK_FREE_SERIES).to_csv(config.RISK_FREE_CACHE, index_label="Date")
        provenance["risk_free"] = rf_source
        print(f"  cached {len(rf)} observations -> {config.RISK_FREE_CACHE.name}")
    except Exception as err:  # noqa: BLE001
        provenance["risk_free"] = (
            f"FALLBACK CONSTANT {config.RISK_FREE_FALLBACK_ANNUAL:.2%} annualized "
            f"(FRED unreachable: {err})"
        )
        print(f"  WARNING: FRED unreachable; fallback constant will be used. ({err})")

    print("Fetching Fama-French 3-factor daily file ...")
    ff3, ff3_source = fetch_ff3_daily(start, end)
    ff3.to_csv(config.FF3_CACHE, index_label="Date")
    provenance["fama_french"] = ff3_source
    print(f"  cached {len(ff3)} rows (through {ff3.index.max():%Y-%m-%d}) -> {config.FF3_CACHE.name}")

    config.PROVENANCE_FILE.write_text(json.dumps(provenance, indent=2))
    return provenance


# --- Cache readers (used by the analysis step; no network) -----------------


def load_prices() -> pd.DataFrame:
    """Read cached adjusted closes (wide: dates x tickers)."""
    return pd.read_csv(config.PRICES_CACHE, index_col="Date", parse_dates=True)


def load_risk_free() -> pd.Series | None:
    """Read the cached DGS3MO series (annualized percent), or None if the
    FRED pull fell back to the hardcoded constant."""
    if not config.RISK_FREE_CACHE.exists():
        return None
    frame = pd.read_csv(config.RISK_FREE_CACHE, index_col="Date", parse_dates=True)
    return frame[config.RISK_FREE_SERIES].dropna()


def load_ff3() -> pd.DataFrame:
    """Read cached Fama-French daily factors (decimal returns)."""
    return pd.read_csv(config.FF3_CACHE, index_col="Date", parse_dates=True)


def load_provenance() -> dict[str, str]:
    """Read the provenance record written at fetch time."""
    return json.loads(config.PROVENANCE_FILE.read_text())
