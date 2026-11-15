"""Central configuration for the portfolio risk analytics pipeline.

All tunable choices live here so the analysis scripts contain no magic
numbers. Each constant documents why its value was chosen.
"""

from __future__ import annotations

from pathlib import Path

# --- Portfolio definition -------------------------------------------------

#: The 15-stock sample portfolio. Chosen for sector diversification across
#: large-cap US equities (tech, financials, healthcare, staples, energy,
#: industrials, utilities, consumer discretionary).
TICKERS: list[str] = [
    "AAPL", "MSFT", "JPM", "JNJ", "PG",
    "XOM", "UNH", "HD", "V", "PEP",
    "KO", "DIS", "CAT", "NEE", "COST",
]

#: Benchmark used for relative performance and the CAPM regression.
BENCHMARK: str = "SPY"

#: Lookback window in calendar years, ending at the most recent available
#: trading day. Five years captures the 2022 bear market and the
#: subsequent recovery, giving the drawdown and VaR statistics a real
#: stress period to work with.
LOOKBACK_YEARS: int = 5

# --- Conventions ----------------------------------------------------------

#: Annualization convention: the standard US equity trading calendar has
#: roughly 252 trading days per year.
TRADING_DAYS_PER_YEAR: int = 252

#: Confidence levels for Value-at-Risk. 95% and 99% are the two levels
#: most commonly quoted in practice (RiskMetrics popularized 95%; Basel
#: market-risk rules use 99%).
VAR_CONFIDENCE_LEVELS: tuple[float, ...] = (0.95, 0.99)

#: Window (in trading days) for the rolling volatility chart. 60 trading
#: days is roughly one quarter, long enough to smooth noise but short
#: enough to show regime changes.
ROLLING_VOL_WINDOW: int = 60

#: FRED series for the risk-free rate: 3-month Treasury constant-maturity
#: yield, quoted as an annualized percentage.
RISK_FREE_SERIES: str = "DGS3MO"

#: Fallback annualized risk-free rate (decimal) used only if FRED is
#: unreachable. Roughly the average 3-month T-bill yield over 2021-2025;
#: if this fallback is triggered the report flags it explicitly.
RISK_FREE_FALLBACK_ANNUAL: float = 0.03

# --- Paths ----------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"

PRICES_CACHE: Path = RAW_DATA_DIR / "prices.csv"
RISK_FREE_CACHE: Path = RAW_DATA_DIR / "risk_free_dgs3mo.csv"
FF3_CACHE: Path = RAW_DATA_DIR / "ff3_daily.csv"
PROVENANCE_FILE: Path = RAW_DATA_DIR / "provenance.json"
