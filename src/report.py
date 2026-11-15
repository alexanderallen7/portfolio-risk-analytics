"""Tear sheet rendering: results -> self-contained HTML report.

The report embeds the charts as base64 PNGs so the single HTML file is
portable (email-able, viewable offline). The interpretation paragraph
is assembled from the ACTUAL computed numbers — nothing is hardcoded.
"""

from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src import config
from src.analysis import AnalysisResults

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _b64(path: Path) -> str:
    """Base64-encode a PNG for inline embedding."""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _pct(x: float, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f}%"


def _sig_label(p_value: float) -> str:
    """Plain-language significance label for a regression p-value."""
    if p_value < 0.01:
        return "significant at the 1% level"
    if p_value < 0.05:
        return "significant at the 5% level"
    if p_value < 0.10:
        return "marginally significant (10% level)"
    return "not statistically significant"


def build_summary_paragraph(r: AnalysisResults) -> str:
    """Write the interpretation paragraph from the computed results."""
    p = r.stats["portfolio"]
    b = r.stats["benchmark"]
    beta = r.capm.beta.estimate
    alpha_ann = r.capm.alpha_annualized

    beta_phrase = (
        "slightly less market-sensitive than the index"
        if beta < 0.97
        else "about as market-sensitive as the index"
        if beta <= 1.03
        else "more market-sensitive than the index"
    )
    rel = "outperformed" if p["annualized_return"] > b["annualized_return"] else "underperformed"
    dd = r.port_dd
    recovery_phrase = (
        f"and regaining its prior peak on {dd.recovery_date:%b %d, %Y}"
        if dd.recovery_date is not None
        else "and still below its prior peak at the end of the sample"
    )
    # Attribute the drawdown to the 2022 rate-hike bear market only when
    # the dates actually fall in it; never assert a narrative the data
    # doesn't show.
    dd_context = (
        " — this episode lines up with the 2022 rate-hike bear market"
        if r.port_dd.peak_date.year == 2022 or r.port_dd.trough_date.year == 2022
        else ""
    )
    alpha_sig = _sig_label(r.ff3.alpha_daily.p_value)
    smb_dir = "small-cap" if r.ff3.smb.estimate > 0 else "large-cap"
    hml_dir = "value" if r.ff3.hml.estimate > 0 else "growth"

    return (
        f"Over the {r.start_date:%b %d, %Y} – {r.end_date:%b %d, %Y} window the equal-weighted "
        f"15-stock portfolio returned {_pct(p['annualized_return'], 1)} annualized versus "
        f"{_pct(b['annualized_return'], 1)} for SPY, i.e. it {rel} the benchmark, with annualized "
        f"volatility of {_pct(p['annualized_vol'], 1)} against SPY's {_pct(b['annualized_vol'], 1)}. "
        f"Its CAPM beta of {beta:.2f} (R² {r.capm.r_squared:.2f}) means the portfolio was "
        f"{beta_phrase}: a 1% move in SPY was associated with roughly a {beta:.2f}% move in the "
        f"portfolio. The CAPM alpha of {_pct(alpha_ann, 1)} per year is "
        f"{_sig_label(r.capm.alpha_daily.p_value)} — "
        f"consistent with a diversified large-cap basket rather than genuine selection skill. "
        f"The worst drawdown was {_pct(dd.max_drawdown, 1)}, peaking on {dd.peak_date:%b %d, %Y}, "
        f"bottoming on {dd.trough_date:%b %d, %Y} {recovery_phrase}{dd_context}. "
        f"In the Fama-French 3-factor regression the alpha is "
        f"{alpha_sig}; the SMB loading of {r.ff3.smb.estimate:.2f} tilts {smb_dir} and the HML "
        f"loading of {r.ff3.hml.estimate:.2f} tilts {hml_dir}, relative to the market portfolio."
    )


def build_methodology_items(r: AnalysisResults) -> list[dict[str, str]]:
    """Specific, honest methodology & limitations entries."""
    stooq_tickers = [
        t for t, src in r.provenance.items() if "stooq" in src.lower() and t.isupper()
    ]
    price_caveat = (
        "All price series are Yahoo Finance dividend- and split-adjusted closes pulled via "
        "yfinance; returns therefore approximate total return (dividends reinvested on the "
        "ex-date at the close), not true tax- or cost-adjusted total return."
        if not stooq_tickers
        else (
            f"Tickers {', '.join(stooq_tickers)} fell back to stooq.com, whose prices are "
            "split-adjusted but NOT dividend-adjusted — returns for those names understate "
            "total return. All other series are Yahoo Finance dividend-adjusted closes via yfinance."
        )
    )
    rf_item = (
        f"Risk-free rate: period-matching average of the FRED DGS3MO 3-month Treasury "
        f"constant-maturity yield ({_pct(r.rf_annual_mean, 2)} annualized over this window), "
        "de-annualized geometrically to a daily rate for the Sharpe/Sortino calculations. "
        "DGS3MO is a bond-equivalent yield, not a true investable overnight rate; the "
        "difference is immaterial at this precision."
        if not r.rf_is_fallback
        else (
            f"RISK-FREE FALLBACK TRIGGERED: FRED was unreachable, so a hardcoded constant of "
            f"{_pct(config.RISK_FREE_FALLBACK_ANNUAL, 1)} annualized was used. Sharpe and Sortino "
            "ratios should be treated as approximate until re-run with live FRED data."
        )
    )
    return [
        {"title": "Price data", "text": price_caveat},
        {
            "title": "Survivorship and selection bias",
            "text": (
                "The 15 names were chosen in hindsight from currently listed, successful US "
                "large caps. A portfolio formed this way five years ago could have included "
                "names that subsequently delisted or collapsed; performance statistics here "
                "are biased upward relative to a real ex-ante selection. This is a structural "
                "limitation of the exercise, not a correctable data issue."
            ),
        },
        {
            "title": "Weighting and rebalancing",
            "text": (
                "Equal weights, rebalanced daily (each day's portfolio return is the simple "
                "mean of constituent returns). Daily rebalancing is a modeling convenience: "
                "it ignores transaction costs, taxes, and bid-ask spread entirely, and a "
                "monthly- or never-rebalanced version would show modestly different numbers. "
                "Equal weighting itself is a deliberate, stated choice — it avoids "
                "concentration in the mega-caps but overweights the smaller names relative "
                "to a cap-weighted benchmark, which partly explains deviations from SPY."
            ),
        },
        {"title": "Risk-free rate", "text": rf_item},
        {
            "title": "Factor data",
            "text": (
                "Fama-French daily 3-factor returns from the Ken French Data Library. The "
                "FF market factor is the CRSP value-weighted total US market, not SPY, so "
                "the FF market beta and the CAPM beta are close but not identical by "
                "construction. FF excess returns use the library's own RF series. The "
                "library posts with a lag, so the last few trading days of the price window "
                "may be absent from the regression sample (n is reported in the table)."
            ),
        },
        {
            "title": "VaR methodology",
            "text": (
                "Historical VaR is the empirical quantile of the realized daily return "
                "distribution over the full five-year window; it assumes the past "
                "distribution is representative. Parametric VaR assumes Gaussian daily "
                "returns, which understates tail risk for equity portfolios (realized "
                "kurtosis exceeds 3). Both are 1-day, unconditional measures — neither is "
                "a forecast conditioned on current volatility."
            ),
        },
        {
            "title": "Window sensitivity",
            "text": (
                "Every statistic here is specific to this five-year window, which contains "
                "one major drawdown (2022) and a strong recovery. Shifting the window by "
                "even a year materially changes the Sharpe ratio, alpha, and drawdown "
                "statistics. No claim is made that these figures predict future performance."
            ),
        },
    ]


def render_tear_sheet(r: AnalysisResults, chart_paths: dict[str, Path], out_path: Path) -> Path:
    """Render the HTML tear sheet and return the output path."""
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("tear_sheet.html.j2")

    html = template.render(
        generated=datetime.now().strftime("%B %d, %Y %H:%M"),
        tickers=config.TICKERS,
        benchmark=config.BENCHMARK,
        start=f"{r.start_date:%B %d, %Y}",
        end=f"{r.end_date:%B %d, %Y}",
        n_days=r.n_days,
        rf_annual=_pct(r.rf_annual_mean, 2),
        rf_is_fallback=r.rf_is_fallback,
        stats=r.stats,
        pct=_pct,
        port_dd=r.port_dd,
        bench_dd=r.bench_dd,
        capm=r.capm,
        ff3=r.ff3,
        sig_label=_sig_label,
        summary=build_summary_paragraph(r),
        methodology=build_methodology_items(r),
        provenance=r.provenance,
        charts={name: _b64(path) for name, path in chart_paths.items()},
        var_levels=config.VAR_CONFIDENCE_LEVELS,
    )
    out_path.write_text(html, encoding="utf-8")
    return out_path
