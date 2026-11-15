"""End-to-end analysis: read cache, compute all metrics, return results.

This module is the glue between the cached raw data and the report
layer. It does no network I/O; everything reads from ``data/raw/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src import config, data_fetch, metrics, regressions


@dataclass
class AnalysisResults:
    """Everything the report and charts need, computed once."""

    start_date: pd.Timestamp
    end_date: pd.Timestamp
    n_days: int
    rf_annual_mean: float  # period-average annualized risk-free rate (decimal)
    rf_is_fallback: bool
    provenance: dict[str, str]

    port_returns: pd.Series | None = field(repr=False, default=None)
    bench_returns: pd.Series | None = field(repr=False, default=None)
    asset_returns: pd.DataFrame | None = field(repr=False, default=None)

    stats: dict[str, dict[str, Any]] = field(default_factory=dict)  # per-leg metric table
    port_dd: metrics.DrawdownResult | None = None
    bench_dd: metrics.DrawdownResult | None = None
    capm: regressions.CAPMResult | None = None
    ff3: regressions.FF3Result | None = None
    corr: pd.DataFrame | None = field(repr=False, default=None)
    port_roll_vol: pd.Series | None = field(repr=False, default=None)
    bench_roll_vol: pd.Series | None = field(repr=False, default=None)


def _leg_stats(returns: pd.Series, rf_daily: pd.Series | float) -> dict[str, Any]:
    """Compute the standard metric block for one return series."""
    out: dict[str, Any] = {
        "total_return": metrics.total_return(returns),
        "annualized_return": metrics.annualized_return(returns),
        "annualized_vol": metrics.annualized_volatility(returns),
        "sharpe": metrics.sharpe_ratio(returns, rf_daily),
        "sortino": metrics.sortino_ratio(returns, rf_daily),
    }
    for conf in config.VAR_CONFIDENCE_LEVELS:
        key = f"{conf:.0%}".rstrip("%")
        out[f"hist_var_{key}"] = metrics.historical_var(returns, conf)
        out[f"param_var_{key}"] = metrics.parametric_var(returns, conf)
    return out


def run_analysis() -> AnalysisResults:
    """Read the cache and compute every metric, regression, and series."""
    prices = data_fetch.load_prices().dropna(how="any")
    provenance = data_fetch.load_provenance()

    asset_prices = prices[config.TICKERS]
    bench_prices = prices[config.BENCHMARK]

    asset_returns = metrics.daily_returns(asset_prices)
    bench_returns = metrics.daily_returns(bench_prices)
    port_returns = metrics.equal_weight_portfolio_returns(asset_returns)

    # --- Risk-free rate: period-matching average of DGS3MO -------------
    rf_series = data_fetch.load_risk_free()
    rf_is_fallback = rf_series is None
    if rf_is_fallback:
        rf_annual_mean = config.RISK_FREE_FALLBACK_ANNUAL
        rf_daily: pd.Series | float = metrics.annual_yield_to_daily(rf_annual_mean)
    else:
        rf_window = rf_series.loc[port_returns.index.min() : port_returns.index.max()]
        rf_annual_mean = float(rf_window.mean()) / 100.0  # percent -> decimal
        # Align the daily rf to return dates (forward-fill holidays/gaps).
        rf_aligned = (rf_series / 100.0).reindex(port_returns.index).ffill().bfill()
        rf_daily = metrics.annual_yield_to_daily(rf_aligned)

    results = AnalysisResults(
        start_date=port_returns.index.min(),
        end_date=port_returns.index.max(),
        n_days=len(port_returns),
        rf_annual_mean=rf_annual_mean,
        rf_is_fallback=rf_is_fallback,
        provenance=provenance,
        port_returns=port_returns,
        bench_returns=bench_returns,
        asset_returns=asset_returns,
    )

    results.stats["portfolio"] = _leg_stats(port_returns, rf_daily)
    results.stats["benchmark"] = _leg_stats(bench_returns, rf_daily)

    results.port_dd = metrics.max_drawdown(port_returns)
    results.bench_dd = metrics.max_drawdown(bench_returns)

    # --- CAPM vs. SPY ----------------------------------------------------
    port_excess = port_returns - rf_daily
    bench_excess = bench_returns - rf_daily
    results.capm = regressions.capm_regression(port_excess, bench_excess)

    # --- Fama-French 3-factor (uses FF's own RF for excess returns) ------
    ff3 = data_fetch.load_ff3()
    port_excess_ff = (port_returns - ff3["RF"]).dropna()
    results.ff3 = regressions.ff3_regression(port_excess_ff, ff3)

    results.corr = metrics.correlation_matrix(asset_returns)
    results.port_roll_vol = metrics.rolling_volatility(port_returns, config.ROLLING_VOL_WINDOW)
    results.bench_roll_vol = metrics.rolling_volatility(bench_returns, config.ROLLING_VOL_WINDOW)

    return results
