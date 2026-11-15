"""Pure metric functions: returns, risk, drawdown, VaR.

Every function here is side-effect free (no I/O, no globals mutated) and
operates on pandas Series/DataFrames of DAILY data, so each one is unit
testable against hand-computed values. Annualization uses the 252
trading-day convention from ``config.TRADING_DAYS_PER_YEAR``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.config import TRADING_DAYS_PER_YEAR


def daily_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Simple daily returns from a price series (first row dropped)."""
    return prices.pct_change().dropna()


def total_return(returns: pd.Series) -> float:
    """Cumulative compounded return over the whole sample, as a decimal."""
    return float((1.0 + returns).prod() - 1.0)


def annualized_return(returns: pd.Series) -> float:
    """Geometric annualized return (CAGR) from daily returns."""
    n = len(returns)
    if n == 0:
        return float("nan")
    growth = float((1.0 + returns).prod())
    return growth ** (TRADING_DAYS_PER_YEAR / n) - 1.0


def annualized_volatility(returns: pd.Series) -> float:
    """Sample std of daily returns scaled by sqrt(252)."""
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(returns: pd.Series, rf_daily: pd.Series | float) -> float:
    """Annualized Sharpe ratio using a daily risk-free series or constant.

    ``rf_daily`` is the PER-DAY risk-free return (not annualized). Excess
    returns are computed day by day, then annualized as mean*252 over
    std*sqrt(252).
    """
    excess = returns - rf_daily
    if isinstance(excess, pd.DataFrame):
        raise TypeError("sharpe_ratio expects a single return series")
    excess = excess.dropna()
    denom = excess.std(ddof=1)
    if denom == 0:
        return float("nan")
    return float(excess.mean() / denom * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(returns: pd.Series, rf_daily: pd.Series | float) -> float:
    """Annualized Sortino ratio: mean excess over downside deviation.

    Downside deviation uses the full-sample count in the denominator
    (the standard definition), with only negative excess returns
    contributing to the sum of squares.
    """
    excess = (returns - rf_daily).dropna()
    downside = np.minimum(excess, 0.0)
    downside_dev = float(np.sqrt((downside**2).sum() / len(excess)))
    if downside_dev == 0:
        return float("nan")
    return float(excess.mean() / downside_dev * np.sqrt(TRADING_DAYS_PER_YEAR))


@dataclass(frozen=True)
class DrawdownResult:
    """Maximum drawdown and its key dates.

    ``recovery`` is None when the series has not yet regained the prior
    peak by the end of the sample.
    """

    max_drawdown: float  # most negative peak-to-trough decline, e.g. -0.25
    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: pd.Timestamp | None
    series: pd.Series  # full drawdown path (for charting)


def max_drawdown(returns: pd.Series) -> DrawdownResult:
    """Maximum peak-to-trough drawdown of the compounded return path."""
    wealth = (1.0 + returns).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0

    trough_date = drawdown.idxmin()
    max_dd = float(drawdown.loc[trough_date])
    # Peak = last date at the running maximum before the trough.
    pre_trough = wealth.loc[:trough_date]
    peak_date = pre_trough[pre_trough == pre_trough.cummax().iloc[-1]].index[-1]
    peak_value = float(wealth.loc[peak_date])
    # Recovery = first date after the trough where wealth regains the peak.
    post_trough = wealth.loc[trough_date:]
    recovered = post_trough[post_trough >= peak_value]
    recovery_date = recovered.index[0] if len(recovered) else None

    return DrawdownResult(
        max_drawdown=max_dd,
        peak_date=pd.Timestamp(peak_date),
        trough_date=pd.Timestamp(trough_date),
        recovery_date=pd.Timestamp(recovery_date) if recovery_date is not None else None,
        series=drawdown,
    )


def historical_var(returns: pd.Series, confidence: float) -> float:
    """Historical (empirical-quantile) 1-day VaR, reported as a POSITIVE loss.

    E.g. confidence=0.95 returns the magnitude of the 5th-percentile
    daily return.
    """
    quantile = float(np.percentile(returns, (1.0 - confidence) * 100.0))
    return -quantile


def parametric_var(returns: pd.Series, confidence: float) -> float:
    """Parametric (Gaussian) 1-day VaR, reported as a POSITIVE loss.

    Assumes daily returns are normal with the sample mean and std:
    VaR = -(mu + z * sigma) where z is the lower-tail normal quantile.
    """
    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1))
    z = float(stats.norm.ppf(1.0 - confidence))
    return -(mu + z * sigma)


def rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Rolling annualized volatility over ``window`` trading days."""
    return returns.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)


def equal_weight_portfolio_returns(asset_returns: pd.DataFrame) -> pd.Series:
    """Daily returns of an equal-weighted, daily-rebalanced portfolio.

    Daily rebalancing to equal weights means the portfolio return each
    day is the simple cross-sectional mean of the constituents' returns.
    (The rebalancing convention is stated in the report's methodology.)
    """
    return asset_returns.mean(axis=1)


def correlation_matrix(asset_returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of daily returns."""
    return asset_returns.corr()


def annual_yield_to_daily(annual_yield_decimal: pd.Series | float):
    """Convert an annualized yield (decimal, e.g. 0.045) to a per-trading-day
    simple return via geometric de-annualization."""
    return (1.0 + annual_yield_decimal) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
