"""Unit tests for the pure metric functions.

Every test verifies against a value computed BY HAND (shown in the
comments), not against the implementation itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import metrics
from src.config import TRADING_DAYS_PER_YEAR


def _series(values: list[float], start: str = "2024-01-01") -> pd.Series:
    """Daily-indexed return series for test readability."""
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx)


# --- returns ----------------------------------------------------------------


def test_daily_returns_from_prices() -> None:
    prices = _series([100.0, 110.0, 99.0])
    returns = metrics.daily_returns(prices)
    # 100 -> 110 is +10%; 110 -> 99 is -10%.
    assert returns.iloc[0] == pytest.approx(0.10)
    assert returns.iloc[1] == pytest.approx(-0.10)
    assert len(returns) == 2  # first row dropped


def test_total_return_compounds() -> None:
    # (1.10)(0.95) - 1 = 0.045 exactly.
    returns = _series([0.10, -0.05])
    assert metrics.total_return(returns) == pytest.approx(0.045)


def test_annualized_return_constant_daily() -> None:
    # A constant 0.1% daily return annualizes to exactly (1.001)^252 - 1,
    # regardless of sample length.
    returns = _series([0.001] * 100)
    expected = 1.001**TRADING_DAYS_PER_YEAR - 1.0
    assert metrics.annualized_return(returns) == pytest.approx(expected)


def test_annualized_volatility_two_points() -> None:
    # For [-1%, +1%]: mean 0, sample var = (0.0001 + 0.0001)/(2-1) = 0.0002,
    # std = sqrt(0.0002); annualized = sqrt(0.0002) * sqrt(252).
    returns = _series([-0.01, 0.01])
    expected = np.sqrt(0.0002) * np.sqrt(TRADING_DAYS_PER_YEAR)
    assert metrics.annualized_volatility(returns) == pytest.approx(expected)


# --- risk-adjusted ratios ----------------------------------------------------


def test_sharpe_zero_rf_hand_computed() -> None:
    # returns [1%, 2%, 3%]: mean 0.02, sample std 0.01;
    # Sharpe = 0.02/0.01 * sqrt(252) = 2*sqrt(252).
    returns = _series([0.01, 0.02, 0.03])
    assert metrics.sharpe_ratio(returns, 0.0) == pytest.approx(2.0 * np.sqrt(252))


def test_sharpe_constant_rf_shifts_mean() -> None:
    # Subtracting a constant rf of 1% daily turns the mean into 0.01 while
    # the std is unchanged: Sharpe = 1*sqrt(252).
    returns = _series([0.01, 0.02, 0.03])
    assert metrics.sharpe_ratio(returns, 0.01) == pytest.approx(1.0 * np.sqrt(252))


def test_sortino_hand_computed() -> None:
    # returns [2%, -1%, 3%, -2%], rf=0: mean = 0.005.
    # Downside dev = sqrt((0.01^2 + 0.02^2)/4) = sqrt(0.000125).
    returns = _series([0.02, -0.01, 0.03, -0.02])
    expected = 0.005 / np.sqrt(0.000125) * np.sqrt(252)
    assert metrics.sortino_ratio(returns, 0.0) == pytest.approx(expected)


def test_sortino_no_downside_is_nan() -> None:
    returns = _series([0.01, 0.02])
    assert np.isnan(metrics.sortino_ratio(returns, 0.0))


# --- drawdown -----------------------------------------------------------------


def test_max_drawdown_known_path_with_recovery() -> None:
    # Prices 100, 110, 99, 88, 110, 121.
    # Peak 110 (day 2), trough 88 (day 4): drawdown = 88/110 - 1 = -20%.
    # Wealth regains 110 on day 5 -> recovery is day 5.
    prices = _series([100.0, 110.0, 99.0, 88.0, 110.0, 121.0])
    returns = metrics.daily_returns(prices)
    result = metrics.max_drawdown(returns)
    assert result.max_drawdown == pytest.approx(-0.20)
    assert result.peak_date == prices.index[1]
    assert result.trough_date == prices.index[3]
    assert result.recovery_date == prices.index[4]


def test_max_drawdown_not_recovered() -> None:
    # Prices 100, 120, 90, 95: drawdown = 90/120 - 1 = -25%, never regains 120.
    prices = _series([100.0, 120.0, 90.0, 95.0])
    returns = metrics.daily_returns(prices)
    result = metrics.max_drawdown(returns)
    assert result.max_drawdown == pytest.approx(-0.25)
    assert result.recovery_date is None


def test_drawdown_series_is_zero_at_running_peaks() -> None:
    prices = _series([100.0, 110.0, 99.0, 88.0, 110.0, 121.0])
    result = metrics.max_drawdown(metrics.daily_returns(prices))
    # At the new highs (110 on day 5, 121 on day 6) drawdown must be ~0.
    assert result.series.iloc[-1] == pytest.approx(0.0)


# --- VaR -----------------------------------------------------------------------


def test_historical_var_exact_percentile() -> None:
    # 21 evenly spaced returns from -10% to +10%: the 5th percentile falls
    # exactly on the second value, -9%. VaR is reported as +9%.
    returns = _series(list(np.linspace(-0.10, 0.10, 21)))
    assert metrics.historical_var(returns, 0.95) == pytest.approx(0.09)


def test_parametric_var_gaussian_formula() -> None:
    # Symmetric [-1%, +1%]: mean 0, sample std = sqrt(0.0002).
    # VaR95 = 1.6448536... * std (z for the 5% lower tail).
    returns = _series([-0.01, 0.01])
    z_95 = 1.6448536269514722
    expected = z_95 * np.sqrt(0.0002)
    assert metrics.parametric_var(returns, 0.95) == pytest.approx(expected)


def test_parametric_var_99_deeper_than_95() -> None:
    rng = np.random.default_rng(seed=7)  # fixed seed for reproducibility
    returns = _series(list(rng.normal(0.0, 0.01, size=500)))
    assert metrics.parametric_var(returns, 0.99) > metrics.parametric_var(returns, 0.95)
    assert metrics.historical_var(returns, 0.99) > metrics.historical_var(returns, 0.95)


# --- portfolio construction -----------------------------------------------------


def test_equal_weight_is_cross_sectional_mean() -> None:
    frame = pd.DataFrame(
        {"A": [0.02, -0.02], "B": [0.04, 0.00]},
        index=pd.bdate_range("2024-01-01", periods=2),
    )
    port = metrics.equal_weight_portfolio_returns(frame)
    assert port.iloc[0] == pytest.approx(0.03)   # (0.02 + 0.04)/2
    assert port.iloc[1] == pytest.approx(-0.01)  # (-0.02 + 0.00)/2


def test_correlation_matrix_perfect_and_inverse() -> None:
    frame = pd.DataFrame(
        {"A": [0.01, 0.02, 0.03], "B": [-0.01, -0.02, -0.03]},
        index=pd.bdate_range("2024-01-01", periods=3),
    )
    corr = metrics.correlation_matrix(frame)
    assert corr.loc["A", "A"] == pytest.approx(1.0)
    assert corr.loc["A", "B"] == pytest.approx(-1.0)


# --- helpers ----------------------------------------------------------------------


def test_annual_yield_to_daily_roundtrip() -> None:
    # 5% annual -> daily rate that compounds back to exactly 5% over 252 days.
    daily = metrics.annual_yield_to_daily(0.05)
    assert (1 + daily) ** TRADING_DAYS_PER_YEAR - 1 == pytest.approx(0.05)


def test_rolling_volatility_window_and_scale() -> None:
    returns = _series([0.01, -0.01] * 50)
    roll = metrics.rolling_volatility(returns, window=10)
    # First 9 entries undefined; thereafter equals sample std of the
    # alternating window * sqrt(252).
    assert roll.iloc[:9].isna().all()
    manual = returns.iloc[:10].std(ddof=1) * np.sqrt(252)
    assert roll.iloc[9] == pytest.approx(manual)
