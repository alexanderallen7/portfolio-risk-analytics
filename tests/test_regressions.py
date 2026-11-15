"""Unit tests for the regression layer.

Noise-free constructions give exactly recoverable coefficients; a
seeded noisy construction checks inference behaves sanely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import regressions
from src.config import TRADING_DAYS_PER_YEAR


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2023-01-02", periods=n)


def test_capm_recovers_exact_coefficients() -> None:
    # y = 0.0002 + 1.25 * x with zero noise: OLS must recover both exactly
    # with R^2 = 1.
    n = 60
    rng = np.random.default_rng(seed=42)  # fixed seed for reproducibility
    x = pd.Series(rng.normal(0, 0.01, n), index=_dates(n))
    y = 0.0002 + 1.25 * x
    result = regressions.capm_regression(y, x)
    assert result.beta.estimate == pytest.approx(1.25, abs=1e-10)
    assert result.alpha_daily.estimate == pytest.approx(0.0002, abs=1e-10)
    assert result.r_squared == pytest.approx(1.0, abs=1e-10)
    assert result.n_obs == n


def test_capm_alpha_annualization() -> None:
    n = 50
    rng = np.random.default_rng(seed=1)
    x = pd.Series(rng.normal(0, 0.01, n), index=_dates(n))
    y = 0.0001 + 1.0 * x
    result = regressions.capm_regression(y, x)
    assert result.alpha_annualized == pytest.approx(0.0001 * TRADING_DAYS_PER_YEAR, rel=1e-6)


def test_capm_beta_significant_alpha_not_in_noisy_data() -> None:
    # Strong beta, zero true alpha, modest noise: beta should be highly
    # significant, alpha should not be.
    n = 1000
    rng = np.random.default_rng(seed=123)
    x = pd.Series(rng.normal(0, 0.01, n), index=_dates(n))
    y = 1.0 * x + pd.Series(rng.normal(0, 0.002, n), index=_dates(n))
    result = regressions.capm_regression(y, x)
    assert result.beta.p_value < 1e-6
    assert result.alpha_daily.p_value > 0.01


def test_ff3_recovers_exact_loadings() -> None:
    # y = 0.0001 + 1.1*mkt + 0.3*smb - 0.2*hml, zero noise.
    n = 120
    rng = np.random.default_rng(seed=7)
    factors = pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0, 0.01, n),
            "SMB": rng.normal(0, 0.005, n),
            "HML": rng.normal(0, 0.005, n),
            "RF": np.full(n, 0.0001),
        },
        index=_dates(n),
    )
    y = 0.0001 + 1.1 * factors["Mkt-RF"] + 0.3 * factors["SMB"] - 0.2 * factors["HML"]
    result = regressions.ff3_regression(y, factors)
    assert result.mkt_beta.estimate == pytest.approx(1.1, abs=1e-10)
    assert result.smb.estimate == pytest.approx(0.3, abs=1e-10)
    assert result.hml.estimate == pytest.approx(-0.2, abs=1e-10)
    assert result.alpha_daily.estimate == pytest.approx(0.0001, abs=1e-10)
    assert result.r_squared == pytest.approx(1.0, abs=1e-10)


def test_regressions_drop_unaligned_dates() -> None:
    # Portfolio has 10 extra days the factor series lacks; the regression
    # must use only the overlap.
    n = 50
    rng = np.random.default_rng(seed=9)
    x = pd.Series(rng.normal(0, 0.01, n), index=_dates(n))
    y_full = pd.Series(rng.normal(0, 0.01, n + 10), index=_dates(n + 10))
    result = regressions.capm_regression(y_full, x)
    assert result.n_obs == n
