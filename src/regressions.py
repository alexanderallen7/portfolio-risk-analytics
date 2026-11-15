"""Factor regressions: CAPM vs. the benchmark and Fama-French 3-factor.

Both are plain OLS with intercept, run on EXCESS daily returns. Results
come back as small dataclasses so the report layer can format them
without re-deriving anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm

from src.config import TRADING_DAYS_PER_YEAR


@dataclass(frozen=True)
class CoefStat:
    """One regression coefficient with its inference stats."""

    estimate: float
    t_stat: float
    p_value: float


@dataclass(frozen=True)
class CAPMResult:
    """CAPM regression: r_p - rf = alpha + beta * (r_m - rf) + eps."""

    alpha_daily: CoefStat
    beta: CoefStat
    r_squared: float
    n_obs: int

    @property
    def alpha_annualized(self) -> float:
        """Daily intercept scaled to an annual figure (x252, simple scaling)."""
        return self.alpha_daily.estimate * TRADING_DAYS_PER_YEAR


@dataclass(frozen=True)
class FF3Result:
    """Fama-French 3-factor regression on daily excess returns."""

    alpha_daily: CoefStat
    mkt_beta: CoefStat
    smb: CoefStat
    hml: CoefStat
    r_squared: float
    n_obs: int

    @property
    def alpha_annualized(self) -> float:
        """Daily intercept scaled to an annual figure (x252, simple scaling)."""
        return self.alpha_daily.estimate * TRADING_DAYS_PER_YEAR


def _coef(fit, name: str) -> CoefStat:
    """Extract one coefficient's estimate/t/p from a statsmodels fit."""
    return CoefStat(
        estimate=float(fit.params[name]),
        t_stat=float(fit.tvalues[name]),
        p_value=float(fit.pvalues[name]),
    )


def capm_regression(portfolio_excess: pd.Series, market_excess: pd.Series) -> CAPMResult:
    """OLS of portfolio excess returns on benchmark excess returns."""
    joined = pd.concat(
        {"port": portfolio_excess, "mkt": market_excess}, axis=1
    ).dropna()
    X = sm.add_constant(joined["mkt"].rename("mkt"))
    fit = sm.OLS(joined["port"], X).fit()
    return CAPMResult(
        alpha_daily=_coef(fit, "const"),
        beta=_coef(fit, "mkt"),
        r_squared=float(fit.rsquared),
        n_obs=int(fit.nobs),
    )


def ff3_regression(portfolio_excess: pd.Series, factors: pd.DataFrame) -> FF3Result:
    """OLS of portfolio excess returns on Mkt-RF, SMB, HML.

    ``factors`` must contain columns 'Mkt-RF', 'SMB', 'HML' in decimal
    daily returns, indexed by date.
    """
    joined = pd.concat([portfolio_excess.rename("port"), factors[["Mkt-RF", "SMB", "HML"]]], axis=1).dropna()
    X = sm.add_constant(joined[["Mkt-RF", "SMB", "HML"]])
    fit = sm.OLS(joined["port"], X).fit()
    return FF3Result(
        alpha_daily=_coef(fit, "const"),
        mkt_beta=_coef(fit, "Mkt-RF"),
        smb=_coef(fit, "SMB"),
        hml=_coef(fit, "HML"),
        r_squared=float(fit.rsquared),
        n_obs=int(fit.nobs),
    )
