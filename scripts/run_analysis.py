"""End-to-end analysis: reads the cache, computes everything, renders output.

Usage, from the repo root (after ``scripts/fetch_data.py`` has run once):

    python scripts/run_analysis.py

Reads ONLY from ``data/raw/`` (no network), writes charts and the tear
sheet to ``output/``, and prints the headline metrics to the console.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import charts, config  # noqa: E402
from src.analysis import run_analysis  # noqa: E402
from src.report import render_tear_sheet  # noqa: E402


def main() -> int:
    """Run the full offline pipeline and print a console summary."""
    missing = [p.name for p in (config.PRICES_CACHE, config.FF3_CACHE) if not p.exists()]
    if missing:
        # First run from a clean environment: pull the data, then proceed,
        # so the whole pipeline works with this one command.
        print(f"Cache incomplete (missing {missing}); fetching data first ...")
        from src.data_fetch import fetch_and_cache_all

        fetch_and_cache_all()

    print("Computing metrics and regressions from cache ...")
    results = run_analysis()

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Rendering charts ...")
    chart_paths = {
        "cumulative": charts.cumulative_return_chart(
            results.port_returns, results.bench_returns, config.OUTPUT_DIR
        ),
        "drawdown": charts.drawdown_chart(
            results.port_dd.series, results.bench_dd.series, config.OUTPUT_DIR
        ),
        "rolling_vol": charts.rolling_vol_chart(
            results.port_roll_vol, results.bench_roll_vol,
            config.ROLLING_VOL_WINDOW, config.OUTPUT_DIR,
        ),
        "correlation": charts.correlation_heatmap(results.corr, config.OUTPUT_DIR),
    }

    tear_sheet = render_tear_sheet(results, chart_paths, config.OUTPUT_DIR / "tear_sheet.html")

    p, b = results.stats["portfolio"], results.stats["benchmark"]
    print(f"\nWindow: {results.start_date:%Y-%m-%d} to {results.end_date:%Y-%m-%d} "
          f"({results.n_days} trading days)")
    print(f"Risk-free (avg DGS3MO): {results.rf_annual_mean:.2%}"
          + ("  [FALLBACK CONSTANT]" if results.rf_is_fallback else ""))
    print(f"{'Metric':<28}{'Portfolio':>12}{'SPY':>12}")
    rows = [
        ("Total return", f"{p['total_return']:.1%}", f"{b['total_return']:.1%}"),
        ("Annualized return", f"{p['annualized_return']:.2%}", f"{b['annualized_return']:.2%}"),
        ("Annualized volatility", f"{p['annualized_vol']:.2%}", f"{b['annualized_vol']:.2%}"),
        ("Sharpe", f"{p['sharpe']:.2f}", f"{b['sharpe']:.2f}"),
        ("Sortino", f"{p['sortino']:.2f}", f"{b['sortino']:.2f}"),
        ("Max drawdown", f"{results.port_dd.max_drawdown:.1%}", f"{results.bench_dd.max_drawdown:.1%}"),
        ("Hist VaR 95% / 99%", f"{p['hist_var_95']:.2%} / {p['hist_var_99']:.2%}",
         f"{b['hist_var_95']:.2%} / {b['hist_var_99']:.2%}"),
        ("Param VaR 95% / 99%", f"{p['param_var_95']:.2%} / {p['param_var_99']:.2%}",
         f"{b['param_var_95']:.2%} / {b['param_var_99']:.2%}"),
    ]
    for name, pv, bv in rows:
        print(f"{name:<28}{pv:>12}{bv:>12}")
    print(f"\nCAPM:  beta={results.capm.beta.estimate:.3f}  "
          f"alpha(ann)={results.capm.alpha_annualized:.2%} "
          f"(p={results.capm.alpha_daily.p_value:.3f})  R2={results.capm.r_squared:.3f}")
    print(f"FF3:   alpha(ann)={results.ff3.alpha_annualized:.2%} "
          f"(p={results.ff3.alpha_daily.p_value:.3f})  "
          f"mkt={results.ff3.mkt_beta.estimate:.3f}  smb={results.ff3.smb.estimate:.3f}  "
          f"hml={results.ff3.hml.estimate:.3f}  R2={results.ff3.r_squared:.3f}")
    print(f"\nTear sheet: {tear_sheet}")
    for name, path in chart_paths.items():
        print(f"Chart ({name}): {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
