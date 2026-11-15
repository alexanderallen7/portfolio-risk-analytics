# portfolio-risk-analytics

A portfolio risk and performance analytics pipeline. It pulls five years of
daily prices for a fixed 15-stock US large-cap portfolio from free, keyless
public sources, computes a standard set of risk and performance metrics,
runs CAPM and Fama-French 3-factor regressions, and renders a self-contained
HTML tear sheet with publication-quality charts.

**What this is:** a static risk and performance analysis of a fixed,
equal-weighted portfolio, in the format a risk or performance team would
produce for an existing book.

**What this isn't:** a backtested trading strategy, a portfolio
recommendation, or a forecast. The holdings were chosen in hindsight for
sector diversification, so the performance statistics carry survivorship
and selection bias. Both the Limitations section below and the report
itself state this.

## Portfolio

Equal-weighted (rebalanced daily): AAPL, MSFT, JPM, JNJ, PG, XOM, UNH, HD,
V, PEP, KO, DIS, CAT, NEE, COST. Benchmark: SPY. Lookback: 5 years ending
at the most recent available trading day. The report states the exact
window used.

## Setup and run

```
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python scripts/fetch_data.py    # one-time pull; caches everything to data/raw/
python scripts/run_analysis.py  # offline: computes metrics, renders output/
pytest                          # unit tests for the metric/regression functions
```

`fetch_data.py` is the only step that touches the network. Every
subsequent `run_analysis.py` reads exclusively from `data/raw/`, so the
analysis is fully reproducible offline. To force a re-pull:
`python scripts/fetch_data.py --refresh`.

## Data sources (all free, no API keys)

| Series | Primary source | Fallback chain |
|---|---|---|
| Daily adjusted closes (15 holdings + SPY) | Yahoo Finance via `yfinance` (dividend- and split-adjusted) | per-ticker yfinance retry, then stooq.com CSV (split-adjusted only; flagged in the report if used) |
| Risk-free rate | FRED series `DGS3MO` (3-month Treasury CMT yield) via `pandas-datareader` | FRED fredgraph CSV endpoint, then a hardcoded constant (loudly flagged in the report if used) |
| Fama-French daily 3-factor returns | Ken French Data Library (Dartmouth Tuck) via `pandas-datareader` | direct zip download, then an Internet Archive snapshot of the same official zip |

All pulls use retry-with-backoff. The report's **Data Provenance** table
records which source supplied each series on the run that produced it,
including any fallbacks that triggered.

## Output

`scripts/run_analysis.py` writes to `output/`:

- `tear_sheet.html` is a self-contained one-to-two page report with the
  charts embedded. It covers performance and risk metrics for the
  portfolio and SPY, historical and Gaussian VaR at 95%/99%, max-drawdown
  dates, CAPM and FF3 regression tables with t-stats and p-values, a
  written interpretation generated from the computed numbers, a
  methodology and limitations section, and the data provenance table.
- `cumulative_return.png`, `drawdown.png`, `rolling_volatility.png`, and
  `correlation_heatmap.png` are the four charts as standalone files.

From the run on the five-year window ending Nov 13, 2026 (1,254 trading
days): the equal-weight portfolio returned 13.4% annualized at 13.6% vol
(Sharpe 0.70) vs. SPY's 11.6% at 17.3% (Sharpe 0.49), with a max drawdown
of −15.3% (Jan 2022 to Jun 2022, recovered Apr 2023) vs. −24.5% for SPY.
CAPM beta is 0.64. The CAPM alpha of about 4% a year has a p-value of
0.25, and in the Fama-French regression it shrinks to about 0.4% a year
(p = 0.91); the three-factor fit attributes the extra return over SPY to
defensive size and value tilts. Your numbers will differ slightly because
the window end moves with the run date.

## Methodology and limitations (summary)

The full version is in the tear sheet; headlines:

- **Survivorship and selection bias.** The 15 names are currently listed,
  successful large caps chosen in hindsight. Performance statistics are
  biased upward relative to any real ex-ante selection. The bias comes
  from how the names were picked, so better price data cannot remove it.
- **Adjusted close approximates total return.** Yahoo adjusted closes
  model dividend reinvestment at the ex-date close and exclude taxes and
  trading costs. A stooq-fallback ticker would also lack dividend
  adjustment; the provenance table records whether this happened (on the
  committed run it did not).
- **Daily rebalancing costs nothing in this model.** Portfolio return is
  the cross-sectional mean of constituent returns each day, which assumes
  zero transaction costs, taxes, and spread. Equal weighting is a stated
  choice; it drives part of the deviation from cap-weighted SPY.
- **VaR is unconditional and backward-looking.** Historical VaR assumes
  the past five years' distribution is representative. Gaussian VaR
  understates equity tail risk by construction.
- **Window sensitivity.** This is one five-year window containing one
  bear market (2022) and a strong recovery. Sharpe, alpha, and drawdown
  figures move materially if the window shifts. None of these figures is
  a forecast.
- **FF factor timing.** The Ken French library posts with a lag, and the
  committed run used an Internet Archive snapshot of the official file
  (snapshot date recorded in the provenance table), so the factor
  regression sample ends slightly before the price window. The report
  prints the exact n used.

## Repo structure

```
data/raw/        cached pulls (gitignored; recreated by fetch_data.py)
data/processed/  reserved for derived datasets
src/             config, data fetching, pure metric functions,
                 regressions, charts, report rendering (+ HTML template)
scripts/         fetch_data.py, run_analysis.py
tests/           pytest suite: metric functions verified against
                 hand-computed values on synthetic series
output/          rendered tear sheet + charts (gitignored)
```

## Tests

`pytest` runs 23 unit tests. The metric tests verify against values
computed by hand on synthetic series, for example a known price path with
an exact −20% drawdown and known peak/trough/recovery dates, and VaR on
evenly spaced returns where the percentile falls exactly on a data point.
The regression tests recover exact coefficients from noise-free
constructions and check that inference behaves sensibly on seeded noisy
data. All random draws use fixed seeds.
