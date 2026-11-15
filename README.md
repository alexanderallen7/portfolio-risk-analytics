# portfolio-risk-analytics

A portfolio risk and performance analytics pipeline: it pulls five years of
daily prices for a fixed 15-stock US large-cap portfolio from free, keyless
public sources, computes a standard set of risk and performance metrics,
runs CAPM and Fama-French 3-factor regressions, and renders a self-contained
HTML tear sheet with publication-quality charts.

**What this is:** a static risk and performance analysis of a fixed,
equal-weighted portfolio — the kind of tear sheet a risk or performance
team produces for an existing book.

**What this isn't:** a backtested trading strategy, a portfolio
recommendation, or a forecast. The holdings were chosen in hindsight for
sector diversification, so the performance statistics carry survivorship
and selection bias (see Limitations below — this is stated in the report
itself, not hidden here).

## Portfolio

Equal-weighted (rebalanced daily): AAPL, MSFT, JPM, JNJ, PG, XOM, UNH, HD,
V, PEP, KO, DIS, CAT, NEE, COST. Benchmark: SPY. Lookback: 5 years ending
at the most recent available trading day (the report states the exact
window used).

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
| Daily adjusted closes (15 holdings + SPY) | Yahoo Finance via `yfinance` (dividend- and split-adjusted) | per-ticker yfinance retry → stooq.com CSV (split-adjusted only; flagged in the report if used) |
| Risk-free rate | FRED series `DGS3MO` (3-month Treasury CMT yield) via `pandas-datareader` | FRED fredgraph CSV endpoint → hardcoded constant (loudly flagged in the report if used) |
| Fama-French daily 3-factor returns | Ken French Data Library (Dartmouth Tuck) via `pandas-datareader` | direct zip download → Internet Archive snapshot of the same official zip |

All pulls use retry-with-backoff. The report's **Data Provenance** table
records which source actually supplied each series on the run that
produced it — the tear sheet reflects what really happened, not the happy
path.

## Output

`scripts/run_analysis.py` writes to `output/`:

- `tear_sheet.html` — a self-contained (charts embedded) one-to-two page
  report: performance and risk metrics for the portfolio and SPY,
  historical and Gaussian VaR at 95%/99%, max-drawdown dates, CAPM and
  FF3 regression tables with t-stats and p-values, a written
  interpretation generated from the computed numbers, a methodology and
  limitations section, and the data provenance table.
- `cumulative_return.png`, `drawdown.png`, `rolling_volatility.png`,
  `correlation_heatmap.png` — the standalone charts.

From the run on the five-year window ending Nov 13, 2026 (1,254 trading
days): the equal-weight portfolio returned 13.4% annualized at 13.6% vol
(Sharpe 0.70) vs. SPY's 11.6% at 17.3% (Sharpe 0.49), with a max drawdown
of −15.3% (Jan 2022 → Jun 2022, recovered Apr 2023) vs. −24.5% for SPY.
CAPM beta 0.64; the CAPM alpha of ~4% a year is not statistically
significant (p ≈ 0.25) and shrinks to ~0.4% (p ≈ 0.91) once the
Fama-French size and value factors are controlled for — i.e., the
"outperformance" is factor tilts, not selection skill. Your numbers will
differ slightly: the window end moves with the run date.

## Methodology and limitations (summary)

The full, specific version is in the tear sheet; headlines:

- **Survivorship/selection bias.** The 15 names are currently listed,
  successful large caps chosen in hindsight. Performance stats are biased
  upward relative to any real ex-ante selection. Structural to the
  exercise; not correctable with better data.
- **Adjusted close ≠ true total return.** Yahoo adjusted closes
  approximate dividend reinvestment at the ex-date close, ignore taxes
  and costs. Any stooq-fallback ticker lacks dividend adjustment entirely
  (the provenance table says if this happened; on the committed run, none
  did).
- **Daily rebalancing is free here, not in reality.** Portfolio return is
  the cross-sectional mean of constituent returns each day: zero
  transaction costs, taxes, or spreads. Equal weighting is a deliberate
  stated choice; it drives part of the deviation from cap-weighted SPY.
- **VaR is unconditional and backward-looking.** Historical VaR assumes
  the past five years' distribution is representative; Gaussian VaR
  understates equity tail risk by construction.
- **Window sensitivity.** One five-year window containing one bear market
  (2022) and a strong recovery. Sharpe, alpha, and drawdown figures move
  materially if the window shifts; nothing here is a forecast.
- **FF factor timing.** The Ken French library posts with a lag (and the
  committed run used an Internet Archive snapshot of the official file,
  dated in the provenance table), so the factor regression sample ends
  slightly before the price window; the report prints the exact n used.

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
computed by hand on synthetic series (e.g., a known price path with an
exact −20% drawdown and known peak/trough/recovery dates; VaR on evenly
spaced returns where the percentile is exact). The regression tests
recover exact coefficients from noise-free constructions and check
inference behaves sensibly on seeded noisy data. Randomness is always
seeded; everything is reproducible.
