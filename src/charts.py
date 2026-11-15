"""Chart generation: cumulative return, drawdown, rolling vol, correlation.

All charts share one style (clean grid, labeled axes, legends) and are
written as PNGs to the output directory. Each function returns the path
it wrote so the report layer can embed the images.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering; no display needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#: Shared figure styling: muted professional palette on a light grid.
PORTFOLIO_COLOR = "#1f4e79"
BENCHMARK_COLOR = "#c0504d"
FIGSIZE_WIDE = (10.0, 4.8)
DPI = 150

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cccccc",
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "legend.frameon": False,
    }
)


def _save(fig: plt.Figure, path: Path) -> Path:
    """Tight-layout save and close."""
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def cumulative_return_chart(
    port_returns: pd.Series, bench_returns: pd.Series, out_dir: Path
) -> Path:
    """Growth of $1: portfolio vs. benchmark."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    (1 + port_returns).cumprod().plot(ax=ax, color=PORTFOLIO_COLOR, lw=1.6, label="Portfolio (equal-weight)")
    (1 + bench_returns).cumprod().plot(ax=ax, color=BENCHMARK_COLOR, lw=1.4, label="SPY")
    ax.set_title("Cumulative Return — Growth of $1")
    ax.set_xlabel("")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left")
    return _save(fig, out_dir / "cumulative_return.png")


def drawdown_chart(port_dd: pd.Series, bench_dd: pd.Series, out_dir: Path) -> Path:
    """Drawdown paths for portfolio and benchmark."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.fill_between(port_dd.index, port_dd.values * 100, 0, color=PORTFOLIO_COLOR, alpha=0.35)
    ax.plot(port_dd.index, port_dd.values * 100, color=PORTFOLIO_COLOR, lw=1.2, label="Portfolio")
    ax.plot(bench_dd.index, bench_dd.values * 100, color=BENCHMARK_COLOR, lw=1.2, label="SPY")
    ax.set_title("Drawdown from Running Peak")
    ax.set_xlabel("")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower right")
    return _save(fig, out_dir / "drawdown.png")


def rolling_vol_chart(
    port_vol: pd.Series, bench_vol: pd.Series, window: int, out_dir: Path
) -> Path:
    """Rolling annualized volatility, portfolio vs. benchmark."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    port_vol.mul(100).plot(ax=ax, color=PORTFOLIO_COLOR, lw=1.4, label="Portfolio")
    bench_vol.mul(100).plot(ax=ax, color=BENCHMARK_COLOR, lw=1.2, label="SPY")
    ax.set_title(f"Rolling {window}-Day Annualized Volatility")
    ax.set_xlabel("")
    ax.set_ylabel("Annualized volatility (%)")
    ax.legend(loc="upper right")
    return _save(fig, out_dir / "rolling_volatility.png")


def correlation_heatmap(corr: pd.DataFrame, out_dir: Path) -> Path:
    """Correlation heatmap of the holdings' daily returns."""
    n = len(corr)
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(n), labels=corr.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(n), labels=corr.index)
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=7,
                color="white" if abs(val) > 0.6 else "#333333",
            )
    ax.set_title("Correlation Matrix — Daily Returns of Holdings")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson correlation")
    return _save(fig, out_dir / "correlation_heatmap.png")
