"""
Chart generation for all output figures.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqqq_r2.config import OUTPUT_DIR


def _ensure_output():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def dollar_equity_curves(equity_dict: dict[str, pd.Series],
                         filename: str = "dollar_equity_curves.png") -> None:
    """Both strategies + benchmarks on $1,000, linear AND log."""
    _ensure_output()
    fig, axes = plt.subplots(2, 1, figsize=(14, 11))
    styles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

    for i, (label, eq) in enumerate(equity_dict.items()):
        axes[0].plot(eq.index, eq.values, label=label, linestyle=styles[i % len(styles)], linewidth=1.5)
        axes[1].semilogy(eq.index, eq.values, label=label, linestyle=styles[i % len(styles)], linewidth=1.5)

    axes[0].set_title("Dollar Equity Curves (linear scale, starting $1,000)")
    axes[0].set_ylabel("Account Value ($)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Dollar Equity Curves (log scale)")
    axes[1].set_ylabel("Account Value ($, log)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()


def monthly_dollar_bars(monthly_dict: dict[str, pd.DataFrame],
                        filename: str = "monthly_dollar_bars.png") -> None:
    """Monthly $ change bar chart, green/red, both strategies."""
    _ensure_output()
    n = len(monthly_dict)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (label, df) in zip(axes, monthly_dict.items()):
        colors = ["green" if v >= 0 else "red" for v in df["dollar_change"]]
        ax.bar(df["month"], df["dollar_change"], color=colors)
        ax.set_title(f"{label}: Monthly $ Change")
        ax.set_ylabel("$ Change")
        ax.tick_params(axis="x", rotation=90, labelsize=5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()


def drawdown_chart(equity_dict: dict[str, pd.Series], filename: str = "drawdown_chart.png") -> None:
    _ensure_output()
    fig, ax = plt.subplots(figsize=(14, 6))
    for label, eq in equity_dict.items():
        dd = (eq / eq.cummax() - 1) * 100
        ax.plot(dd.index, dd.values, label=label, linewidth=1.2)
    ax.set_title("Drawdown Over Time")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()


def annual_returns_heatmap(equity_dict: dict[str, pd.Series],
                           filename: str = "annual_returns_heatmap.png") -> None:
    _ensure_output()
    rows = {}
    for label, eq in equity_dict.items():
        yr = eq.resample("YE").last().pct_change().dropna() * 100
        rows[label] = yr

    df = pd.DataFrame(rows)
    df.index = [i.year for i in df.index]

    fig, ax = plt.subplots(figsize=(max(10, len(df.columns) * 1.5), max(6, len(df) * 0.5)))
    im = ax.imshow(df.T, aspect="auto", cmap="RdYlGn", vmin=-60, vmax=60)
    ax.set_xticks(range(len(df.index)))
    ax.set_xticklabels(df.index, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(df.columns)))
    ax.set_yticklabels(df.columns, fontsize=8)
    ax.set_title("Annual Returns Heatmap (%)")
    plt.colorbar(im, ax=ax)

    for i in range(len(df.columns)):
        for j in range(len(df.index)):
            val = df.iloc[j, i]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()


def rolling_metrics_chart(equity_dict: dict[str, pd.Series],
                          filename: str = "rolling_metrics.png") -> None:
    _ensure_output()
    from tqqq_r2.walk_forward import rolling_windows
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for label, eq in equity_dict.items():
        rw = rolling_windows(eq)
        if rw.empty:
            continue
        axes[0].plot(rw["window_end"], rw["cagr_pct"], label=label, linewidth=1.2)
        axes[1].plot(rw["window_end"], rw["max_drawdown_pct"], label=label, linewidth=1.2)

    axes[0].set_title("Rolling 3-Year CAGR (%)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Rolling 3-Year Max Drawdown (%)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()


def vol_targeting_diagnostic(signals: pd.DataFrame, allocations: pd.DataFrame,
                              filename: str = "vol_targeting_diagnostic.png") -> None:
    """
    Realized TQQQ vol vs applied weight. MUST show weight varying below 1.0
    in high-vol periods (2018-Q4, 2020, 2022) -- proves the vol-on-TQQQ fix.
    """
    _ensure_output()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    rv = signals["tqqq_realized_vol"].dropna()
    ax1.plot(rv.index, rv * 100, color="steelblue", linewidth=1.0, label="Realized TQQQ Vol (ann.)")
    ax1.set_ylabel("Annualized Vol (%)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    w = allocations["vol_weight"].dropna()
    ax2.plot(w.index, w, color="darkorange", linewidth=1.0, label="Applied Weight")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Vol Target Weight")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax1.set_title("Vol Targeting Diagnostic: Realized TQQQ Vol vs Applied Weight")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
