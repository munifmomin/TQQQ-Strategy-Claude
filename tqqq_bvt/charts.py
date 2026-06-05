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
import matplotlib.ticker as mticker
from tqqq_bvt.config import OUTPUT_DIR


def _ensure_output():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def equity_curves(equity_dict: dict[str, pd.Series], filename: str = "equity_curves.png") -> None:
    _ensure_output()
    fig, ax = plt.subplots(figsize=(14, 7))
    styles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
    for i, (label, eq) in enumerate(equity_dict.items()):
        eq_norm = eq / eq.iloc[0] * 100
        ax.semilogy(eq_norm.index, eq_norm.values, label=label, linestyle=styles[i % len(styles)], linewidth=1.5)
    ax.set_title("Equity Curves (log scale, rebased to 100)")
    ax.set_ylabel("Portfolio Value (rebased)")
    ax.legend(fontsize=8)
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
    from tqqq_bvt.walk_forward import rolling_windows
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
    _ensure_output()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    rv = signals["realized_vol_qqq"].dropna()
    ax1.plot(rv.index, rv * 100, color="steelblue", linewidth=1.0, label="Realized Vol QQQ (ann.)")
    ax1.set_ylabel("Annualized Vol (%)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    w = allocations["vol_weight"].dropna()
    ax2.plot(w.index, w, color="darkorange", linewidth=1.0, label="Applied Weight")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Vol Target Weight")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax1.set_title("Vol Targeting Diagnostic: Realized Vol vs Applied Weight")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()


def optimization_surface(opt_results: pd.DataFrame,
                         filename: str = "optimization_surface.png") -> None:
    _ensure_output()
    if opt_results.empty or "calmar" not in opt_results.columns:
        return

    params_2d = [("buffer_up", "buffer_dn"), ("target_vol", "vol_lookback")]
    fig, axes = plt.subplots(1, len(params_2d), figsize=(14, 5))

    for ax, (px, py) in zip(axes, params_2d):
        if px not in opt_results.columns or py not in opt_results.columns:
            continue
        pivot = opt_results.pivot_table(values="calmar", index=py, columns=px, aggfunc="max")
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", origin="lower")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns], rotation=45, fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.2f}" for v in pivot.index], fontsize=7)
        ax.set_xlabel(px)
        ax.set_ylabel(py)
        ax.set_title(f"Calmar: {px} vs {py}")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
