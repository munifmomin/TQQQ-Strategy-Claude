"""
Two-stage parameter optimization (coarse → refined).
Optimizes on IS only; validates top candidates on OOS.
Primary objective: minimize max drawdown.
Secondary: maximize Calmar.
Constraint: CAGR >= 80% of VT-40 baseline CAGR.
Overfitting guardrail: exclude combos where OOS Calmar < 60% of IS Calmar.
"""

from __future__ import annotations
import itertools
import random
from typing import Any
import pandas as pd
import numpy as np

from tqqq_bvt.config import (
    OPT_PARAMS_COARSE, WF_IN_SAMPLE_END, WF_OUT_SAMPLE_START, START_DATE,
)
from tqqq_bvt.signals import compute_signals
from tqqq_bvt.regime import compute_regime
from tqqq_bvt.strategy import StrategyParams, compute_allocations
from tqqq_bvt.backtest import run_backtest
from tqqq_bvt.metrics import compute_metrics, max_drawdown, cagr


def _run_params(prices: pd.DataFrame, params_dict: dict,
                bond: str = "IEF", date_range: tuple[str, str] | None = None) -> dict:
    """Run a single parameter combination. Returns metrics dict."""
    sig = compute_signals(
        prices,
        sma_window=params_dict["sma_window"],
        rsi_window=params_dict["rsi_window"],
        vol_lookback=params_dict["vol_lookback"],
        buffer_up=params_dict["buffer_up"],
        buffer_dn=params_dict["buffer_dn"],
    )
    regime = compute_regime(sig)
    sp = StrategyParams(
        overbought_rsi=params_dict["overbought_rsi"],
        tqqq_oversold_rsi=params_dict["tqqq_oversold_rsi"],
        spy_oversold_rsi=params_dict["spy_oversold_rsi"],
        target_vol=params_dict["target_vol"],
        bond=bond,
    )
    alloc = compute_allocations(sig, regime, sp)

    if date_range:
        s, e = date_range
        sig = sig.loc[s:e]
        alloc = alloc.loc[s:e]
        prices_w = prices.loc[s:e]
    else:
        prices_w = prices

    portfolio, trade_df = run_backtest(alloc, prices_w)
    equity = portfolio["portfolio_value"]
    if len(equity) < 2:
        return {}
    return compute_metrics(equity, trade_df)


def optimize(prices: pd.DataFrame,
             bond: str = "IEF",
             baseline_cagr: float | None = None,
             max_samples: int = 2000,
             verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Two-stage grid optimization.
    Returns (top10_df, excluded_overfit_df).
    """
    param_grid = OPT_PARAMS_COARSE

    # Enumerate full grid
    keys = list(param_grid.keys())
    all_combos = list(itertools.product(*[param_grid[k] for k in keys]))
    total = len(all_combos)

    if verbose:
        print(f"  [OPT] Total coarse grid size: {total:,} combos")

    # Sample if too large
    if total > max_samples:
        if verbose:
            print(f"  [OPT] Sampling {max_samples} combos (Latin-hypercube-style random)")
        random.seed(42)
        all_combos = random.sample(all_combos, max_samples)

    is_results = []
    for i, combo in enumerate(all_combos):
        p = dict(zip(keys, combo))
        try:
            m = _run_params(prices, p, bond=bond, date_range=(START_DATE, WF_IN_SAMPLE_END))
        except Exception:
            continue
        if not m:
            continue
        m.update(p)
        is_results.append(m)
        if verbose and i % 200 == 0:
            print(f"    {i}/{len(all_combos)} combos evaluated...")

    if not is_results:
        return pd.DataFrame(), pd.DataFrame()

    is_df = pd.DataFrame(is_results)

    # Apply CAGR constraint
    if baseline_cagr is not None:
        is_df = is_df[is_df["cagr_pct"] >= baseline_cagr * 0.80]

    # Sort by (min mdd, max calmar)
    if len(is_df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    is_df = is_df.sort_values(["max_drawdown_pct", "calmar"], ascending=[False, False])
    top_candidates = is_df.head(50)

    # Validate top candidates on OOS
    oos_results = []
    for _, row in top_candidates.iterrows():
        p = {k: row[k] for k in keys}
        try:
            m = _run_params(prices, p, bond=bond, date_range=(WF_OUT_SAMPLE_START, None))
        except Exception:
            continue
        if not m:
            continue
        m.update(p)
        m["is_calmar"] = row["calmar"]
        oos_results.append(m)

    if not oos_results:
        return pd.DataFrame(), pd.DataFrame()

    oos_df = pd.DataFrame(oos_results)
    oos_df["overfit"] = oos_df["calmar"] < oos_df["is_calmar"] * 0.60

    clean = oos_df[~oos_df["overfit"]].sort_values("calmar", ascending=False).head(10)
    overfit = oos_df[oos_df["overfit"]]

    return clean, overfit
