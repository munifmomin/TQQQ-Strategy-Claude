"""
Decision logic (stateless baseline) + CORRECTED volatility-targeted sizing.

Key fix vs the prior spec: vol targeting weight uses TQQQ's own realized vol
(tqqq_realized_vol / tqqq_realized_vol_ewma from signals.py), not QQQ's.

Produces daily target allocations: {ticker: weight} summing to 1.0.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from tqqq_r2.config import (
    OVERBOUGHT_RSI, TQQQ_OVERSOLD_RSI, SPY_OVERSOLD_RSI,
    TARGET_VOL, REBAL_THRESHOLD, LIGHT_BUFFER_UP, LIGHT_BUFFER_DN,
)
from tqqq_r2.regime import compute_buffered_regime, BULL, BEAR


@dataclass
class StrategyParams:
    overbought_rsi: int = OVERBOUGHT_RSI
    tqqq_oversold_rsi: int = TQQQ_OVERSOLD_RSI
    spy_oversold_rsi: int = SPY_OVERSOLD_RSI
    target_vol: float | None = TARGET_VOL   # None = R2-NoVT (binary 100/0)
    rebal_threshold: float = REBAL_THRESHOLD
    bond: str = "TLT"

    # Structural test toggles
    use_ewma_vol: bool = False                 # T1
    target_vol_bull: float | None = None       # T2 (overrides target_vol in BULL)
    target_vol_bear: float | None = None       # T2 (overrides target_vol in BEAR)
    apply_vol_to_defensive: bool = False       # T4
    vix_hard_filter: float | None = None       # T3 (e.g. 35.0)

    # Optional buffer variant (Section 2.6, NOT baseline)
    buffer_up: float = 0.0
    buffer_dn: float = 0.0


def pick_target(spy_above_200: bool,
                qqq_rsi: float,
                spy_rsi: float,
                qqq_above_20sma: bool,
                sqqq_rsi10: float,
                bond_rsi10: float,
                bond: str,
                params: StrategyParams) -> tuple[str, bool]:
    """
    Return (target_ticker, risk_on).
    risk_on=True -> vol targeting applied to sizing (subject to apply_vol_to_defensive).
    """
    if spy_above_200:
        if qqq_rsi >= params.overbought_rsi:
            return "BIL", False
        return "TQQQ", True

    # Bearish regime
    if qqq_rsi <= params.tqqq_oversold_rsi:
        return "TQQQ", True
    if spy_rsi <= params.spy_oversold_rsi:
        return "UPRO", True
    if not qqq_above_20sma:
        if sqqq_rsi10 >= bond_rsi10:
            return "SQQQ", False
        return bond, False
    # Transitional: hold TQQQ (validated in prior run's S4 test)
    return "TQQQ", True


def _vol_weight(rv: float, target_vol: float | None) -> float:
    if target_vol is None:
        return 1.0
    if pd.isna(rv) or rv <= 0:
        return 1.0
    return float(np.clip(target_vol / rv, 0.0, 1.0))


def compute_allocations(prices: pd.DataFrame,
                        signals: pd.DataFrame,
                        params: StrategyParams,
                        vix: pd.Series | None = None) -> pd.DataFrame:
    """
    Walk forward day by day and compute target allocations.
    Returns DataFrame with columns:
      target, risk_on, vol_weight, trigger
    Trigger is 'SIGNAL_CHANGE', 'VOL_RESIZE', 'INITIAL', 'WARMUP', or 'HOLD'.

    NOTE: allocations computed at close D; execution at open D+1 is handled
    by the backtest engine.
    """
    bond = params.bond
    bond_rsi_col = f"{bond.lower()}_rsi10"

    # Determine the regime series. Baseline (buffer_up=buffer_dn=0) -> raw crossover.
    if params.buffer_up == 0 and params.buffer_dn == 0:
        spy_above_200 = signals["spy_above_200"]
        regime_series = spy_above_200.map(lambda x: BULL if x else (BEAR if x is False else None))
    else:
        spy = prices["SPY"]
        regime_series = compute_buffered_regime(spy, signals["spy_sma"], params.buffer_up, params.buffer_dn)
        spy_above_200 = regime_series.map(lambda r: r == BULL if r is not None else None)

    vol_col = "tqqq_realized_vol_ewma" if params.use_ewma_vol else "tqqq_realized_vol"

    records = []
    prev_target: str | None = None
    prev_weight: float | None = None

    for date, row in signals.iterrows():
        sa200 = spy_above_200.loc[date]
        if sa200 is None or pd.isna(row["qqq_rsi"]) or pd.isna(row["spy_rsi"]) or pd.isna(row[vol_col]):
            records.append({
                "date": date, "target": "BIL", "risk_on": False,
                "vol_weight": 1.0, "trigger": "WARMUP",
            })
            continue

        # T3: VIX hard filter overrides everything
        if params.vix_hard_filter is not None and vix is not None and date in vix.index:
            v = vix.loc[date]
            if not pd.isna(v) and v > params.vix_hard_filter:
                target, risk_on = "BIL", False
                weight = 1.0
                trigger = _trigger(prev_target, prev_weight, target, weight, params)
                records.append({"date": date, "target": target, "risk_on": risk_on,
                                 "vol_weight": weight, "trigger": trigger})
                if trigger in ("INITIAL", "SIGNAL_CHANGE", "VOL_RESIZE"):
                    prev_target, prev_weight = target, weight
                continue

        target, risk_on = pick_target(
            spy_above_200=bool(sa200),
            qqq_rsi=row["qqq_rsi"],
            spy_rsi=row["spy_rsi"],
            qqq_above_20sma=bool(row["qqq_above_20sma"]),
            sqqq_rsi10=row["sqqq_rsi10"],
            bond_rsi10=row[bond_rsi_col],
            bond=bond,
            params=params,
        )

        rv = row[vol_col]

        # T2: regime-dependent target vol
        if params.target_vol_bull is not None or params.target_vol_bear is not None:
            tv = params.target_vol_bull if sa200 else params.target_vol_bear
        else:
            tv = params.target_vol

        if risk_on:
            weight = _vol_weight(rv, tv)
        elif params.apply_vol_to_defensive:
            # T4: scale defensive leg too
            weight = _vol_weight(rv, tv)
        else:
            weight = 1.0

        trigger = _trigger(prev_target, prev_weight, target, weight, params)

        records.append({
            "date": date,
            "target": target,
            "risk_on": risk_on,
            "vol_weight": weight,
            "trigger": trigger,
        })

        if trigger in ("INITIAL", "SIGNAL_CHANGE", "VOL_RESIZE"):
            prev_target = target
            prev_weight = weight

    alloc = pd.DataFrame(records).set_index("date")
    return alloc


def _trigger(prev_target: str | None, prev_weight: float | None,
             target: str, weight: float, params: StrategyParams) -> str:
    if prev_target is None:
        return "INITIAL"
    if target != prev_target:
        return "SIGNAL_CHANGE"
    if abs(weight - (prev_weight or 0.0)) >= params.rebal_threshold:
        return "VOL_RESIZE"
    return "HOLD"
