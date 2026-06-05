"""
Decision logic + volatility-targeted sizing.
Produces daily target allocations: {ticker: weight} summing to 1.0.
"""

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np
from tqqq_bvt.regime import BULL, BEAR
from tqqq_bvt.config import (
    OVERBOUGHT_RSI, TQQQ_OVERSOLD_RSI, SPY_OVERSOLD_RSI,
    TARGET_VOL, REBAL_THRESHOLD,
)


@dataclass
class StrategyParams:
    overbought_rsi: int = OVERBOUGHT_RSI
    tqqq_oversold_rsi: int = TQQQ_OVERSOLD_RSI
    spy_oversold_rsi: int = SPY_OVERSOLD_RSI
    target_vol: float | None = TARGET_VOL   # None = VT-OFF
    rebal_threshold: float = REBAL_THRESHOLD
    bond: str = "IEF"
    use_uvxy_overbought: bool = False  # S1 structural test only


def pick_target(regime: str,
                qqq_rsi: float,
                spy_rsi: float,
                qqq_above_20sma: bool,
                sqqq_rsi10: float,
                bond_rsi10: float,
                bond: str,
                params: StrategyParams) -> tuple[str, bool]:
    """
    Return (target_ticker, risk_on).
    risk_on=True → vol targeting applied to sizing.
    risk_on=False → hold at full weight (1.0), no vol scaling.
    """
    if regime == BULL:
        if qqq_rsi >= params.overbought_rsi:
            if params.use_uvxy_overbought:
                return "UVXY", False
            return "BIL", False
        return "TQQQ", True

    # BEAR regime
    if qqq_rsi <= params.tqqq_oversold_rsi:
        return "TQQQ", True
    if spy_rsi <= params.spy_oversold_rsi:
        return "UPRO", True
    if not qqq_above_20sma:
        # pick SQQQ or bond by higher 10D RSI
        if sqqq_rsi10 >= bond_rsi10:
            return "SQQQ", False
        return bond, False
    # Transitional: de-risk to cash
    return "BIL", False


def compute_allocations(signals: pd.DataFrame,
                        regime: pd.Series,
                        params: StrategyParams) -> pd.DataFrame:
    """
    Walk forward day by day and compute target allocations.
    Returns DataFrame with columns:
      target, risk_on, vol_weight, alloc_<target>, alloc_BIL, trigger
    Trigger is 'SIGNAL_CHANGE' or 'VOL_RESIZE' or 'INITIAL'.

    NOTE: allocations computed at close D; execution at open D+1 is handled
    by the backtest engine (shift by 1).
    """
    records = []
    prev_target: str | None = None
    prev_weight: float | None = None

    bond = params.bond
    bond_rsi_col = f"{bond.lower()}_rsi10"

    for date, row in signals.iterrows():
        r = regime.loc[date]
        if r is None or pd.isna(row["qqq_rsi"]) or pd.isna(row["spy_rsi"]):
            records.append({
                "date": date, "target": "BIL", "risk_on": False,
                "vol_weight": 1.0, "trigger": "WARMUP",
            })
            continue

        target, risk_on = pick_target(
            regime=r,
            qqq_rsi=row["qqq_rsi"],
            spy_rsi=row["spy_rsi"],
            qqq_above_20sma=bool(row["qqq_above_20sma"]),
            sqqq_rsi10=row["sqqq_rsi10"],
            bond_rsi10=row[bond_rsi_col],
            bond=bond,
            params=params,
        )

        # Vol targeting
        if risk_on and params.target_vol is not None:
            rv = row["realized_vol_qqq"]
            if pd.isna(rv) or rv <= 0:
                weight = 1.0
            else:
                weight = float(np.clip(params.target_vol / rv, 0.0, 1.0))
        else:
            weight = 1.0

        # Determine trigger
        if prev_target is None:
            trigger = "INITIAL"
        elif target != prev_target:
            trigger = "SIGNAL_CHANGE"
        elif abs(weight - (prev_weight or 0.0)) >= params.rebal_threshold:
            trigger = "VOL_RESIZE"
        else:
            trigger = "HOLD"

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
