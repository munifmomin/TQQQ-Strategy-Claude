"""
Faithful FTLT-V2 (80/20) comparison arm.

Reproduces the ORIGINAL "TQQQ For The Long Term" rotation rules exactly
(raw 200D SMA crossover, UVXY on overbought, oversold bounces, SQQQ/bond
defensive), sized 80% signal instrument / 20% cash (BIL) at all times.
This was the prior run's best risk-adjusted sweet spot
(~72.6% CAGR, -53% MDD, Calmar 1.36, ~20 trades/yr).

Runs through the same backtest engine and capital sim as the R2 variants
for a true side-by-side $1,000 comparison.
"""

from __future__ import annotations
import pandas as pd

from tqqq_r2.config import (
    OVERBOUGHT_RSI, TQQQ_OVERSOLD_RSI, SPY_OVERSOLD_RSI,
    FTLT_V2_RISK_WEIGHT, FTLT_V2_CASH_WEIGHT,
)


def pick_target_ftlt(spy_above_200: bool,
                     qqq_rsi: float,
                     spy_rsi: float,
                     qqq_above_20sma: bool,
                     sqqq_rsi10: float,
                     bond_rsi10: float,
                     bond: str,
                     uvxy_available: bool) -> str:
    """Original FTLT decision tree. Returns the target ticker."""
    if spy_above_200:
        if qqq_rsi >= OVERBOUGHT_RSI:
            return "UVXY" if uvxy_available else "BIL"
        return "TQQQ"

    # Bearish regime
    if qqq_rsi <= TQQQ_OVERSOLD_RSI:
        return "TQQQ"
    if spy_rsi <= SPY_OVERSOLD_RSI:
        return "UPRO"
    if not qqq_above_20sma:
        if sqqq_rsi10 >= bond_rsi10:
            return "SQQQ"
        return bond
    # Transitional: original holds TQQQ here
    return "TQQQ"


def compute_allocations_ftlt(prices: pd.DataFrame,
                             signals: pd.DataFrame,
                             bond: str = "TLT") -> pd.DataFrame:
    """
    Compute FTLT-V2 (80/20) allocations.
    Returns DataFrame with columns: target, vol_weight (always 0.80), trigger.
    """
    bond_rsi_col = f"{bond.lower()}_rsi10"
    uvxy_available = "UVXY" in prices.columns

    records = []
    prev_target: str | None = None

    for date, row in signals.iterrows():
        sa200 = row["spy_above_200"]
        if pd.isna(sa200) or pd.isna(row["qqq_rsi"]) or pd.isna(row["spy_rsi"]):
            records.append({"date": date, "target": "BIL", "vol_weight": 1.0, "trigger": "WARMUP"})
            continue

        target = pick_target_ftlt(
            spy_above_200=bool(sa200),
            qqq_rsi=row["qqq_rsi"],
            spy_rsi=row["spy_rsi"],
            qqq_above_20sma=bool(row["qqq_above_20sma"]),
            sqqq_rsi10=row["sqqq_rsi10"],
            bond_rsi10=row[bond_rsi_col],
            bond=bond,
            uvxy_available=uvxy_available,
        )

        weight = FTLT_V2_RISK_WEIGHT  # always 80% target / 20% BIL

        if prev_target is None:
            trigger = "INITIAL"
        elif target != prev_target:
            trigger = "SIGNAL_CHANGE"
        else:
            trigger = "HOLD"

        records.append({"date": date, "target": target, "vol_weight": weight, "trigger": trigger})

        if trigger in ("INITIAL", "SIGNAL_CHANGE"):
            prev_target = target

    return pd.DataFrame(records).set_index("date")
