"""
Regime determination.

Baseline: raw 200D SMA crossover (stateless) — `spy_above_200` from signals.py
is sufficient and this module isn't needed for the baseline path.

This module exists only for the OPTIONAL R2-VT55-Buf2 side experiment
(Section 2.6): a light ±2%/-2% buffer with hysteresis, reusing the
stateful state-machine pattern validated in the prior (BVT) spec.
"""

from __future__ import annotations
import pandas as pd

BULL = "BULL"
BEAR = "BEAR"


def compute_buffered_regime(spy: pd.Series, spy_sma: pd.Series,
                             buffer_up: float, buffer_dn: float) -> pd.Series:
    """
    Stateful buffered regime with hysteresis.
    If buffer_up == buffer_dn == 0, this degenerates to a raw crossover
    (still stateful, but bands collapse to the SMA itself).
    """
    above_band = spy > spy_sma * (1 + buffer_up)
    below_band = spy < spy_sma * (1 - buffer_dn)

    regime_list: list[str | None] = []
    prior: str | None = None

    for i in range(len(spy)):
        ab = above_band.iloc[i]
        bl = below_band.iloc[i]

        if pd.isna(spy_sma.iloc[i]):
            regime_list.append(None)
            continue

        if ab:
            current = BULL
        elif bl:
            current = BEAR
        else:
            current = prior if prior is not None else BULL

        regime_list.append(current)
        prior = current

    return pd.Series(regime_list, index=spy.index, name="regime", dtype=object)
