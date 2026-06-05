"""
Stateful buffered regime machine.

The buffer creates hysteresis: between the upper/lower bands lies a dead zone
where the regime does NOT change — it holds the previous state.
This is the #1 correctness risk; every day must carry prior_regime forward.
"""

from __future__ import annotations
import pandas as pd


BULL = "BULL"
BEAR = "BEAR"


def compute_regime(signals: pd.DataFrame) -> pd.Series:
    """
    Walk the signal DataFrame day by day, carrying prior_regime through the dead zone.
    Returns a Series of 'BULL' / 'BEAR' aligned to signals.index.

    Initialization: first valid day → BULL if spy_above_band, else BEAR.
    """
    regime_list: list[str | None] = []
    prior: str | None = None

    above = signals["spy_above_band"]
    below = signals["spy_below_band"]

    for i in range(len(signals)):
        ab = above.iloc[i]
        bl = below.iloc[i]

        if pd.isna(ab) or pd.isna(bl):
            regime_list.append(None)
            continue

        if ab:
            current = BULL
        elif bl:
            current = BEAR
        else:
            # dead zone — carry prior
            if prior is None:
                # first day in dead zone before any clear signal; default BULL
                current = BULL
            else:
                current = prior

        if prior is None:
            prior = current

        regime_list.append(current)
        prior = current

    return pd.Series(regime_list, index=signals.index, name="regime", dtype=object)


def test_regime_state_machine() -> None:
    """
    Unit test: synthetic SPY series to verify hysteresis.
    SPY starts below SMA → BEAR.
    Crosses up through +5% band → BULL.
    Drifts into dead zone → stays BULL.
    Crosses down through −3% band → BEAR.
    """
    import numpy as np

    n = 300
    sma_val = 100.0

    # Day 0–99: SPY well below SMA (< 97) → BEAR
    # Day 100–149: SPY rises into dead zone (97–105) → still BEAR
    # Day 150–199: SPY crosses above 105 → BULL
    # Day 200–249: SPY drifts back into dead zone (97–105) → still BULL
    # Day 250–299: SPY falls below 97 → BEAR
    spy_prices = np.concatenate([
        np.full(100, 94.0),   # below band
        np.full(50, 101.0),   # in dead zone (> 97, < 105)
        np.full(50, 108.0),   # above upper band
        np.full(50, 101.0),   # back in dead zone
        np.full(50, 94.0),    # below lower band
    ])

    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    sig = pd.DataFrame(index=idx)
    sig["spy_sma"] = sma_val
    sig["spy_above_band"] = spy_prices > sma_val * 1.05
    sig["spy_below_band"] = spy_prices < sma_val * 0.97
    sig["spy_in_deadzone"] = ~sig["spy_above_band"] & ~sig["spy_below_band"]

    regime = compute_regime(sig)

    # Assertions
    assert all(regime.iloc[0:100] == BEAR), "Segment 0–99 should be BEAR"
    assert all(regime.iloc[100:150] == BEAR), "Segment 100–149 should stay BEAR (dead zone)"
    assert all(regime.iloc[150:200] == BULL), "Segment 150–199 should be BULL"
    assert all(regime.iloc[200:250] == BULL), "Segment 200–249 should stay BULL (dead zone)"
    assert all(regime.iloc[250:300] == BEAR), "Segment 250–299 should be BEAR"

    print("  [TEST] Regime state machine: PASSED")


if __name__ == "__main__":
    test_regime_state_machine()
