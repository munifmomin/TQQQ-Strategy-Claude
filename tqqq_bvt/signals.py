"""
Signal computation — SMA, Wilder RSI, realized vol, buffer bands.
IMPORTANT: All indicators computed on QQQ/SPY (unleveraged) — never on TQQQ.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def wilder_rsi(series: pd.Series, window: int = 10) -> pd.Series:
    """Wilder (smoothed) RSI, identical to TradingView default RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder smoothing: seed with SMA, then EWM with alpha=1/window
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def realized_vol(series: pd.Series, lookback: int = 20) -> pd.Series:
    """
    Annualized realized vol of daily log returns over `lookback` days.
    Uses returns through close D only (no look-ahead).
    """
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(lookback).std() * np.sqrt(252)


def compute_signals(prices: pd.DataFrame,
                    sma_window: int = 200,
                    rsi_window: int = 10,
                    vol_lookback: int = 20,
                    buffer_up: float = 0.05,
                    buffer_dn: float = 0.03) -> pd.DataFrame:
    """
    Compute all signals required by the strategy.
    Returns a DataFrame with columns:
      spy_sma, spy_above_band, spy_below_band, spy_in_deadzone,
      qqq_rsi, spy_rsi, qqq_above_20sma, sqqq_rsi10, bond_rsi10 (per bond),
      realized_vol_qqq
    """
    spy = prices["SPY"]
    qqq = prices["QQQ"]

    sig = pd.DataFrame(index=prices.index)

    sig["spy_sma"] = sma(spy, sma_window)
    sig["spy_above_band"] = spy > sig["spy_sma"] * (1 + buffer_up)
    sig["spy_below_band"] = spy < sig["spy_sma"] * (1 - buffer_dn)
    sig["spy_in_deadzone"] = ~sig["spy_above_band"] & ~sig["spy_below_band"]

    sig["qqq_rsi"] = wilder_rsi(qqq, rsi_window)
    sig["spy_rsi"] = wilder_rsi(spy, rsi_window)
    sig["qqq_above_20sma"] = qqq > sma(qqq, 20)

    sig["realized_vol_qqq"] = realized_vol(qqq, vol_lookback)

    # RSI for SQQQ and bond selection
    sig["sqqq_rsi10"] = wilder_rsi(prices["SQQQ"], rsi_window)
    for bond in ["BSV", "IEF", "TLT"]:
        sig[f"{bond.lower()}_rsi10"] = wilder_rsi(prices[bond], rsi_window)

    return sig
