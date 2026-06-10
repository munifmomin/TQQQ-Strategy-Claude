"""
Signal computation — SMA, Wilder RSI, realized vol (simple + EWMA), buffer bands.

CRITICAL FIX vs the prior (BVT) spec: realized vol for sizing is measured on
TQQQ (the actual held leveraged instrument), NOT on QQQ. QQQ vol runs ~10-30%
which is far below any sane TARGET_VOL, so the overlay was previously inert.
TQQQ vol runs ~30-90%, so a TARGET_VOL near 0.55 actually engages.

Trend/RSI signals remain on QQQ/SPY (unleveraged) — kept separate from sizing vol.
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

    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def realized_vol(series: pd.Series, lookback: int = 20) -> pd.Series:
    """
    Annualized realized vol of daily log returns over `lookback` days (simple rolling std).
    Uses returns through close D only (no look-ahead).
    """
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(lookback).std() * np.sqrt(252)


def realized_vol_ewma(series: pd.Series, halflife: int = 10) -> pd.Series:
    """EWMA realized vol (faster-reacting alternative for T1 structural test)."""
    log_ret = np.log(series / series.shift(1))
    return log_ret.ewm(halflife=halflife, min_periods=halflife).std() * np.sqrt(252)


def compute_signals(prices: pd.DataFrame,
                    sma_window: int = 200,
                    rsi_window: int = 10,
                    vol_lookback: int = 20,
                    vol_ewma_halflife: int = 10) -> pd.DataFrame:
    """
    Compute all signals required by the strategy.
    Returns a DataFrame with columns:
      spy_sma, spy_above_200,
      qqq_rsi, spy_rsi, qqq_above_20sma,
      sqqq_rsi10, <bond>_rsi10,
      tqqq_realized_vol, tqqq_realized_vol_ewma
    """
    spy = prices["SPY"]
    qqq = prices["QQQ"]
    tqqq = prices["TQQQ"]

    sig = pd.DataFrame(index=prices.index)

    sig["spy_sma"] = sma(spy, sma_window)
    sig["spy_above_200"] = spy > sig["spy_sma"]

    sig["qqq_rsi"] = wilder_rsi(qqq, rsi_window)
    sig["spy_rsi"] = wilder_rsi(spy, rsi_window)
    sig["qqq_above_20sma"] = qqq > sma(qqq, 20)

    # CRITICAL: vol measured on TQQQ (the held instrument), not QQQ
    sig["tqqq_realized_vol"] = realized_vol(tqqq, vol_lookback)
    sig["tqqq_realized_vol_ewma"] = realized_vol_ewma(tqqq, vol_ewma_halflife)

    sig["sqqq_rsi10"] = wilder_rsi(prices["SQQQ"], rsi_window)
    for bond in ["BSV", "IEF", "TLT"]:
        sig[f"{bond.lower()}_rsi10"] = wilder_rsi(prices[bond], rsi_window)

    return sig
