"""
Data fetching and caching via yfinance.
All prices are split+dividend adjusted (auto_adjust=True).
ERs are already embedded in adjusted prices — do NOT deduct them again.
"""

from __future__ import annotations
import os
import warnings
import pandas as pd
import yfinance as yf
from tqqq_r2.config import TICKERS, DATA_CACHE_PATH, START_DATE


def fetch_data(tickers: list[str] = TICKERS,
               start: str = "2007-01-01",
               force_refresh: bool = False) -> pd.DataFrame:
    """Return a DataFrame of daily adjusted close prices, columns = tickers."""
    if os.path.exists(DATA_CACHE_PATH) and not force_refresh:
        df = pd.read_parquet(DATA_CACHE_PATH)
        missing = [t for t in tickers if t not in df.columns]
        if not missing:
            return df

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            tickers,
            start=start,
            auto_adjust=True,
            progress=False,
            threads=True,
        )

    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    else:
        closes = raw[["Close"]]
        closes.columns = tickers[:1]

    closes = closes.dropna(how="all")
    closes.to_parquet(DATA_CACHE_PATH)
    return closes


def align_to_spy(prices: pd.DataFrame) -> pd.DataFrame:
    """Align all series to SPY trading calendar; warn on gaps; forward/back-fill."""
    spy_idx = prices["SPY"].dropna().index
    aligned = prices.reindex(spy_idx)

    for col in aligned.columns:
        n_missing = aligned[col].isna().sum()
        if n_missing > 0:
            print(f"  [DATA] {col}: {n_missing} missing days on SPY calendar (will forward-fill)")
        total_days = len(aligned[col].dropna())
        if total_days < 252:
            print(f"  [WARN] {col} has only {total_days} trading days — less than 1 year")

    aligned = aligned.ffill().bfill()
    return aligned


def validate(prices: pd.DataFrame, active_tickers: list[str], start: str = START_DATE) -> None:
    """Assert no NaN in active tickers from start date onwards."""
    window = prices.loc[start:]
    for t in active_tickers:
        if t not in window.columns:
            raise ValueError(f"Ticker {t} not in price data")
        n_nan = window[t].isna().sum()
        if n_nan > 0:
            raise AssertionError(f"NaN found in {t}: {n_nan} days from {start}")
