"""
Event-driven daily backtest engine.
Supports fractional multi-instrument holdings (TQQQ + BIL splits).
Signals at close D → execute at open D+1.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from tqqq_bvt.config import (
    INITIAL_CAPITAL, SLIPPAGE_BPS, SLIPPAGE_BPS_ILLIQUID, ILLIQUID_TICKERS, COMMISSION_PER_TRADE,
)


def _slippage_factor(ticker: str, side: str = "buy") -> float:
    """Return 1 ± slippage. side='buy' → pay more; side='sell' → receive less."""
    bps = SLIPPAGE_BPS_ILLIQUID if ticker in ILLIQUID_TICKERS else SLIPPAGE_BPS
    factor = bps / 10_000
    return (1 + factor) if side == "buy" else (1 - factor)


def run_backtest(allocations: pd.DataFrame,
                 prices: pd.DataFrame,
                 initial_capital: float = INITIAL_CAPITAL) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the backtest.

    allocations: DataFrame with columns [target, vol_weight, trigger] indexed by date.
    prices: DataFrame of adjusted OHLCV data (needs 'Open' and 'Close').

    Returns:
      portfolio: daily portfolio value + holdings
      trade_log: every trade with trigger tag
    """
    # Build open price DataFrame
    if isinstance(prices.columns, pd.MultiIndex):
        opens = prices["Open"]
        closes = prices["Close"]
    else:
        opens = prices  # already just Close — use same for both (intraday not modelled)
        closes = prices

    # Align allocations to closes index (skip D+1 shift — we apply it below)
    common = closes.index.intersection(allocations.index)
    alloc = allocations.reindex(common)

    portfolio_records = []
    trade_records = []

    # Current holdings: {ticker: shares}
    holdings: dict[str, float] = {}
    cash = initial_capital   # all capital starts as uninvested cash
    portfolio_value = initial_capital

    # Pre-compute target weights for each day as (target, weight_in_target, weight_in_bil)
    def get_target_weights(row) -> dict[str, float]:
        tgt = row["target"]
        w = float(row["vol_weight"])
        if tgt == "BIL":
            return {"BIL": 1.0}
        # risk_on with partial BIL fill
        weights: dict[str, float] = {tgt: w}
        if w < 1.0:
            weights["BIL"] = 1.0 - w
        return weights

    for i, date in enumerate(common):
        row = alloc.loc[date]
        trigger = row["trigger"]

        # Mark-to-market current portfolio at today's close
        if i > 0:
            pv = sum(
                holdings.get(t, 0) * closes.loc[date, t]
                for t in holdings
                if t in closes.columns
            ) + cash
            portfolio_value = pv

        # Determine if we need to trade (execute signal from previous day at today's open)
        # On day 0 we always trade (INITIAL)
        needs_trade = trigger in ("INITIAL", "SIGNAL_CHANGE", "VOL_RESIZE")

        if needs_trade and "Open" in prices.columns or needs_trade:
            target_weights = get_target_weights(row)
            exec_prices: dict[str, float] = {}
            for t in list(set(target_weights.keys()) | set(holdings.keys())):
                if t in opens.columns:
                    p = opens.loc[date, t] if date in opens.index else np.nan
                    exec_prices[t] = p if not pd.isna(p) else closes.loc[date, t] if t in closes.columns else np.nan

            # Compute target dollar amounts
            target_dollars = {t: portfolio_value * w for t, w in target_weights.items()}

            # Current holdings value at execution prices
            current_dollars: dict[str, float] = {}
            for t, shares in holdings.items():
                ep = exec_prices.get(t, np.nan)
                current_dollars[t] = shares * ep if not pd.isna(ep) else 0.0

            # Build trade list: (ticker, delta_dollars)
            all_tickers = set(target_dollars.keys()) | set(current_dollars.keys())
            trades: list[tuple[str, float]] = []
            for t in all_tickers:
                tgt_d = target_dollars.get(t, 0.0)
                cur_d = current_dollars.get(t, 0.0)
                delta = tgt_d - cur_d
                if abs(delta) > 1.0:  # ignore sub-dollar noise
                    trades.append((t, delta))

            # Execute trades
            total_cost = 0.0
            for t, delta in trades:
                ep = exec_prices.get(t, np.nan)
                if pd.isna(ep) or ep <= 0:
                    continue
                side = "buy" if delta > 0 else "sell"
                slip = _slippage_factor(t, side)
                exec_price = ep * slip
                shares_delta = delta / exec_price
                holdings[t] = holdings.get(t, 0.0) + shares_delta
                # remove zero/tiny holdings
                if abs(holdings[t]) < 1e-6:
                    del holdings[t]
                cost = abs(delta) * abs(slip - 1) + COMMISSION_PER_TRADE
                total_cost += cost
                cash -= delta  # selling adds cash, buying reduces

                trade_records.append({
                    "date": date,
                    "ticker": t,
                    "side": side,
                    "dollars": abs(delta),
                    "exec_price": exec_price,
                    "shares": abs(shares_delta),
                    "slippage_cost": cost,
                    "trigger": trigger,
                })

        portfolio_records.append({
            "date": date,
            "portfolio_value": portfolio_value,
            "regime": row.get("regime", None),
            "target": row["target"],
            "vol_weight": row["vol_weight"],
        })

    portfolio_df = pd.DataFrame(portfolio_records).set_index("date")
    trade_df = pd.DataFrame(trade_records) if trade_records else pd.DataFrame(
        columns=["date", "ticker", "side", "dollars", "exec_price", "shares", "slippage_cost", "trigger"]
    )

    return portfolio_df, trade_df


def run_benchmark(ticker: str, prices: pd.DataFrame,
                  start: str, initial_capital: float = INITIAL_CAPITAL) -> pd.Series:
    """Buy & hold benchmark — single purchase at start, slippage on entry only."""
    if isinstance(prices.columns, pd.MultiIndex):
        closes = prices["Close"]
    else:
        closes = prices

    if ticker not in closes.columns:
        raise ValueError(f"{ticker} not in price data")

    series = closes[ticker].loc[start:].dropna()
    entry_price = series.iloc[0] * _slippage_factor(ticker, "buy")
    shares = initial_capital / entry_price
    return (series * shares).rename(ticker)


def run_60_40(prices: pd.DataFrame, start: str,
              initial_capital: float = INITIAL_CAPITAL) -> pd.Series:
    """60% SPY + 40% IEF, annual rebalance."""
    if isinstance(prices.columns, pd.MultiIndex):
        closes = prices["Close"]
    else:
        closes = prices

    spy = closes["SPY"].loc[start:].dropna()
    ief = closes["IEF"].loc[start:].dropna()
    common = spy.index.intersection(ief.index)
    spy, ief = spy.reindex(common), ief.reindex(common)

    portfolio = pd.Series(index=common, dtype=float)
    spy_shares = (initial_capital * 0.60) / (spy.iloc[0] * _slippage_factor("SPY", "buy"))
    ief_shares = (initial_capital * 0.40) / (ief.iloc[0] * _slippage_factor("IEF", "buy"))

    rebal_years: set[int] = set()
    for i, date in enumerate(common):
        year = date.year
        pv = spy.iloc[i] * spy_shares + ief.iloc[i] * ief_shares
        if year not in rebal_years and i > 0:
            # Annual rebalance
            spy_shares = pv * 0.60 / spy.iloc[i]
            ief_shares = pv * 0.40 / ief.iloc[i]
            rebal_years.add(year)
        portfolio.iloc[i] = spy.iloc[i] * spy_shares + ief.iloc[i] * ief_shares

    return portfolio
