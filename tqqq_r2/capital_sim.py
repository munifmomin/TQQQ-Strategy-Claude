"""
$1,000 capital simulation layer (Section 5 of the spec).

Tracks the portfolio in real dollars and (optionally fractional) share
counts, starting from STARTING_CAPITAL. Produces monthly and annual
dollar tables plus the comparison/side-by-side outputs.

Capital sim must compound correctly: target weights apply to *current*
equity each rebalance, not the original $1,000.
"""

from __future__ import annotations
import math
import pandas as pd
import numpy as np

from tqqq_r2.config import (
    STARTING_CAPITAL, FRACTIONAL_SHARES, MIN_TRADE_DOLLARS,
    SLIPPAGE_BPS, SLIPPAGE_BPS_ILLIQUID, ILLIQUID_TICKERS, COMMISSION_PER_TRADE,
)
from tqqq_r2.backtest import get_target_weights, _slippage_factor


def run_capital_sim(allocations: pd.DataFrame,
                    prices: pd.DataFrame,
                    starting_capital: float = STARTING_CAPITAL,
                    fractional_shares: bool = FRACTIONAL_SHARES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the $-denominated simulation.

    Returns:
      portfolio: daily DataFrame with columns
        [portfolio_value, cash, target, vol_weight, holdings_repr]
      trade_log: every trade, dollars + shares
    """
    if isinstance(prices.columns, pd.MultiIndex):
        opens = prices["Open"]
        closes = prices["Close"]
    else:
        opens = prices
        closes = prices

    common = closes.index.intersection(allocations.index)
    alloc = allocations.reindex(common)

    holdings: dict[str, float] = {}
    cash = starting_capital
    portfolio_value = starting_capital

    portfolio_records = []
    trade_records = []

    for i, date in enumerate(common):
        row = alloc.loc[date]
        trigger = row["trigger"]

        if i > 0:
            pv = sum(
                holdings.get(t, 0) * closes.loc[date, t]
                for t in holdings
                if t in closes.columns
            ) + cash
            portfolio_value = pv

        needs_trade = trigger in ("INITIAL", "SIGNAL_CHANGE", "VOL_RESIZE")

        if needs_trade:
            target_weights = get_target_weights(row)
            exec_prices: dict[str, float] = {}
            for t in list(set(target_weights.keys()) | set(holdings.keys())):
                if t in opens.columns:
                    p = opens.loc[date, t] if date in opens.index else np.nan
                    exec_prices[t] = p if not pd.isna(p) else (closes.loc[date, t] if t in closes.columns else np.nan)

            target_dollars = {t: portfolio_value * w for t, w in target_weights.items()}

            current_dollars: dict[str, float] = {}
            for t, shares in holdings.items():
                ep = exec_prices.get(t, np.nan)
                current_dollars[t] = shares * ep if not pd.isna(ep) else 0.0

            all_tickers = set(target_dollars.keys()) | set(current_dollars.keys())
            trades: list[tuple[str, float]] = []
            for t in all_tickers:
                tgt_d = target_dollars.get(t, 0.0)
                cur_d = current_dollars.get(t, 0.0)
                delta = tgt_d - cur_d
                if abs(delta) >= MIN_TRADE_DOLLARS:
                    trades.append((t, delta))

            for t, delta in trades:
                ep = exec_prices.get(t, np.nan)
                if pd.isna(ep) or ep <= 0:
                    continue
                side = "buy" if delta > 0 else "sell"
                slip = _slippage_factor(t, side)
                exec_price = ep * slip
                shares_delta = delta / exec_price

                if not fractional_shares:
                    current_shares = holdings.get(t, 0.0)
                    target_shares_total = current_shares + shares_delta
                    if side == "buy":
                        target_shares_total = math.floor(target_shares_total)
                    else:
                        target_shares_total = math.ceil(target_shares_total)
                    shares_delta = target_shares_total - current_shares
                    delta = shares_delta * exec_price
                    if abs(delta) < 0.01:
                        continue

                holdings[t] = holdings.get(t, 0.0) + shares_delta
                if abs(holdings[t]) < 1e-9:
                    del holdings[t]
                cost = abs(delta) * abs(slip - 1) + COMMISSION_PER_TRADE
                cash -= delta

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
            "cash": cash,
            "target": row["target"],
            "vol_weight": row["vol_weight"],
        })

    portfolio_df = pd.DataFrame(portfolio_records).set_index("date")
    trade_df = pd.DataFrame(trade_records) if trade_records else pd.DataFrame(
        columns=["date", "ticker", "side", "dollars", "exec_price", "shares", "slippage_cost", "trigger"]
    )

    return portfolio_df, trade_df


def monthly_table(portfolio: pd.DataFrame, trade_df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly $ table: start balance, end balance, $ change, % return,
    rotations that month, dominant instrument held.
    """
    pv = portfolio["portfolio_value"]
    rows = []

    for period, group in pv.groupby(pv.index.to_period("M")):
        start_bal = group.iloc[0]
        end_bal = group.iloc[-1]
        # use prior month's last balance as the "start" if available, for continuity
        idx_pos = pv.index.get_loc(group.index[0])
        if idx_pos > 0:
            start_bal = pv.iloc[idx_pos - 1]

        dollar_change = end_bal - start_bal
        pct_return = (end_bal / start_bal - 1) * 100 if start_bal != 0 else 0.0

        month_dates = group.index
        n_rotations = 0
        if len(trade_df) > 0:
            mask = (trade_df["date"] >= month_dates[0]) & (trade_df["date"] <= month_dates[-1])
            n_rotations = trade_df.loc[mask, "date"].nunique()

        target_counts = portfolio.loc[month_dates, "target"].value_counts()
        dominant = target_counts.idxmax() if len(target_counts) > 0 else ""

        rows.append({
            "month": str(period),
            "start_balance": round(start_bal, 2),
            "end_balance": round(end_bal, 2),
            "dollar_change": round(dollar_change, 2),
            "pct_return": round(pct_return, 2),
            "rotations": n_rotations,
            "dominant_instrument": dominant,
        })

    return pd.DataFrame(rows)


def annual_table(portfolio: pd.DataFrame) -> pd.DataFrame:
    """
    Annual $ table: start balance, end balance, $ change, % return,
    best month, worst month, max intra-year drawdown $.
    """
    pv = portfolio["portfolio_value"]
    rows = []

    for year, group in pv.groupby(pv.index.year):
        start_bal = group.iloc[0]
        idx_pos = pv.index.get_loc(group.index[0])
        if idx_pos > 0:
            start_bal = pv.iloc[idx_pos - 1]
        end_bal = group.iloc[-1]
        dollar_change = end_bal - start_bal
        pct_return = (end_bal / start_bal - 1) * 100 if start_bal != 0 else 0.0

        monthly_ret = group.resample("ME").last().pct_change().dropna() * 100
        best_month = round(monthly_ret.max(), 2) if len(monthly_ret) > 0 else 0.0
        worst_month = round(monthly_ret.min(), 2) if len(monthly_ret) > 0 else 0.0

        roll_max = group.cummax()
        dd_dollars = (group - roll_max).min()

        rows.append({
            "year": int(year),
            "start_balance": round(start_bal, 2),
            "end_balance": round(end_bal, 2),
            "dollar_change": round(dollar_change, 2),
            "pct_return": round(pct_return, 2),
            "best_month_pct": best_month,
            "worst_month_pct": worst_month,
            "max_intra_year_dd_dollars": round(dd_dollars, 2),
        })

    return pd.DataFrame(rows)


def comparison_table(port_a: pd.DataFrame, port_b: pd.DataFrame,
                     label_a: str, label_b: str) -> pd.DataFrame:
    """Month-by-month side-by-side $ balances for two strategies."""
    pv_a = port_a["portfolio_value"]
    pv_b = port_b["portfolio_value"]

    rows = []
    common_months = sorted(set(pv_a.index.to_period("M")) & set(pv_b.index.to_period("M")))

    for period in common_months:
        ga = pv_a[pv_a.index.to_period("M") == period]
        gb = pv_b[pv_b.index.to_period("M") == period]
        rows.append({
            "month": str(period),
            f"{label_a}_balance": round(ga.iloc[-1], 2),
            f"{label_b}_balance": round(gb.iloc[-1], 2),
            f"{label_a}_pct_return": round((ga.iloc[-1] / ga.iloc[0] - 1) * 100, 2),
            f"{label_b}_pct_return": round((gb.iloc[-1] / gb.iloc[0] - 1) * 100, 2),
        })

    return pd.DataFrame(rows)
