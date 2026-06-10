"""
Performance metrics computation.
Identical metric definitions to prior specs for direct head-to-head comparison.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def total_return(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) < 2:
        return 0.0
    return (equity.iloc[-1] / equity.iloc[0] - 1) * 100


def cagr(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) < 2:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return ((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) * 100


def annual_vol(equity: pd.Series) -> float:
    ret = equity.pct_change().dropna()
    if len(ret) < 2:
        return 0.0
    return ret.std() * np.sqrt(252) * 100


def max_drawdown(equity: pd.Series) -> tuple[float, int, int]:
    """Returns (mdd_pct, peak_to_trough_days, days_to_recovery)."""
    equity = equity.dropna()
    if len(equity) < 2:
        return 0.0, 0, 0
    roll_max = equity.cummax()
    dd = equity / roll_max - 1
    if dd.isna().all():
        return 0.0, 0, 0
    mdd = dd.min() * 100

    trough_idx = dd.idxmin()
    peak_idx = roll_max.loc[:trough_idx].idxmax()
    p2t = (trough_idx - peak_idx).days

    peak_val = equity.loc[peak_idx]
    after_trough = equity.loc[trough_idx:]
    recovered = after_trough[after_trough >= peak_val]
    recovery_days = (recovered.index[0] - trough_idx).days if len(recovered) > 0 else -1

    return mdd, p2t, recovery_days


def sharpe(equity: pd.Series, rf: float = 0.0) -> float:
    ret = equity.pct_change().dropna()
    if len(ret) < 2 or ret.std() == 0:
        return 0.0
    excess = ret - rf / 252
    return (excess.mean() / excess.std()) * np.sqrt(252)


def sortino(equity: pd.Series, mar: float = 0.0) -> float:
    ret = equity.pct_change().dropna()
    if len(ret) < 2:
        return 0.0
    downside = ret[ret < mar]
    dd_std = downside.std() * np.sqrt(252)
    ann_ret = cagr(equity) / 100
    return ann_ret / dd_std if dd_std > 0 else 0.0


def calmar(equity: pd.Series) -> float:
    mdd, _, _ = max_drawdown(equity)
    ann = cagr(equity)
    return ann / abs(mdd) if mdd != 0 else 0.0


def annual_returns(equity: pd.Series) -> pd.Series:
    return equity.resample("YE").last().pct_change().dropna() * 100


def compute_metrics(equity: pd.Series, trade_df: pd.DataFrame | None = None,
                    label: str = "") -> dict:
    mdd, p2t, rec = max_drawdown(equity)
    equity_clean = equity.dropna()
    years = (equity_clean.index[-1] - equity_clean.index[0]).days / 365.25 if len(equity_clean) > 1 else 0
    trades_yr = len(trade_df) / years if trade_df is not None and years > 0 else 0.0

    signal_trades = vol_trades = 0
    if trade_df is not None and len(trade_df) > 0:
        signal_trades = (trade_df["trigger"] == "SIGNAL_CHANGE").sum()
        vol_trades = (trade_df["trigger"] == "VOL_RESIZE").sum()

    return {
        "label": label,
        "total_return_pct": round(total_return(equity), 2),
        "cagr_pct": round(cagr(equity), 2),
        "ann_vol_pct": round(annual_vol(equity), 2),
        "max_drawdown_pct": round(mdd, 2),
        "mdd_peak_to_trough_days": p2t,
        "mdd_recovery_days": rec,
        "calmar": round(calmar(equity), 3),
        "sharpe": round(sharpe(equity), 3),
        "sortino": round(sortino(equity), 3),
        "trades_per_year": round(trades_yr, 1),
        "signal_change_trades": signal_trades,
        "vol_resize_trades": vol_trades,
    }


def regime_breakdown(equity: pd.Series, regime: pd.Series) -> pd.DataFrame:
    rows = []
    for reg in ["BULL", "BEAR"]:
        mask = regime == reg
        if not mask.any():
            continue
        sub = equity[mask]
        if len(sub) < 2:
            continue
        rows.append({
            "regime": reg,
            "days": len(sub),
            "max_drawdown_pct": round(max_drawdown(sub)[0], 2),
        })
    return pd.DataFrame(rows)
