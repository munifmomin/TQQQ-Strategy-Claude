"""
Walk-forward validation. IDENTICAL methodology to prior specs.
In-sample: 2011-10-01 -> 2018-12-31
Out-of-sample: 2019-01-01 -> present
Rolling 3-year windows, step 6 months.
"""

from __future__ import annotations
import pandas as pd
from dateutil.relativedelta import relativedelta
from tqqq_r2.config import WF_IN_SAMPLE_END, WF_OUT_SAMPLE_START, WF_ROLLING_YEARS, WF_STEP_MONTHS
from tqqq_r2.metrics import compute_metrics, max_drawdown, cagr


def split_is_oos(equity: pd.Series) -> tuple[pd.Series, pd.Series]:
    is_ = equity.loc[:WF_IN_SAMPLE_END]
    oos = equity.loc[WF_OUT_SAMPLE_START:]
    return is_, oos


def check_red_flags(is_metrics: dict, oos_metrics: dict) -> list[str]:
    flags = []
    for key in ["cagr_pct", "calmar", "sharpe", "sortino"]:
        iv = is_metrics.get(key, 0)
        ov = oos_metrics.get(key, 0)
        if iv != 0 and ov / iv < 0.5:
            flags.append(f"RED FLAG: {key} degraded {iv:.2f} IS -> {ov:.2f} OOS ({ov/iv*100:.0f}% of IS)")
    return flags


def rolling_windows(equity: pd.Series,
                    years: int = WF_ROLLING_YEARS,
                    step_months: int = WF_STEP_MONTHS) -> pd.DataFrame:
    rows = []
    equity = equity.dropna()
    if len(equity) < 2:
        return pd.DataFrame()
    start = equity.index[0]
    end = equity.index[-1]
    window_delta = relativedelta(years=years)
    step_delta = relativedelta(months=step_months)

    cursor = start
    while cursor + window_delta <= end:
        ws = cursor
        we = cursor + window_delta
        sub = equity.loc[ws:we]
        if len(sub) < 30:
            cursor += step_delta
            continue
        mdd, _, _ = max_drawdown(sub)
        rows.append({
            "window_start": ws,
            "window_end": we,
            "cagr_pct": round(cagr(sub), 2) if len(sub) > 50 else None,
            "max_drawdown_pct": round(mdd, 2),
        })
        cursor += step_delta

    return pd.DataFrame(rows)


def walk_forward_summary(equity: pd.Series, label: str = "",
                         trade_df: pd.DataFrame | None = None) -> dict:
    is_eq, oos_eq = split_is_oos(equity)
    if len(is_eq) < 2 or len(oos_eq) < 2:
        return {}

    is_trade = trade_df[trade_df["date"] <= WF_IN_SAMPLE_END] if trade_df is not None else None
    oos_trade = trade_df[trade_df["date"] >= WF_OUT_SAMPLE_START] if trade_df is not None else None

    is_m = compute_metrics(is_eq, is_trade, label=f"{label}_IS")
    oos_m = compute_metrics(oos_eq, oos_trade, label=f"{label}_OOS")
    flags = check_red_flags(is_m, oos_m)

    return {"IS": is_m, "OOS": oos_m, "red_flags": flags}
