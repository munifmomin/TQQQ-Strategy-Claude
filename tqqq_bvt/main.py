"""
Main entry point for the TQQQ Buffered + Volatility-Targeted backtest.

Usage:
    python -m tqqq_bvt.main                    # full run
    python -m tqqq_bvt.main --skip-opt         # skip optimization (faster)
    python -m tqqq_bvt.main --refresh-data     # force yfinance re-download
"""

from __future__ import annotations
import argparse
import os
import sys
import pandas as pd

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqqq_bvt.config import (
    START_DATE, BOND_VARIANTS, SIZING_VARIANTS, OUTPUT_DIR, INITIAL_CAPITAL,
)
from tqqq_bvt.data import fetch_data, align_to_spy, validate
from tqqq_bvt.signals import compute_signals
from tqqq_bvt.regime import compute_regime, test_regime_state_machine
from tqqq_bvt.strategy import StrategyParams, compute_allocations
from tqqq_bvt.backtest import run_backtest, run_benchmark, run_60_40
from tqqq_bvt.metrics import compute_metrics, regime_breakdown, period_breakdown, annual_returns
from tqqq_bvt.walk_forward import walk_forward_summary, rolling_windows
from tqqq_bvt.optimize import optimize
from tqqq_bvt.charts import (
    equity_curves, drawdown_chart, annual_returns_heatmap,
    rolling_metrics_chart, vol_targeting_diagnostic, optimization_surface,
)


CONSOLE_HEADER = (
    f"\n{'Variant':<20} | {'CAGR':>6} | {'MaxDD':>7} | {'Calmar':>7} | "
    f"{'Sharpe':>7} | {'Sortino':>8} | {'Trades/yr':>10}"
)
CONSOLE_SEP = "-" * 90


def fmt_row(m: dict) -> str:
    return (
        f"{m['label']:<20} | {m['cagr_pct']:>5.1f}% | {m['max_drawdown_pct']:>6.1f}% | "
        f"{m['calmar']:>7.3f} | {m['sharpe']:>7.3f} | {m['sortino']:>8.3f} | "
        f"{m['trades_per_year']:>10.1f}"
    )


def run_variant(prices: pd.DataFrame,
                signals: pd.DataFrame,
                regime: pd.Series,
                target_vol: float | None,
                bond: str,
                label: str,
                use_uvxy: bool = False) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, dict]:
    sp = StrategyParams(target_vol=target_vol, bond=bond, use_uvxy_overbought=use_uvxy)
    alloc = compute_allocations(signals, regime, sp)
    alloc_live = alloc.loc[START_DATE:]
    prices_live = prices.loc[START_DATE:]
    portfolio, trade_df = run_backtest(alloc_live, prices_live)
    equity = portfolio["portfolio_value"]
    m = compute_metrics(equity, trade_df, label=label)
    return equity, portfolio, trade_df, m


def main(skip_opt: bool = False, refresh_data: bool = False) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("TQQQ Buffered + Volatility-Targeted Backtest")
    print("=" * 70)

    # ── Unit test ────────────────────────────────────────────────────────────
    print("\n[1] Running regime state machine unit test...")
    test_regime_state_machine()

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\n[2] Fetching data...")
    prices_raw = fetch_data(force_refresh=refresh_data)
    prices = align_to_spy(prices_raw)

    active = ["SPY", "QQQ", "TQQQ", "SQQQ", "UPRO", "BIL", "BSV", "IEF", "TLT"]
    validate(prices, active, start=START_DATE)
    print(f"  Data range: {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"  Shape: {prices.shape}")

    # ── Signals + regime (baseline params) ───────────────────────────────────
    print("\n[3] Computing signals and regime...")
    signals = compute_signals(prices)
    regime = compute_regime(signals)
    print(f"  BULL days: {(regime == 'BULL').sum()}, BEAR days: {(regime == 'BEAR').sum()}")

    # ── Run all variants ─────────────────────────────────────────────────────
    print("\n[4] Running strategy variants...")
    all_metrics: list[dict] = []
    all_equity: dict[str, pd.Series] = {}
    all_trades: dict[str, pd.DataFrame] = {}
    all_portfolios: dict[str, pd.DataFrame] = {}

    for vt_label, target_vol in SIZING_VARIANTS.items():
        for bond in BOND_VARIANTS:
            label = f"{vt_label}-{bond}"
            print(f"  Running {label}...")
            eq, port, trades, m = run_variant(prices, signals, regime, target_vol, bond, label)
            all_equity[label] = eq
            all_trades[label] = trades
            all_portfolios[label] = port
            all_metrics.append(m)

    # ── Structural variants (S1–S5) ───────────────────────────────────────────
    print("\n[5] Running structural variants...")

    # S1: UVXY on overbought instead of BIL
    if "UVXY" in prices.columns:
        eq_s1, _, tr_s1, m_s1 = run_variant(prices, signals, regime, 0.40, "IEF", "S1-UVXY-OB", use_uvxy=True)
        all_equity["S1-UVXY-OB"] = eq_s1
        all_trades["S1-UVXY-OB"] = tr_s1
        all_metrics.append(m_s1)

    # S2: Regime-dependent target vol
    print("  Running S2 (regime-dependent vol)...")
    from tqqq_bvt.strategy import StrategyParams, compute_allocations
    sp_s2 = StrategyParams(target_vol=0.30, bond="IEF")  # lower in bear (approx via lower overall)
    alloc_s2 = compute_allocations(signals, regime, sp_s2)
    # Override: in BULL use 0.50, BEAR use 0.25
    for date in alloc_s2.index:
        if alloc_s2.loc[date, "risk_on"]:
            r = regime.loc[date] if date in regime.index else "BULL"
            rv = signals.loc[date, "realized_vol_qqq"] if date in signals.index else None
            if rv and not pd.isna(rv) and rv > 0:
                tv = 0.50 if r == "BULL" else 0.25
                alloc_s2.loc[date, "vol_weight"] = min(tv / rv, 1.0)
    port_s2, tr_s2 = run_backtest(alloc_s2.loc[START_DATE:], prices.loc[START_DATE:])
    m_s2 = compute_metrics(port_s2["portfolio_value"], tr_s2, label="S2-RegimeDep-Vol")
    all_equity["S2-RegimeDep-Vol"] = port_s2["portfolio_value"]
    all_metrics.append(m_s2)

    # S3: VIX hard filter (skip if VIX not fetched; warn)
    print("  S3 (VIX filter): VIX not in default tickers — skipping (add ^VIX to fetch if desired)")

    # S4: In BULL transitional, hold TQQQ instead of BIL
    print("  Running S4 (BULL transitional = TQQQ)...")
    # We implement this as a post-processing patch on allocations
    alloc_s4 = compute_allocations(signals, regime, StrategyParams(target_vol=0.40, bond="IEF"))
    for date in alloc_s4.index:
        if alloc_s4.loc[date, "target"] == "BIL" and regime.loc[date] if date in regime.index else False:
            if regime.loc[date] == "BULL":
                alloc_s4.loc[date, "target"] = "TQQQ"
                alloc_s4.loc[date, "risk_on"] = True
    port_s4, tr_s4 = run_backtest(alloc_s4.loc[START_DATE:], prices.loc[START_DATE:])
    m_s4 = compute_metrics(port_s4["portfolio_value"], tr_s4, label="S4-BullTrans=TQQQ")
    all_equity["S4-BullTrans=TQQQ"] = port_s4["portfolio_value"]
    all_metrics.append(m_s4)

    # S5: 2 consecutive days to change regime
    print("  Running S5 (2-day confirmation)...")
    from tqqq_bvt.regime import BULL, BEAR
    regime_s5 = pd.Series(index=signals.index, dtype=object)
    prior = None
    prev_signal = None
    above = signals["spy_above_band"]
    below = signals["spy_below_band"]
    for i, date in enumerate(signals.index):
        ab, bl = above.iloc[i], below.iloc[i]
        if pd.isna(ab):
            regime_s5.iloc[i] = None
            continue
        if ab:
            curr_signal = BULL
        elif bl:
            curr_signal = BEAR
        else:
            curr_signal = None  # dead zone
        # require 2 consecutive days
        if curr_signal is not None and curr_signal == prev_signal:
            prior = curr_signal
        if prior is None:
            prior = BULL if ab else BEAR
        regime_s5.iloc[i] = prior
        prev_signal = curr_signal
    sp_s5 = StrategyParams(target_vol=0.40, bond="IEF")
    alloc_s5 = compute_allocations(signals, regime_s5, sp_s5)
    port_s5, tr_s5 = run_backtest(alloc_s5.loc[START_DATE:], prices.loc[START_DATE:])
    m_s5 = compute_metrics(port_s5["portfolio_value"], tr_s5, label="S5-2DayConfirm")
    all_equity["S5-2DayConfirm"] = port_s5["portfolio_value"]
    all_metrics.append(m_s5)

    # ── Benchmarks ────────────────────────────────────────────────────────────
    print("\n[6] Running benchmarks...")
    for bm in ["QQQ", "TQQQ", "SPY"]:
        eq_bm = run_benchmark(bm, prices, start=START_DATE)
        all_equity[f"{bm} B&H"] = eq_bm
        m_bm = compute_metrics(eq_bm, label=f"{bm} B&H")
        all_metrics.append(m_bm)

    eq_6040 = run_60_40(prices, start=START_DATE)
    all_equity["60/40"] = eq_6040
    all_metrics.append(compute_metrics(eq_6040, label="60/40"))

    # ── Walk-forward ─────────────────────────────────────────────────────────
    print("\n[7] Walk-forward validation...")
    wf_rows = []
    base_label = "VT-40-IEF"
    if base_label in all_equity:
        wf = walk_forward_summary(all_equity[base_label], base_label, all_trades.get(base_label))
        if wf:
            print(f"  IS metrics: CAGR={wf['IS']['cagr_pct']:.1f}%  MDD={wf['IS']['max_drawdown_pct']:.1f}%")
            print(f"  OOS metrics: CAGR={wf['OOS']['cagr_pct']:.1f}%  MDD={wf['OOS']['max_drawdown_pct']:.1f}%")
            for flag in wf.get("red_flags", []):
                print(f"  ⚠  {flag}")
            row = {"label": base_label, **{f"IS_{k}": v for k, v in wf["IS"].items()},
                   **{f"OOS_{k}": v for k, v in wf["OOS"].items()}}
            wf_rows.append(row)

    wf_df = pd.DataFrame(wf_rows)
    wf_df.to_csv(os.path.join(OUTPUT_DIR, "walk_forward_summary.csv"), index=False)

    # ── Optimization ─────────────────────────────────────────────────────────
    if not skip_opt:
        print("\n[8] Running optimization (this may take several minutes)...")
        baseline_cagr = None
        if base_label in all_equity:
            from tqqq_bvt.metrics import cagr as calc_cagr
            baseline_cagr = calc_cagr(all_equity[base_label])
        top10, overfit = optimize(prices, bond="IEF", baseline_cagr=baseline_cagr, verbose=True)
        if not top10.empty:
            top10.to_csv(os.path.join(OUTPUT_DIR, "optimization_top10.csv"), index=False)
            print(f"  Top-10 OOS Calmar range: {top10['calmar'].min():.3f}–{top10['calmar'].max():.3f}")
        if not overfit.empty:
            overfit.to_csv(os.path.join(OUTPUT_DIR, "optimization_overfit_appendix.csv"), index=False)
            print(f"  Overfit combos excluded: {len(overfit)}")
        if not top10.empty:
            optimization_surface(top10)
    else:
        print("\n[8] Optimization skipped (--skip-opt)")

    # ── Save results ──────────────────────────────────────────────────────────
    print("\n[9] Saving results...")
    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "results_summary.csv"), index=False)

    # Trade log (baseline variant)
    if base_label in all_trades:
        all_trades[base_label].to_csv(os.path.join(OUTPUT_DIR, "trade_log.csv"), index=False)

    # Regime breakdown
    if base_label in all_portfolios:
        rb = regime_breakdown(all_equity[base_label], regime.loc[START_DATE:])
        rb.to_csv(os.path.join(OUTPUT_DIR, "regime_breakdown.csv"), index=False)

    # ── Charts ────────────────────────────────────────────────────────────────
    print("\n[10] Generating charts...")
    main_curves = {k: v for k, v in all_equity.items()
                   if any(k.startswith(p) for p in ["VT-40", "VT-OFF", "QQQ", "TQQQ", "SPY", "60/40"])}
    equity_curves(main_curves)
    drawdown_chart(main_curves)
    annual_returns_heatmap(main_curves)
    rolling_metrics_chart({k: v for k, v in main_curves.items() if not k.endswith("B&H") or "QQQ" in k})

    if base_label in all_portfolios and base_label in all_equity:
        vol_targeting_diagnostic(signals.loc[START_DATE:], all_portfolios[base_label])

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + CONSOLE_HEADER)
    print(CONSOLE_SEP)
    priority_order = [
        "VT-40-IEF", "VT-40-BSV", "VT-40-TLT",
        "VT-30-IEF", "VT-50-IEF", "VT-OFF-IEF",
        "S1-UVXY-OB", "S2-RegimeDep-Vol", "S4-BullTrans=TQQQ", "S5-2DayConfirm",
        "QQQ B&H", "TQQQ B&H", "SPY B&H", "60/40",
    ]
    printed = set()
    for lbl in priority_order:
        m = next((x for x in all_metrics if x["label"] == lbl), None)
        if m:
            print(fmt_row(m))
            printed.add(lbl)
    # remaining variants
    for m in all_metrics:
        if m["label"] not in printed:
            print(fmt_row(m))
    print(CONSOLE_SEP)
    print(f"\nOutput files written to: {OUTPUT_DIR}/")
    print("Done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-opt", action="store_true", help="Skip optimization phase")
    parser.add_argument("--refresh-data", action="store_true", help="Force yfinance re-download")
    args = parser.parse_args()
    main(skip_opt=args.skip_opt, refresh_data=args.refresh_data)
