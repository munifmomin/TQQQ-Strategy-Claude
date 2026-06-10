"""
Main entry point for the TQQQ Round-2 (corrected vol-target + $1,000 capital sim) backtest.

Usage:
    python -m tqqq_r2.main                    # full run
    python -m tqqq_r2.main --skip-opt         # skip optimization (faster)
    python -m tqqq_r2.main --refresh-data     # force yfinance re-download
"""

from __future__ import annotations
import argparse
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqqq_r2.config import (
    START_DATE, BOND_VARIANTS, SIZING_VARIANTS, OUTPUT_DIR,
    TARGET_VOL, STARTING_CAPITAL,
)
from tqqq_r2.data import fetch_data, align_to_spy, validate
from tqqq_r2.signals import compute_signals
from tqqq_r2.strategy import StrategyParams, compute_allocations
from tqqq_r2.ftlt_original import compute_allocations_ftlt
from tqqq_r2.backtest import run_backtest, run_benchmark, run_60_40
from tqqq_r2.capital_sim import run_capital_sim, monthly_table, annual_table, comparison_table
from tqqq_r2.metrics import compute_metrics, cagr as calc_cagr
from tqqq_r2.walk_forward import walk_forward_summary, check_red_flags
from tqqq_r2.optimize import optimize
from tqqq_r2.charts import (
    dollar_equity_curves, monthly_dollar_bars, drawdown_chart,
    annual_returns_heatmap, rolling_metrics_chart, vol_targeting_diagnostic,
)


RISK_HEADER = (
    f"\n{'Variant':<22} | {'CAGR':>7} | {'MaxDD':>7} | {'Calmar':>7} | "
    f"{'Sharpe':>7} | {'Sortino':>8} | {'Trades/yr':>10}"
)
RISK_SEP = "-" * 80

DOLLAR_HEADER = (
    f"\n{'Strategy':<14} | {'CAGR':>7} | {'MaxDD':>7} | {'Calmar':>7} | "
    f"{'Sharpe':>7} | {'Trades/yr':>10} | {'$1k -> End':>12} | {'Yr1 End($)':>11}"
)
DOLLAR_SEP = "-" * 90


def fmt_risk_row(m: dict) -> str:
    return (
        f"{m['label']:<22} | {m['cagr_pct']:>6.1f}% | {m['max_drawdown_pct']:>6.1f}% | "
        f"{m['calmar']:>7.3f} | {m['sharpe']:>7.3f} | {m['sortino']:>8.3f} | "
        f"{m['trades_per_year']:>10.1f}"
    )


def fmt_dollar_row(label: str, m: dict, end_balance: float, yr1_end: float) -> str:
    return (
        f"{label:<14} | {m['cagr_pct']:>6.1f}% | {m['max_drawdown_pct']:>6.1f}% | "
        f"{m['calmar']:>7.3f} | {m['sharpe']:>7.3f} | {m['trades_per_year']:>10.1f} | "
        f"${end_balance:>10,.2f} | ${yr1_end:>9,.2f}"
    )


def main(skip_opt: bool = False, refresh_data: bool = False) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("TQQQ Round 2 -- Corrected Vol-Target + $1,000 Capital Simulation")
    print("=" * 70)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\n[1] Fetching data...")
    prices_raw = fetch_data(force_refresh=refresh_data)
    prices = align_to_spy(prices_raw)

    active = ["SPY", "QQQ", "TQQQ", "SQQQ", "UPRO", "BIL", "BSV", "IEF", "TLT"]
    validate(prices, active, start=START_DATE)
    print(f"  Data range: {prices.index[0].date()} -> {prices.index[-1].date()}")
    print(f"  Shape: {prices.shape}")

    # ── Signals (baseline params) ───────────────────────────────────────────
    print("\n[2] Computing signals...")
    signals_full = compute_signals(prices)
    signals = signals_full.loc[START_DATE:]
    prices_live = prices.loc[START_DATE:]

    vix = prices["^VIX"] if "^VIX" in prices.columns else None
    if vix is not None:
        vix = vix.loc[START_DATE:]

    # ── Main grid: 4 sizing variants x 3 bonds ──────────────────────────────
    print("\n[3] Running main R2 grid (4 sizing variants x 3 bonds)...")
    all_metrics: list[dict] = []
    all_equity: dict[str, pd.Series] = {}
    all_trades: dict[str, pd.DataFrame] = {}
    all_portfolios: dict[str, pd.DataFrame] = {}
    all_alloc: dict[str, pd.DataFrame] = {}

    for vt_label, target_vol in SIZING_VARIANTS.items():
        for bond in BOND_VARIANTS:
            label = f"{vt_label}-{bond}"
            print(f"  Running {label}...")
            sp = StrategyParams(target_vol=target_vol, bond=bond)
            alloc = compute_allocations(prices_live, signals, sp, vix=vix)
            port, trades = run_backtest(alloc, prices_live)
            eq = port["portfolio_value"]
            m = compute_metrics(eq, trades, label=label)
            all_equity[label] = eq
            all_trades[label] = trades
            all_portfolios[label] = port
            all_alloc[label] = alloc
            all_metrics.append(m)

    # ── Walk-forward validation for R2-VT55 across all 3 bonds ─────────────
    print("\n[4] Walk-forward validation (R2-VT55 across bonds)...")
    wf_rows = []
    bond_scores: dict[str, tuple[int, float]] = {}
    for bond in BOND_VARIANTS:
        label = f"R2-VT55-{bond}"
        wf = walk_forward_summary(all_equity[label], label, all_trades[label])
        if not wf:
            continue
        n_flags = len(wf.get("red_flags", []))
        oos_calmar = wf["OOS"]["calmar"]
        bond_scores[bond] = (n_flags, -oos_calmar)
        print(f"  {label}: IS CAGR={wf['IS']['cagr_pct']:.1f}% MDD={wf['IS']['max_drawdown_pct']:.1f}% | "
              f"OOS CAGR={wf['OOS']['cagr_pct']:.1f}% MDD={wf['OOS']['max_drawdown_pct']:.1f}% "
              f"Calmar={oos_calmar:.3f} | red flags: {n_flags}")
        for flag in wf.get("red_flags", []):
            print(f"    !! {flag}")
        row = {"label": label, **{f"IS_{k}": v for k, v in wf["IS"].items()},
               **{f"OOS_{k}": v for k, v in wf["OOS"].items()}}
        wf_rows.append(row)

    chosen_bond = min(bond_scores, key=lambda b: bond_scores[b]) if bond_scores else "TLT"
    print(f"\n  Most robust bond (fewest red flags, best OOS Calmar): {chosen_bond}")

    wf_df = pd.DataFrame(wf_rows)
    wf_df.to_csv(os.path.join(OUTPUT_DIR, "walk_forward_summary.csv"), index=False)

    base_label = f"R2-VT55-{chosen_bond}"

    # ── Optional buffer variant: R2-VT55-Buf2 ───────────────────────────────
    print("\n[5] Running optional R2-VT55-Buf2 (light +-2% buffer)...")
    sp_buf2 = StrategyParams(target_vol=TARGET_VOL, bond=chosen_bond,
                              buffer_up=0.02, buffer_dn=0.02)
    alloc_buf2 = compute_allocations(prices_live, signals, sp_buf2, vix=vix)
    port_buf2, trades_buf2 = run_backtest(alloc_buf2, prices_live)
    eq_buf2 = port_buf2["portfolio_value"]
    m_buf2 = compute_metrics(eq_buf2, trades_buf2, label="R2-VT55-Buf2")
    all_equity["R2-VT55-Buf2"] = eq_buf2
    all_metrics.append(m_buf2)

    # ── Structural tests T1-T4 vs R2-VT55 baseline ──────────────────────────
    print("\n[6] Running structural tests T1-T4...")

    # T1: EWMA realized vol (halflife=10)
    sp_t1 = StrategyParams(target_vol=TARGET_VOL, bond=chosen_bond, use_ewma_vol=True)
    alloc_t1 = compute_allocations(prices_live, signals, sp_t1, vix=vix)
    port_t1, trades_t1 = run_backtest(alloc_t1, prices_live)
    m_t1 = compute_metrics(port_t1["portfolio_value"], trades_t1, label="T1-EWMA-Vol")
    all_equity["T1-EWMA-Vol"] = port_t1["portfolio_value"]
    all_metrics.append(m_t1)

    # T2: regime-dependent target vol (bull=0.65, bear=0.35)
    sp_t2 = StrategyParams(bond=chosen_bond, target_vol_bull=0.65, target_vol_bear=0.35)
    alloc_t2 = compute_allocations(prices_live, signals, sp_t2, vix=vix)
    port_t2, trades_t2 = run_backtest(alloc_t2, prices_live)
    m_t2 = compute_metrics(port_t2["portfolio_value"], trades_t2, label="T2-RegimeVol")
    all_equity["T2-RegimeVol"] = port_t2["portfolio_value"]
    all_metrics.append(m_t2)

    # T3: VIX > 35 hard filter -> BIL
    if vix is not None:
        sp_t3 = StrategyParams(target_vol=TARGET_VOL, bond=chosen_bond, vix_hard_filter=35.0)
        alloc_t3 = compute_allocations(prices_live, signals, sp_t3, vix=vix)
        port_t3, trades_t3 = run_backtest(alloc_t3, prices_live)
        m_t3 = compute_metrics(port_t3["portfolio_value"], trades_t3, label="T3-VIXFilter")
        all_equity["T3-VIXFilter"] = port_t3["portfolio_value"]
        all_metrics.append(m_t3)
    else:
        print("  T3 (VIX hard filter): ^VIX not available -- skipping")

    # T4: vol targeting applied to defensive leg too
    sp_t4 = StrategyParams(target_vol=TARGET_VOL, bond=chosen_bond, apply_vol_to_defensive=True)
    alloc_t4 = compute_allocations(prices_live, signals, sp_t4, vix=vix)
    port_t4, trades_t4 = run_backtest(alloc_t4, prices_live)
    m_t4 = compute_metrics(port_t4["portfolio_value"], trades_t4, label="T4-VolDefensive")
    all_equity["T4-VolDefensive"] = port_t4["portfolio_value"]
    all_metrics.append(m_t4)

    # ── FTLT-V2 (faithful 80/20 original) comparison arm ────────────────────
    print("\n[7] Running FTLT-V2 (80/20 original) comparison arm...")
    alloc_ftlt = compute_allocations_ftlt(prices_live, signals, bond=chosen_bond)
    port_ftlt, trades_ftlt = run_backtest(alloc_ftlt, prices_live)
    eq_ftlt = port_ftlt["portfolio_value"]
    m_ftlt = compute_metrics(eq_ftlt, trades_ftlt, label="FTLT-V2")
    all_equity["FTLT-V2"] = eq_ftlt
    all_trades["FTLT-V2"] = trades_ftlt
    all_metrics.append(m_ftlt)

    # ── Benchmarks ───────────────────────────────────────────────────────────
    print("\n[8] Running benchmarks...")
    for bm in ["QQQ", "TQQQ", "SPY"]:
        eq_bm = run_benchmark(bm, prices_live, start=START_DATE)
        all_equity[f"{bm} B&H"] = eq_bm
        all_metrics.append(compute_metrics(eq_bm, label=f"{bm} B&H"))

    eq_6040 = run_60_40(prices_live, start=START_DATE)
    all_equity["60/40"] = eq_6040
    all_metrics.append(compute_metrics(eq_6040, label="60/40"))

    # ── $1,000 capital simulation: R2-VT55 (chosen bond) vs FTLT-V2 ─────────
    print(f"\n[9] Running $1,000 capital simulation for {base_label} and FTLT-V2...")
    cap_port_r2, cap_trades_r2 = run_capital_sim(all_alloc[base_label], prices_live)
    cap_port_ftlt, cap_trades_ftlt = run_capital_sim(alloc_ftlt, prices_live)

    monthly_r2 = monthly_table(cap_port_r2, cap_trades_r2)
    annual_r2 = annual_table(cap_port_r2)
    monthly_ftlt = monthly_table(cap_port_ftlt, cap_trades_ftlt)
    annual_ftlt = annual_table(cap_port_ftlt)
    comp = comparison_table(cap_port_r2, cap_port_ftlt, "R2VT55", "FTLTV2")

    monthly_r2.to_csv(os.path.join(OUTPUT_DIR, f"monthly_returns_R2VT55_{chosen_bond}.csv"), index=False)
    annual_r2.to_csv(os.path.join(OUTPUT_DIR, f"annual_returns_R2VT55_{chosen_bond}.csv"), index=False)
    monthly_ftlt.to_csv(os.path.join(OUTPUT_DIR, "monthly_returns_FTLTV2.csv"), index=False)
    annual_ftlt.to_csv(os.path.join(OUTPUT_DIR, "annual_returns_FTLTV2.csv"), index=False)
    comp.to_csv(os.path.join(OUTPUT_DIR, "comparison_R2VT55_vs_FTLTV2.csv"), index=False)
    cap_trades_r2.to_csv(os.path.join(OUTPUT_DIR, "trade_log.csv"), index=False)

    end_balance_r2 = cap_port_r2["portfolio_value"].iloc[-1]
    end_balance_ftlt = cap_port_ftlt["portfolio_value"].iloc[-1]
    yr1_end_r2 = annual_r2.iloc[0]["end_balance"] if len(annual_r2) > 0 else STARTING_CAPITAL
    yr1_end_ftlt = annual_ftlt.iloc[0]["end_balance"] if len(annual_ftlt) > 0 else STARTING_CAPITAL

    # ── Optimization ─────────────────────────────────────────────────────────
    if not skip_opt:
        print("\n[10] Running optimization (this may take several minutes)...")
        baseline_cagr = calc_cagr(all_equity[base_label])
        top10, overfit = optimize(prices, bond=chosen_bond, baseline_cagr=baseline_cagr, verbose=True)
        if not top10.empty:
            top10.to_csv(os.path.join(OUTPUT_DIR, "optimization_top10.csv"), index=False)
            print(f"  Top-10 OOS Calmar range: {top10['calmar'].min():.3f}-{top10['calmar'].max():.3f}")
        if not overfit.empty:
            overfit.to_csv(os.path.join(OUTPUT_DIR, "optimization_overfit_appendix.csv"), index=False)
            print(f"  Overfit combos excluded: {len(overfit)}")
    else:
        print("\n[10] Optimization skipped (--skip-opt)")

    # ── Save results summary ─────────────────────────────────────────────────
    print("\n[11] Saving results summary...")
    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "results_summary.csv"), index=False)

    # ── Charts ────────────────────────────────────────────────────────────────
    print("\n[12] Generating charts...")
    dollar_equity_dict = {
        f"R2-VT55-{chosen_bond}": cap_port_r2["portfolio_value"],
        "FTLT-V2": cap_port_ftlt["portfolio_value"],
        "QQQ B&H": run_benchmark("QQQ", prices_live, start=START_DATE, initial_capital=STARTING_CAPITAL),
        "TQQQ B&H": run_benchmark("TQQQ", prices_live, start=START_DATE, initial_capital=STARTING_CAPITAL),
        "60/40": run_60_40(prices_live, start=START_DATE, initial_capital=STARTING_CAPITAL),
    }
    dollar_equity_curves(dollar_equity_dict)

    monthly_dollar_dict = {
        f"R2-VT55-{chosen_bond}": monthly_r2,
        "FTLT-V2": monthly_ftlt,
    }
    monthly_dollar_bars(monthly_dollar_dict)

    main_curves = {k: v for k, v in all_equity.items()
                   if k.startswith("R2-VT55-") or k in ("FTLT-V2", "QQQ B&H", "TQQQ B&H", "SPY B&H", "60/40")}
    drawdown_chart(main_curves)
    annual_returns_heatmap(main_curves)
    rolling_metrics_chart(main_curves)

    vol_targeting_diagnostic(signals, all_alloc[base_label])

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + RISK_HEADER)
    print(RISK_SEP)
    priority_order = (
        [f"R2-NoVT-{b}" for b in BOND_VARIANTS]
        + [f"R2-VT45-{b}" for b in BOND_VARIANTS]
        + [f"R2-VT55-{b}" for b in BOND_VARIANTS]
        + [f"R2-VT65-{b}" for b in BOND_VARIANTS]
        + ["R2-VT55-Buf2", "T1-EWMA-Vol", "T2-RegimeVol", "T3-VIXFilter", "T4-VolDefensive",
           "FTLT-V2", "QQQ B&H", "TQQQ B&H", "SPY B&H", "60/40"]
    )
    printed = set()
    for lbl in priority_order:
        m = next((x for x in all_metrics if x["label"] == lbl), None)
        if m:
            print(fmt_risk_row(m))
            printed.add(lbl)
    for m in all_metrics:
        if m["label"] not in printed:
            print(fmt_risk_row(m))
    print(RISK_SEP)

    print("\n--- $1,000 Capital Simulation: Headline Comparison ---")
    print(DOLLAR_HEADER)
    print(DOLLAR_SEP)
    m_base = next(m for m in all_metrics if m["label"] == base_label)
    print(fmt_dollar_row(base_label, m_base, end_balance_r2, yr1_end_r2))
    print(fmt_dollar_row("FTLT-V2", m_ftlt, end_balance_ftlt, yr1_end_ftlt))
    print(DOLLAR_SEP)

    print(f"\nOutput files written to: {OUTPUT_DIR}/")
    print("Done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-opt", action="store_true", help="Skip optimization phase")
    parser.add_argument("--refresh-data", action="store_true", help="Force yfinance re-download")
    args = parser.parse_args()
    main(skip_opt=args.skip_opt, refresh_data=args.refresh_data)
