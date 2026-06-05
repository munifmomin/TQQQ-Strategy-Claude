# TQQQ Buffered + Volatility-Targeted Backtest

A professional-grade backtest for the **TQQQ BVT** strategy — an improved variant of the "TQQQ For The Long Term" (FTLT) strategy by Pawel. This implementation adds four theory-grounded improvements and is designed to be directly comparable to the FTLT backtest.

## Differences from FTLT Baseline

| Feature | FTLT (Pawel) | BVT (This repo) |
|---|---|---|
| Trend filter | Binary 200D SMA cross | **Buffered** 200D SMA (±5%/3% dead zone + hysteresis) |
| Signal source | TQQQ/UPRO prices | **QQQ/SPY** (unleveraged index) |
| Overbought leg | UVXY | **BIL** (T-bills) |
| Position sizing | Fixed allocation brackets | **Volatility targeting** (target vol / realized vol) |

The buffer suppresses whipsaw. Computing signals on unleveraged QQQ/SPY avoids path-dependency noise. Removing UVXY eliminates the volatility-lottery leg. Volatility targeting smoothly scales exposure when markets are turbulent.

## Project Structure

```
tqqq_bvt/
  config.py          – all constants (no magic numbers elsewhere)
  data.py            – yfinance fetch + cache + validation
  signals.py         – SMA, Wilder RSI, realized vol, buffer band calc
  regime.py          – STATEFUL buffered regime machine (hysteresis)
  strategy.py        – decision logic + vol-target weight computation
  backtest.py        – event-driven engine; fractional multi-instrument holdings
  metrics.py         – CAGR, MDD, Calmar, Sortino, Sharpe, trade stats
  optimize.py        – two-stage coarse→fine grid; IS/OOS overfitting check
  walk_forward.py    – IS/OOS split + rolling windows + red-flag detection
  charts.py          – all output figures
  main.py            – orchestrator entry point
```

## Usage

```bash
pip install -r requirements.txt

# Full run (includes optimization — slow)
python -m tqqq_bvt.main

# Fast run (skip optimization)
python -m tqqq_bvt.main --skip-opt

# Force data refresh
python -m tqqq_bvt.main --skip-opt --refresh-data
```

## Key Design Decisions

### Stateful Regime Machine
The buffered regime is **stateful** — it carries `prior_regime` through the dead zone. A unit test verifies this: a synthetic SPY series that crosses up through +5%, drifts in the dead zone, crosses down through −3%, and the regime flips exactly at the bands and holds in between.

### Volatility Targeting
`weight = clamp(TARGET_VOL / realized_vol, 0, 1)` applied daily to risk-on positions only. Defensive/cash positions are always held at full size. Rebalancing triggered only when weight changes by ≥ `REBAL_THRESHOLD` (10%) to avoid churning on noise. All vol-resize trades tagged separately from signal-change trades in the log.

### Cost Model
Identical to FTLT spec: 5bps slippage per side (liquid), 10bps (SQQQ/TECL/UPRO). No commissions. ERs already embedded in yfinance adjusted prices — not double-applied.

## Output Files (`output_bvt/`)

| File | Description |
|---|---|
| `results_summary.csv` | All variants × bonds × metrics |
| `equity_curves.png` | All variants + benchmarks, log scale |
| `drawdown_chart.png` | Drawdown over time |
| `annual_returns_heatmap.png` | Year × variant heatmap |
| `rolling_metrics.png` | Rolling 3Y CAGR and MDD |
| `optimization_surface.png` | Calmar across param slices |
| `trade_log.csv` | Every trade with SIGNAL_CHANGE / VOL_RESIZE tag |
| `regime_breakdown.csv` | Metrics by buffered regime |
| `walk_forward_summary.csv` | IS vs OOS |
| `vol_targeting_diagnostic.png` | Realized vol vs applied weight over time |

## Sizing Variants

| Variant | Target Vol | Description |
|---|---|---|
| VT-OFF | — | Binary 100%/0% (isolates buffer+signal effect) |
| VT-30 | 30% | Aggressive de-sizing |
| VT-40 | 40% | **Baseline** |
| VT-50 | 50% | Light de-sizing |

## Structural Tests (S1–S5)

| ID | Test | Purpose |
|---|---|---|
| S1 | Re-add UVXY on overbought | Verify removing UVXY was correct |
| S2 | Regime-dependent target vol | Lower target vol in bear |
| S3 | VIX > 40 hard filter | Compare to smooth vol targeting |
| S4 | BULL transitional → TQQQ | Test Pawel's choice vs BIL |
| S5 | 2-day regime confirmation | Belt-and-suspenders vs added lag |
