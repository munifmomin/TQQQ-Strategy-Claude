# TQQQ Round 2 — Corrected Vol-Target + $1,000 Capital Simulation

A second-generation backtest for the TQQQ rotation strategy family. This round
fixes a critical bug in the volatility-targeting overlay and adds a
dollar-denominated ($1,000 starting capital) simulation layer so results map
directly onto "if I actually traded this with real money" expectations.

## The Critical Fix: Vol Targeting on TQQQ, Not QQQ

The prior (BVT) spec computed realized volatility on **QQQ** (the unleveraged
index) and used it to scale TQQQ exposure. QQQ's annualized vol typically runs
10–30%, almost always *below* any sane `TARGET_VOL` (40–55%), so
`weight = clamp(target_vol / realized_vol, 0, 1)` was permanently clamped at
`1.0` — the overlay never actually engaged.

**Round 2 measures realized volatility on TQQQ itself** (`tqqq_realized_vol` /
`tqqq_realized_vol_ewma` in `signals.py`). TQQQ's vol runs 30–90%+, so a
target near 0.55 actually bites — exposure gets scaled down in real high-vol
regimes (2018-Q4, 2020, 2022). `output_r2/vol_targeting_diagnostic.png` shows
the applied weight visibly dropping below 1.0 in those windows; if it were
ever pinned at 1.0 across the whole history, the bug would be back.

## Differences from Prior Specs

| Feature | BVT (Round 1) | FTLT-V2 (original, comparison arm) | R2 (this build) |
|---|---|---|---|
| Trend filter | Buffered 200D SMA (stateful) | Raw 200D SMA crossover | **Raw 200D SMA crossover (stateless)**; optional ±2% buffer variant `R2-VT55-Buf2` |
| Vol measured on | QQQ (inert) | n/a | **TQQQ (engages)** |
| Overbought leg | BIL | UVXY | BIL |
| Sizing | Vol target on QQQ vol (broken) | Fixed 80% / 20% BIL | Vol target on TQQQ vol — `R2-NoVT` (100/0 binary), `R2-VT45`, `R2-VT55` (baseline), `R2-VT65` |
| Capital tracking | $100,000 index only | $100,000 index only | **Both**: $100,000 index (`backtest.py`) for risk metrics + **$1,000 real-dollar sim** (`capital_sim.py`) for monthly/annual dollar tables |

## Project Structure

```
tqqq_r2/
  config.py          – all constants (no magic numbers elsewhere)
  data.py            – yfinance fetch + cache + alignment/validation
  signals.py         – SMA, Wilder RSI, realized vol (simple + EWMA on TQQQ)
  regime.py          – stateful buffered regime (only for R2-VT55-Buf2)
  strategy.py        – stateless decision tree + corrected vol-target sizing
  ftlt_original.py   – faithful FTLT-V2 (80/20, UVXY-on-overbought) comparison arm
  backtest.py        – $100,000 index-based engine for risk-adjusted metrics
  capital_sim.py      – $1,000 real-dollar simulation, monthly/annual tables
  metrics.py         – CAGR, MDD, Calmar, Sharpe, Sortino, trade stats
  optimize.py        – two-stage IS/OOS grid search with overfit guardrail
  walk_forward.py    – IS/OOS split, rolling windows, red-flag detection
  charts.py          – all output figures
  main.py            – orchestrator entry point
```

## Usage

```bash
pip install -r requirements.txt

# Full run (includes optimization — slow)
python -m tqqq_r2.main

# Fast run (skip optimization)
python -m tqqq_r2.main --skip-opt

# Force data refresh
python -m tqqq_r2.main --skip-opt --refresh-data
```

## Sizing Variants (main grid: 4 × 3 bonds = 12 runs)

| Variant | Target Vol | Description |
|---|---|---|
| R2-NoVT | — | Binary 100%/0% (vol targeting off) |
| R2-VT45 | 45% | More aggressive de-sizing |
| R2-VT55 | 55% | **Baseline** |
| R2-VT65 | 65% | Lighter de-sizing |

Each is run against three defensive bonds: BSV, IEF, TLT. Walk-forward
validation (IS 2011-10-01→2018-12-31, OOS 2019-01-01→present) on R2-VT55
identifies the most robust bond (fewest red flags, best OOS Calmar); that
bond is used for the headline $1,000 capital simulation, the structural
tests (T1–T4), and the optional buffer variant.

## Optional Variant: R2-VT55-Buf2

A light ±2% / −2% buffer with hysteresis around the 200D SMA, applied to the
baseline R2-VT55 sizing. Not part of the main grid — run once for comparison.

## Structural Tests (T1–T4)

| ID | Test | Purpose |
|---|---|---|
| T1 | EWMA realized vol (halflife=10) | Faster-reacting vol estimate vs. simple rolling std |
| T2 | Regime-dependent target vol (bull=0.65, bear=0.35) | More aggressive in confirmed uptrends, more defensive in bear |
| T3 | VIX > 35 hard filter → BIL | Smooth vol targeting vs. a hard panic-button override |
| T4 | Vol targeting applied to defensive leg too | Does scaling the bond/SQQQ leg change the picture? |

## $1,000 Capital Simulation

`capital_sim.py` tracks the strategy in real dollars (fractional shares by
default), starting from $1,000. Rebalances apply target weights to *current*
portfolio value (correct compounding), with the same slippage/commission cost
model as the index backtest. Outputs:

- `monthly_returns_R2VT55_<bond>.csv` / `monthly_returns_FTLTV2.csv` — start/end
  balance, $ change, % return, rotation count, dominant holding, per month
- `annual_returns_R2VT55_<bond>.csv` / `annual_returns_FTLTV2.csv` — annual $ tables
  with best/worst month and max intra-year drawdown in dollars
- `comparison_R2VT55_vs_FTLTV2.csv` — month-by-month side-by-side balances
- `dollar_equity_curves.png` — $1,000 equity curves (linear + log) for R2-VT55,
  FTLT-V2, and benchmarks
- `monthly_dollar_bars.png` — green/red monthly $ change bars

## ⚠ Setting Expectations: Read This Before Looking at the Numbers

This strategy targets roughly **55–70% CAGR** with drawdowns that can exceed
**-50%**. On $1,000 of starting capital, that means:

- After year 1, $1,000 might grow to roughly **$1,600–$1,700** — a meaningful
  gain, but not life-changing money.
- **Monthly returns are small in dollar terms and lumpy.** A "good month" on
  $1,000 might be +$80. A bad month can be -$150 or worse. There will be
  multiple **red months in a row**, including during the strategy's best
  multi-year stretches.
- **Do not annualize a single month's dollar gain into an income projection
  anywhere.** A +6% month is not "$60/month forever" or "$720/year" — monthly
  returns compound non-linearly and are dominated by a handful of very large
  up/down days. The CAGR and the $1,000→end-balance tables in this output are
  the only honest way to reason about long-run growth; a single month's $
  change is not.
- Drawdowns of -50% to -80% on the index-based ($100k) equity curves
  correspond to the *same percentage* drawdown on the $1,000 sim — i.e., a
  $1,000 account can fall to $200–$500 during the worst stretches before
  recovering. Position sizing and risk tolerance should be set with this in
  mind, not with the CAGR alone.

## Output Files (`output_r2/`)

| File | Description |
|---|---|
| `results_summary.csv` | All variants × bonds × structural tests × benchmarks |
| `dollar_equity_curves.png` | $1,000 equity curves, R2-VT55 vs FTLT-V2 vs benchmarks (linear + log) |
| `monthly_dollar_bars.png` | Monthly $ change, green/red |
| `drawdown_chart.png` | Drawdown over time |
| `annual_returns_heatmap.png` | Year × variant heatmap |
| `rolling_metrics.png` | Rolling 3Y CAGR and MDD |
| `vol_targeting_diagnostic.png` | **Realized TQQQ vol vs applied weight — must show weight varying below 1.0** |
| `monthly_returns_R2VT55_<bond>.csv`, `annual_returns_R2VT55_<bond>.csv` | $1,000 sim tables for the headline R2-VT55 variant |
| `monthly_returns_FTLTV2.csv`, `annual_returns_FTLTV2.csv` | $1,000 sim tables for the FTLT-V2 comparison arm |
| `comparison_R2VT55_vs_FTLTV2.csv` | Month-by-month side-by-side $ balances |
| `trade_log.csv` | Every $1,000-sim trade (R2-VT55 headline) |
| `walk_forward_summary.csv` | IS vs OOS for R2-VT55 across all 3 bonds |
| `optimization_top10.csv`, `optimization_overfit_appendix.csv` | Two-stage optimization results (when not skipped) |
