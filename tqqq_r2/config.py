"""
All configuration constants for the TQQQ Round-2 (corrected vol-target + $1k capital sim).
No magic numbers should appear in any other module.
"""

import os

# ── Date range ────────────────────────────────────────────────────────────────
START_DATE = "2011-10-01"
END_DATE = None  # None → use today

# ── Tickers ───────────────────────────────────────────────────────────────────
# NOTE: UVXY added beyond the spec's section 3 list because the FTLT-V2
# comparison arm (Section 6) is faithful to the original strategy, which
# rotates into UVXY on overbought. Not used by any R2-* variant.
# ^VIX added for the T3 structural test (VIX > 35 hard filter -> BIL).
TICKERS = ["SPY", "TQQQ", "SQQQ", "UPRO", "BSV", "IEF", "TLT", "BIL", "QQQ", "UVXY", "^VIX"]

SIGNAL_TICKER_NASDAQ = "QQQ"
SIGNAL_TICKER_BROAD = "SPY"
VOL_TICKER = "TQQQ"   # CRITICAL FIX: realized vol measured on TQQQ, not QQQ

BOND_VARIANTS = ["BSV", "IEF", "TLT"]

# ── Trend filter (raw crossover, no buffer in baseline) ──────────────────────
SMA_WINDOW = 200

# ── RSI parameters ────────────────────────────────────────────────────────────
RSI_WINDOW = 10
OVERBOUGHT_RSI = 79
TQQQ_OVERSOLD_RSI = 31
SPY_OVERSOLD_RSI = 30

# ── Volatility targeting (on TQQQ) ────────────────────────────────────────────
VOL_LOOKBACK = 20
TARGET_VOL = 0.55          # baseline R2-VT55
REBAL_THRESHOLD = 0.10
EWMA_HALFLIFE = 10          # for T1 structural test

# ── Optional buffer variant (R2-VT55-Buf2 only, NOT baseline) ────────────────
LIGHT_BUFFER_UP = 0.02
LIGHT_BUFFER_DN = 0.02

# ── Cost model ────────────────────────────────────────────────────────────────
COMMISSION_PER_TRADE = 0.0
SLIPPAGE_BPS = 5
SLIPPAGE_BPS_ILLIQUID = 10
ILLIQUID_TICKERS = {"SQQQ", "UPRO"}

# ── Portfolio (index-based backtest) ─────────────────────────────────────────
INITIAL_CAPITAL = 100_000.0

# ── Capital simulation (the headline $1,000 deliverable) ─────────────────────
STARTING_CAPITAL = 1000.00
FRACTIONAL_SHARES = True
MIN_TRADE_DOLLARS = 1.00

# ── Sizing variants ───────────────────────────────────────────────────────────
SIZING_VARIANTS = {
    "R2-NoVT": None,
    "R2-VT45": 0.45,
    "R2-VT55": 0.55,
    "R2-VT65": 0.65,
}

# ── FTLT-V2 (faithful original, comparison arm) ───────────────────────────────
FTLT_V2_RISK_WEIGHT = 0.80   # 80% signal instrument / 20% cash (BIL)
FTLT_V2_CASH_WEIGHT = 0.20

# ── Walk-forward ──────────────────────────────────────────────────────────────
WF_IN_SAMPLE_END = "2018-12-31"
WF_OUT_SAMPLE_START = "2019-01-01"
WF_ROLLING_YEARS = 3
WF_STEP_MONTHS = 6

# ── Optimization (coarse grid) ────────────────────────────────────────────────
OPT_PARAMS_COARSE = {
    "sma_window":        list(range(150, 260, 10)),
    "target_vol":        [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
    "vol_lookback":      [10, 15, 20, 25, 30, 35, 40],
    "overbought_rsi":    list(range(70, 86, 1)),
    "tqqq_oversold_rsi": list(range(25, 41, 1)),
    "spy_oversold_rsi":  list(range(20, 41, 1)),
    "rsi_window":        list(range(7, 15, 1)),
}

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CACHE_PATH = os.path.join(BASE_DIR, "data_cache.parquet")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_r2")
