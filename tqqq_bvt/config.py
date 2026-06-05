"""
All configuration constants for the TQQQ Buffered + Volatility-Targeted backtest.
No magic numbers should appear in any other module.
"""

# ── Date range ────────────────────────────────────────────────────────────────
START_DATE = "2011-10-01"
END_DATE = None  # None → use today

# ── Tickers ───────────────────────────────────────────────────────────────────
TICKERS = ["SPY", "TQQQ", "SQQQ", "UVXY", "UPRO", "TECL", "BSV", "IEF", "TLT", "BIL", "QQQ"]

# Signal sources (unleveraged)
SIGNAL_TICKER_NASDAQ = "QQQ"
SIGNAL_TICKER_BROAD = "SPY"

# Bond variants
BOND_VARIANTS = ["BSV", "IEF", "TLT"]

# ── Trend filter ──────────────────────────────────────────────────────────────
SMA_WINDOW = 200          # SPY SMA window
BUFFER_UP = 0.05          # must be 5% above SMA to confirm uptrend
BUFFER_DN = 0.03          # must be 3% below SMA to confirm downtrend

# ── RSI parameters ────────────────────────────────────────────────────────────
RSI_WINDOW = 10           # Wilder RSI lookback
OVERBOUGHT_RSI = 79       # BULL regime: rotate to BIL
TQQQ_OVERSOLD_RSI = 31    # BEAR regime: buy TQQQ bounce
SPY_OVERSOLD_RSI = 30     # BEAR regime: buy UPRO bounce

# ── Volatility targeting ──────────────────────────────────────────────────────
VOL_LOOKBACK = 20         # days for realized vol calculation
TARGET_VOL = 0.40         # baseline annualized target vol (VT-40)
REBAL_THRESHOLD = 0.10    # min weight change to trigger vol-resize trade

# ── Cost model ────────────────────────────────────────────────────────────────
COMMISSION_PER_TRADE = 0.0
SLIPPAGE_BPS = 5          # bps per side for liquid instruments
SLIPPAGE_BPS_ILLIQUID = 10  # bps per side for SQQQ, TECL, UPRO
ILLIQUID_TICKERS = {"SQQQ", "TECL", "UPRO"}

# ── Portfolio ─────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 100_000.0

# ── Sizing variants ───────────────────────────────────────────────────────────
SIZING_VARIANTS = {
    "VT-OFF": None,   # binary 100%/0%
    "VT-30": 0.30,
    "VT-40": 0.40,
    "VT-50": 0.50,
}

# ── Walk-forward ──────────────────────────────────────────────────────────────
WF_IN_SAMPLE_END = "2018-12-31"
WF_OUT_SAMPLE_START = "2019-01-01"
WF_ROLLING_YEARS = 3
WF_STEP_MONTHS = 6

# ── Optimization (coarse stage) ───────────────────────────────────────────────
OPT_PARAMS_COARSE = {
    "sma_window":        list(range(150, 260, 20)),
    "buffer_up":         [0.00, 0.03, 0.05, 0.07, 0.10],
    "buffer_dn":         [0.00, 0.02, 0.03, 0.05, 0.08, 0.10],
    "target_vol":        [0.25, 0.35, 0.40, 0.50, 0.60],
    "vol_lookback":      [10, 20, 30, 40],
    "overbought_rsi":    list(range(70, 86, 5)),
    "tqqq_oversold_rsi": list(range(25, 42, 5)),
    "spy_oversold_rsi":  list(range(20, 42, 5)),
    "rsi_window":        [7, 10, 14],
}

# Fine stage ranges (populated after coarse identifies promising region)
OPT_PARAMS_FINE = {
    "sma_window":        list(range(150, 260, 10)),
    "buffer_up":         [round(x * 0.01, 2) for x in range(0, 11)],
    "buffer_dn":         [round(x * 0.01, 2) for x in range(0, 11)],
    "target_vol":        [round(x * 0.05, 2) for x in range(5, 13)],
    "vol_lookback":      list(range(10, 45, 5)),
    "overbought_rsi":    list(range(70, 86, 1)),
    "tqqq_oversold_rsi": list(range(25, 41, 1)),
    "spy_oversold_rsi":  list(range(20, 41, 1)),
    "rsi_window":        list(range(7, 15, 1)),
}

# ── Paths ─────────────────────────────────────────────────────────────────────
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CACHE_PATH = os.path.join(BASE_DIR, "data_cache.parquet")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_bvt")
