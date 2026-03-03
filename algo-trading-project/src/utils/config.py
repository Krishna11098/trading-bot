"""
Global configuration for trading system.
"""

from pathlib import Path

# ----------------------------
# PATHS
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDICATORS_DIR = DATA_DIR / "indicators"
SIGNALS_DIR = DATA_DIR / "signals"
MODELS_DIR = BASE_DIR / "models"

# ----------------------------
# TICKERS (Kaggle India indices)
# ----------------------------
DEFAULT_TICKERS = [
    "NIFTY BANK",
    "NIFTY COMMODITIES",
    "NIFTY CONSUMPTION",
    "NIFTY FIN SERVICE",
    "NIFTY INDIA MFG",
    "INDIA VIX",
]

# ----------------------------
# DATA SPLIT DATES
# ----------------------------
TRAIN_START = "2015-01-01"
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2024-12-31"

# ----------------------------
# CAPITAL & COSTS
# ----------------------------
INITIAL_CAPITAL = 100000.0
COMMISSION_RATE = 0.0005        # 0.05%

# ----------------------------
# RISK MANAGEMENT
# ----------------------------
RISK_PER_TRADE = 0.01           # 1% risk per trade
STOP_LOSS_PCT = 0.003           # 0.3%
TAKE_PROFIT_PCT = 0.004         # 0.4%

# ----------------------------
# SCALPING PARAMETERS
# ----------------------------
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# ----------------------------
# ML PARAMETERS
# ----------------------------
ML_DEFAULT_WEIGHT = 0.6
MIN_PROB_BUY = 0.55
MAX_PROB_SELL = 0.45
TEST_SIZE = 0.2             # For sklearn (fallback)
RANDOM_STATE = 42
N_ESTIMATORS = 100

# ----------------------------
# BACKTEST / EXECUTION
# ----------------------------
TRADING_DAYS = 252
