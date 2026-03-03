"""
Global configuration for trading system.

Provides centralized configuration management with type safety and validation.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import os


# ============================================================
# PATH CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDICATORS_DIR = DATA_DIR / "indicators"
SIGNALS_DIR = DATA_DIR / "signals"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"


# ============================================================
# TICKER CONFIGURATION
# ============================================================
DEFAULT_TICKERS = [
    "NIFTY BANK",
    "NIFTY COMMODITIES",
    "NIFTY CONSUMPTION",
    "NIFTY FIN SERVICE",
    "NIFTY INDIA MFG",
    "INDIA VIX",
]


# ============================================================
# TIME PERIODS
# ============================================================
TRAIN_START = "2015-01-01"
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2024-12-31"


# ============================================================
# CAPITAL & TRANSACTION COSTS
# ============================================================
INITIAL_CAPITAL = 100_000.0
COMMISSION_RATE = 0.0005        # 0.05%
SLIPPAGE = 0.0001               # 0.01%


# ============================================================
# RISK MANAGEMENT PARAMETERS
# ============================================================
RISK_PER_TRADE = 0.01           # 1% risk per trade
MAX_POSITION_SIZE = 0.25        # Max 25% of capital in single position
STOP_LOSS_PCT = 0.003           # 0.3%
TAKE_PROFIT_PCT = 0.004         # 0.4%
MAX_DRAWDOWN_LIMIT = 0.20       # 20% max drawdown
DAILY_LOSS_LIMIT = 0.05         # 5% daily loss limit
MAX_CONSECUTIVE_LOSSES = 5      # Stop trading after 5 consecutive losses


# ============================================================
# TECHNICAL INDICATOR PARAMETERS
# ============================================================
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

SMA_SHORT = 20
SMA_LONG = 50
EMA_SHORT = 12
EMA_LONG = 26

BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0

ATR_PERIOD = 14


# ============================================================
# ML MODEL PARAMETERS
# ============================================================
ML_DEFAULT_WEIGHT = 0.6
MIN_PROB_BUY = 0.55
MAX_PROB_SELL = 0.45
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100
CV_FOLDS = 5


# ============================================================
# EXECUTION PARAMETERS
# ============================================================
TRADING_DAYS = 252
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30


# ============================================================
# CONFIGURATION CLASS
# ============================================================
@dataclass
class TradingConfig:
    """
    Centralized trading configuration with validation.
    
    Usage:
        config = TradingConfig()
        config = TradingConfig(initial_capital=500_000)
    """
    # Capital
    initial_capital: float = INITIAL_CAPITAL
    commission_rate: float = COMMISSION_RATE
    slippage: float = SLIPPAGE
    
    # Risk management
    risk_per_trade: float = RISK_PER_TRADE
    max_position_size: float = MAX_POSITION_SIZE
    stop_loss_pct: float = STOP_LOSS_PCT
    take_profit_pct: float = TAKE_PROFIT_PCT
    max_drawdown_limit: float = MAX_DRAWDOWN_LIMIT
    daily_loss_limit: float = DAILY_LOSS_LIMIT
    
    # ML parameters
    ml_weight: float = ML_DEFAULT_WEIGHT
    min_prob_buy: float = MIN_PROB_BUY
    max_prob_sell: float = MAX_PROB_SELL
    
    # Technical indicators
    rsi_oversold: int = RSI_OVERSOLD
    rsi_overbought: int = RSI_OVERBOUGHT
    
    def __post_init__(self):
        """Validate configuration values."""
        self._validate()
    
    def _validate(self):
        """Run validation checks on config values."""
        assert 0 < self.risk_per_trade <= 0.1, "Risk per trade must be 0-10%"
        assert 0 < self.max_position_size <= 1.0, "Max position size must be 0-100%"
        assert 0 <= self.commission_rate <= 0.01, "Commission rate seems too high"
        assert 0 < self.stop_loss_pct <= 0.1, "Stop loss must be 0-10%"
        assert 0 < self.take_profit_pct <= 0.2, "Take profit must be 0-20%"
        assert 0 <= self.ml_weight <= 1.0, "ML weight must be 0-1"
        assert 0 <= self.rsi_oversold < self.rsi_overbought <= 100, "Invalid RSI bounds"
    
    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio from SL and TP."""
        return self.take_profit_pct / self.stop_loss_pct
    
    def to_dict(self) -> Dict[str, Any]:
        """Export config as dictionary."""
        return {
            "initial_capital": self.initial_capital,
            "commission_rate": self.commission_rate,
            "risk_per_trade": self.risk_per_trade,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "risk_reward_ratio": self.risk_reward_ratio,
            "ml_weight": self.ml_weight,
        }

    @classmethod
    def from_env(cls) -> "TradingConfig":
        """Create config from environment variables."""
        return cls(
            initial_capital=float(os.getenv("TRADING_CAPITAL", INITIAL_CAPITAL)),
            risk_per_trade=float(os.getenv("RISK_PER_TRADE", RISK_PER_TRADE)),
            stop_loss_pct=float(os.getenv("STOP_LOSS_PCT", STOP_LOSS_PCT)),
            take_profit_pct=float(os.getenv("TAKE_PROFIT_PCT", TAKE_PROFIT_PCT)),
        )


def get_config() -> Dict[str, Any]:
    """
    Get configuration as dictionary (for API compatibility).
    
    Returns:
        Configuration dictionary with paths and settings
    """
    return {
        "paths": {
            "base": str(BASE_DIR),
            "data": str(DATA_DIR),
            "raw": str(RAW_DATA_DIR),
            "processed": str(PROCESSED_DATA_DIR),
            "indicators": str(INDICATORS_DIR),
            "signals": str(SIGNALS_DIR),
            "models": str(MODELS_DIR),
            "logs": str(LOGS_DIR),
        },
        "tickers": DEFAULT_TICKERS,
        "periods": {
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "test_start": TEST_START,
            "test_end": TEST_END,
        },
        "capital": {
            "initial": INITIAL_CAPITAL,
            "commission": COMMISSION_RATE,
            "slippage": SLIPPAGE,
        },
        "risk": {
            "per_trade": RISK_PER_TRADE,
            "stop_loss": STOP_LOSS_PCT,
            "take_profit": TAKE_PROFIT_PCT,
            "max_drawdown": MAX_DRAWDOWN_LIMIT,
        },
        "ml": {
            "weight": ML_DEFAULT_WEIGHT,
            "min_prob_buy": MIN_PROB_BUY,
            "max_prob_sell": MAX_PROB_SELL,
            "n_estimators": N_ESTIMATORS,
            "random_state": RANDOM_STATE,
        },
    }
