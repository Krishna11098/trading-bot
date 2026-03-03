"""
Utility package exports.

Core utilities for the trading system including:
- Logging
- Configuration management
- Risk management
- Performance reporting
- Data validation
- ML model helpers
"""
from .logger import get_logger, setup_file_logger
from .config import (
    # Paths
    BASE_DIR,
    DATA_DIR,
    MODELS_DIR,
    # Config
    INITIAL_CAPITAL,
    COMMISSION_RATE,
    RISK_PER_TRADE,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    DEFAULT_TICKERS,
    TradingConfig,
    get_config,
)
from .enums import (
    Signal,
    OrderType,
    OrderSide,
    PositionState,
    TimeFrame,
    TrendDirection,
    MarketRegime,
    TradeResult,
)
from .risk_manager import RiskManager, Position, RiskMetrics
from .performance import PerformanceReporter, PerformanceMetrics, TradeRecord
from .validators import DataValidator, ValidationError, validate_signal_column
from .helpers import (
    get_trading_dates,
    calculate_returns,
    calculate_log_returns,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
)

# Export ML helpers (defined in models.py)
from .models import load_model, predict

__all__ = [
    # Logging
    "get_logger",
    "setup_file_logger",
    # Config
    "TradingConfig",
    "get_config",
    "INITIAL_CAPITAL",
    "DEFAULT_TICKERS",
    # Enums
    "Signal",
    "OrderType",
    "OrderSide",
    "PositionState",
    "TimeFrame",
    "TradeResult",
    # Risk Management
    "RiskManager",
    "Position",
    "RiskMetrics",
    # Performance
    "PerformanceReporter",
    "PerformanceMetrics",
    # Validation
    "DataValidator",
    "ValidationError",
    # Helpers
    "calculate_returns",
    "calculate_sharpe_ratio",
    "calculate_max_drawdown",
    # ML
    "load_model",
    "predict",
]
