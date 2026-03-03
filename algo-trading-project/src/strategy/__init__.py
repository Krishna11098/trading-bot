"""
Strategy package for trading signal generation and backtesting.

Provides:
- Combined ML + Technical strategy
- Scalping signal generation
- Backtesting engine with performance metrics
"""
from .backtest import ProperBacktester, Trade
from .combined_strategy import (
    CombinedStrategy,
    generate_combined_strategy,
    combine_signals,
)
from .scalping_logic import (
    calculate_scalping_signals,
    calculate_rsi_signal,
    calculate_bollinger_signal,
    calculate_macd_signal,
)

__all__ = [
    # Backtesting
    "ProperBacktester",
    "Trade",
    # Combined Strategy
    "CombinedStrategy",
    "generate_combined_strategy",
    "combine_signals",
    # Scalping
    "calculate_scalping_signals",
    "calculate_rsi_signal",
    "calculate_bollinger_signal",
    "calculate_macd_signal",
]