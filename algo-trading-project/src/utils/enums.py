"""
Trading system enumerations and constants.

Provides type-safe enums for signals, order types, and trading states.
"""
from enum import Enum, IntEnum


class Signal(IntEnum):
    """Trading signal types."""
    SELL = -1
    HOLD = 0
    BUY = 1
    
    @classmethod
    def from_score(cls, score: float, threshold: float = 0.5) -> "Signal":
        """Convert numerical score to signal."""
        if score >= threshold:
            return cls.BUY
        elif score <= -threshold:
            return cls.SELL
        return cls.HOLD
    
    @property
    def is_entry(self) -> bool:
        """Check if signal triggers an entry."""
        return self == Signal.BUY
    
    @property
    def is_exit(self) -> bool:
        """Check if signal triggers an exit."""
        return self == Signal.SELL


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class PositionState(Enum):
    """Position state enumeration."""
    FLAT = "flat"
    LONG = "long"
    SHORT = "short"


class TimeFrame(Enum):
    """Trading timeframe enumeration."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    
    @property
    def minutes(self) -> int:
        """Get timeframe duration in minutes."""
        mapping = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "4h": 240, "1d": 1440, "1w": 10080
        }
        return mapping[self.value]


class TrendDirection(Enum):
    """Market trend direction."""
    UP = "uptrend"
    DOWN = "downtrend"
    SIDEWAYS = "sideways"


class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    QUIET = "quiet"


class TradeResult(Enum):
    """Trade outcome enumeration."""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    STOPPED_OUT = "stopped_out"
    TAKE_PROFIT = "take_profit"
    MANUAL_EXIT = "manual_exit"
