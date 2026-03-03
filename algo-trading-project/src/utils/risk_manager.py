"""
Risk Manager for trading position sizing and risk control.

Provides proper position sizing, drawdown protection, and risk metrics.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

from src.utils.config import (
    INITIAL_CAPITAL,
    RISK_PER_TRADE,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
)
from src.utils.enums import Signal, PositionState


@dataclass
class Position:
    """Represents an open trading position."""
    entry_price: float
    quantity: int
    stop_loss: float
    take_profit: float
    entry_time: Optional[pd.Timestamp] = None
    
    @property
    def risk_amount(self) -> float:
        """Calculate risk amount for the position."""
        return self.quantity * abs(self.entry_price - self.stop_loss)
    
    @property
    def reward_amount(self) -> float:
        """Calculate potential reward."""
        return self.quantity * abs(self.take_profit - self.entry_price)
    
    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio."""
        if self.risk_amount == 0:
            return 0.0
        return self.reward_amount / self.risk_amount
    
    def check_exit(self, price_low: float, price_high: float) -> Optional[float]:
        """Check if position should be exited. Returns exit price or None."""
        if price_low <= self.stop_loss:
            return self.stop_loss
        if price_high >= self.take_profit:
            return self.take_profit
        return None


@dataclass
class RiskMetrics:
    """Container for risk metrics."""
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0
    consecutive_losses: int = 0
    daily_loss: float = 0.0
    position_count: int = 0


class RiskManager:
    """
    Manages position sizing and risk controls.
    
    Features:
    - Kelly Criterion position sizing
    - Maximum drawdown protection
    - Daily loss limits
    - Consecutive loss protection
    - Position concentration limits
    """
    
    def __init__(
        self,
        capital: float = INITIAL_CAPITAL,
        risk_per_trade: float = RISK_PER_TRADE,
        max_drawdown_limit: float = 0.20,
        daily_loss_limit: float = 0.05,
        max_consecutive_losses: int = 5,
        max_positions: int = 3,
    ):
        self.initial_capital = capital
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.max_drawdown_limit = max_drawdown_limit
        self.daily_loss_limit = daily_loss_limit
        self.max_consecutive_losses = max_consecutive_losses
        self.max_positions = max_positions
        
        self._metrics = RiskMetrics(peak_equity=capital)
        self._positions: list[Position] = []
        self._daily_pnl = 0.0
        self._trade_history: list[float] = []
    
    @property
    def metrics(self) -> RiskMetrics:
        """Get current risk metrics."""
        return self._metrics
    
    @property
    def state(self) -> PositionState:
        """Get current position state."""
        if len(self._positions) == 0:
            return PositionState.FLAT
        return PositionState.LONG
    
    @property
    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed based on risk limits."""
        # Check drawdown limit
        if self._metrics.current_drawdown <= -self.max_drawdown_limit:
            return False
        
        # Check daily loss limit
        if self._daily_pnl <= -self.capital * self.daily_loss_limit:
            return False
        
        # Check consecutive losses
        if self._metrics.consecutive_losses >= self.max_consecutive_losses:
            return False
        
        # Check position limit
        if len(self._positions) >= self.max_positions:
            return False
        
        return True
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        use_kelly: bool = False,
    ) -> int:
        """
        Calculate position size based on risk parameters.
        
        Args:
            entry_price: Planned entry price
            stop_loss: Stop loss price
            use_kelly: Whether to use Kelly Criterion
        
        Returns:
            Number of shares/contracts to trade
        """
        if not self.is_trading_allowed:
            return 0
        
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return 0
        
        # Base risk amount
        risk_amount = self.capital * self.risk_per_trade
        
        # Apply Kelly Criterion if enabled and we have history
        if use_kelly and len(self._trade_history) >= 20:
            kelly_fraction = self._calculate_kelly_fraction()
            risk_amount = min(risk_amount, self.capital * kelly_fraction)
        
        # Calculate base position size
        position_size = int(risk_amount / risk_per_share)
        
        # Ensure we can afford it
        max_affordable = int(self.capital * 0.95 / entry_price)  # Use 95% max
        position_size = min(position_size, max_affordable)
        
        return max(0, position_size)
    
    def _calculate_kelly_fraction(self) -> float:
        """Calculate Kelly Criterion fraction from trade history."""
        if len(self._trade_history) < 10:
            return self.risk_per_trade
        
        wins = [t for t in self._trade_history if t > 0]
        losses = [t for t in self._trade_history if t < 0]
        
        if not wins or not losses:
            return self.risk_per_trade
        
        win_rate = len(wins) / len(self._trade_history)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        if avg_loss == 0:
            return self.risk_per_trade
        
        # Kelly formula: f = (p*b - q) / b
        # where p = win rate, q = loss rate, b = win/loss ratio
        b = avg_win / avg_loss
        kelly = (win_rate * b - (1 - win_rate)) / b
        
        # Use half-Kelly for safety
        kelly = kelly * 0.5
        
        # Clamp to reasonable range
        return np.clip(kelly, 0.005, 0.05)
    
    def open_position(
        self,
        entry_price: float,
        quantity: int,
        stop_loss_pct: float = STOP_LOSS_PCT,
        take_profit_pct: float = TAKE_PROFIT_PCT,
        entry_time: Optional[pd.Timestamp] = None,
    ) -> Optional[Position]:
        """Open a new position."""
        if not self.is_trading_allowed or quantity <= 0:
            return None
        
        position = Position(
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=entry_price * (1 - stop_loss_pct),
            take_profit=entry_price * (1 + take_profit_pct),
            entry_time=entry_time,
        )
        
        cost = quantity * entry_price
        if cost > self.capital:
            return None
        
        self.capital -= cost
        self._positions.append(position)
        self._metrics.position_count = len(self._positions)
        
        return position
    
    def close_position(
        self,
        position: Position,
        exit_price: float,
        commission: float = 0.0005,
    ) -> float:
        """Close a position and return PnL."""
        if position not in self._positions:
            return 0.0
        
        gross_pnl = (exit_price - position.entry_price) * position.quantity
        commission_cost = position.quantity * exit_price * commission
        net_pnl = gross_pnl - commission_cost
        
        # Update capital
        proceeds = position.quantity * exit_price - commission_cost
        self.capital += proceeds
        
        # Update metrics
        self._daily_pnl += net_pnl
        self._trade_history.append(net_pnl)
        
        if net_pnl > 0:
            self._metrics.consecutive_losses = 0
        else:
            self._metrics.consecutive_losses += 1
        
        # Update peak and drawdown
        if self.capital > self._metrics.peak_equity:
            self._metrics.peak_equity = self.capital
        
        self._metrics.current_drawdown = (
            (self.capital - self._metrics.peak_equity) / self._metrics.peak_equity
        )
        self._metrics.max_drawdown = min(
            self._metrics.max_drawdown, self._metrics.current_drawdown
        )
        
        # Remove position
        self._positions.remove(position)
        self._metrics.position_count = len(self._positions)
        
        return net_pnl
    
    def reset_daily(self):
        """Reset daily tracking metrics."""
        self._daily_pnl = 0.0
    
    def get_summary(self) -> dict:
        """Get risk management summary."""
        return {
            "capital": self.capital,
            "initial_capital": self.initial_capital,
            "total_return": (self.capital - self.initial_capital) / self.initial_capital,
            "peak_equity": self._metrics.peak_equity,
            "current_drawdown": self._metrics.current_drawdown,
            "max_drawdown": self._metrics.max_drawdown,
            "consecutive_losses": self._metrics.consecutive_losses,
            "position_count": len(self._positions),
            "total_trades": len(self._trade_history),
            "is_trading_allowed": self.is_trading_allowed,
        }
