"""
Realistic backtesting engine with proper execution.

Features:
- Proper position sizing based on risk
- Stop-loss and take-profit orders
- Commission handling
- Detailed trade tracking
- Comprehensive performance metrics
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np

from src.utils.config import (
    INITIAL_CAPITAL,
    COMMISSION_RATE,
    RISK_PER_TRADE,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    TRADING_DAYS,
)
from src.utils.enums import Signal, TradeResult


@dataclass
class Trade:
    """Detailed trade record."""
    entry_time: datetime
    entry_price: float
    quantity: int
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    result: TradeResult = TradeResult.WIN
    
    @property
    def return_pct(self) -> float:
        """Return percentage."""
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price


class ProperBacktester:
    """
    Production-grade backtesting engine.
    
    Attributes:
        initial_capital: Starting capital amount
        commission: Commission rate per transaction
        trades: List of completed trades
        equity_curve: Historical equity values
    """

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        commission: float = COMMISSION_RATE,
        risk_per_trade: float = RISK_PER_TRADE,
        stop_loss_pct: float = STOP_LOSS_PCT,
        take_profit_pct: float = TAKE_PROFIT_PCT,
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.reset()

    def reset(self) -> None:
        """Reset backtester state for a new run."""
        self.capital = self.initial_capital
        self.position = 0
        self.entry_price: Optional[float] = None
        self.entry_time: Optional[datetime] = None
        self.stop_loss: Optional[float] = None
        self.take_profit: Optional[float] = None
        self.trades: list[Trade] = []
        self.equity_curve: list[float] = []
        self.dates: list[datetime] = []

    def position_size(self, entry: float, stop: float) -> int:
        """
        Calculate position size based on risk management.
        
        Args:
            entry: Entry price
            stop: Stop-loss price
        
        Returns:
            Number of shares to trade
        """
        risk_amount = self.capital * self.risk_per_trade
        risk_per_share = abs(entry - stop)
        if risk_per_share <= 0:
            return 0
        
        size = int(risk_amount / risk_per_share)
        # Ensure we can afford it
        max_affordable = int(self.capital * 0.95 / entry)
        return min(size, max_affordable)

    def backtest(
        self,
        data: pd.DataFrame,
        signal_col: str = "combined_signal",
    ) -> dict:
        """
        Run backtest on historical data.
        
        Args:
            data: OHLCV DataFrame with signals
            signal_col: Column name containing trading signals
        
        Returns:
            Dictionary of performance metrics
        """
        self.reset()
        data = data.copy()

        # Prevent lookahead bias
        data[signal_col] = data[signal_col].shift(1).fillna(0)

        for i in range(1, len(data)):
            row = data.iloc[i]
            prev = data.iloc[i - 1]
            current_time = data.index[i]

            price_open = row["Open"]
            price_high = row["High"]
            price_low = row["Low"]
            signal = int(prev[signal_col])

            # ---------- ENTRY ----------
            if self.position == 0 and signal == 1:
                entry = price_open
                sl = entry * (1 - self.stop_loss_pct)
                tp = entry * (1 + self.take_profit_pct)

                qty = self.position_size(entry, sl)
                if qty > 0:
                    cost = qty * entry * (1 + self.commission)
                    if cost <= self.capital:
                        self.capital -= cost
                        self.position = qty
                        self.entry_price = entry
                        self.entry_time = current_time
                        self.stop_loss = sl
                        self.take_profit = tp

            # ---------- EXIT ----------
            if self.position > 0:
                exit_price = None
                trade_result = None

                if price_low <= self.stop_loss:
                    exit_price = self.stop_loss
                    trade_result = TradeResult.STOPPED_OUT
                elif price_high >= self.take_profit:
                    exit_price = self.take_profit
                    trade_result = TradeResult.TAKE_PROFIT
                elif signal == -1:
                    exit_price = price_open
                    trade_result = TradeResult.MANUAL_EXIT

                if exit_price:
                    proceeds = self.position * exit_price * (1 - self.commission)
                    pnl = proceeds - (self.position * self.entry_price)
                    self.capital += proceeds

                    # Record detailed trade
                    trade = Trade(
                        entry_time=self.entry_time,
                        entry_price=self.entry_price,
                        quantity=self.position,
                        exit_time=current_time,
                        exit_price=exit_price,
                        pnl=pnl,
                        result=trade_result if pnl > 0 else TradeResult.LOSS,
                    )
                    self.trades.append(trade)

                    self.position = 0
                    self.entry_price = None
                    self.entry_time = None

            # ---------- Equity ----------
            equity = self.capital + self.position * price_open
            self.equity_curve.append(equity)
            self.dates.append(current_time)

        return self.metrics()

    def metrics(self) -> dict:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            Dictionary with all performance metrics
        """
        if not self.equity_curve:
            return self._empty_metrics()
        
        equity = pd.Series(self.equity_curve)
        returns = equity.pct_change().dropna()

        # Drawdown calculations
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_dd = drawdown.min()
        avg_dd = drawdown.mean()

        # Trade statistics
        pnls = [t.pnl for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        win_rate = len(wins) / len(pnls) if pnls else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")
        
        # Risk-adjusted returns
        sharpe = self._calculate_sharpe(returns)
        sortino = self._calculate_sortino(returns)
        
        # Consecutive analysis
        max_consec_wins = self._max_consecutive_wins()
        max_consec_losses = self._max_consecutive_losses()

        return {
            # Basic returns
            "final_equity": equity.iloc[-1],
            "total_return": (equity.iloc[-1] - self.initial_capital) / self.initial_capital,
            "total_pnl": equity.iloc[-1] - self.initial_capital,
            
            # Trade statistics
            "total_trades": len(self.trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": win_rate,
            
            # PnL analysis
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": max(wins) if wins else 0,
            "largest_loss": min(losses) if losses else 0,
            "profit_factor": profit_factor,
            "expectancy": (win_rate * avg_win + (1 - win_rate) * avg_loss) if pnls else 0,
            
            # Risk metrics
            "max_drawdown": max_dd,
            "avg_drawdown": avg_dd,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            
            # Streaks
            "max_consecutive_wins": max_consec_wins,
            "max_consecutive_losses": max_consec_losses,
            
            # Exit analysis
            "stop_losses_hit": sum(1 for t in self.trades if t.result == TradeResult.STOPPED_OUT),
            "take_profits_hit": sum(1 for t in self.trades if t.result == TradeResult.TAKE_PROFIT),
        }
    
    def _empty_metrics(self) -> dict:
        """Return empty metrics when no data."""
        return {
            "final_equity": self.initial_capital,
            "total_return": 0.0,
            "total_pnl": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
            "avg_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "stop_losses_hit": 0,
            "take_profits_hit": 0,
        }
    
    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.05) -> float:
        """Calculate Sharpe ratio."""
        if returns.std() == 0 or len(returns) == 0:
            return 0.0
        excess_returns = returns - (risk_free_rate / TRADING_DAYS)
        return np.sqrt(TRADING_DAYS) * excess_returns.mean() / returns.std()
    
    def _calculate_sortino(self, returns: pd.Series, risk_free_rate: float = 0.05) -> float:
        """Calculate Sortino ratio (downside risk only)."""
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0 or negative_returns.std() == 0:
            return 0.0
        excess_return = returns.mean() - (risk_free_rate / TRADING_DAYS)
        return np.sqrt(TRADING_DAYS) * excess_return / negative_returns.std()
    
    def _max_consecutive_wins(self) -> int:
        """Calculate max consecutive winning trades."""
        max_streak = 0
        current_streak = 0
        for trade in self.trades:
            if trade.pnl > 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak
    
    def _max_consecutive_losses(self) -> int:
        """Calculate max consecutive losing trades."""
        max_streak = 0
        current_streak = 0
        for trade in self.trades:
            if trade.pnl <= 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak
    
    def get_trades_df(self) -> pd.DataFrame:
        """Export trades as DataFrame for analysis."""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "pnl": t.pnl,
                "return_pct": t.return_pct,
                "result": t.result.value,
            }
            for t in self.trades
        ])
    
    def get_equity_df(self) -> pd.DataFrame:
        """Export equity curve as DataFrame."""
        return pd.DataFrame({
            "date": self.dates,
            "equity": self.equity_curve,
        }).set_index("date")
