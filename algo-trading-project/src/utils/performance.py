"""
Performance Reporter for comprehensive trading metrics and analysis.

Provides detailed performance statistics, trade analysis, and reporting.
"""
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import numpy as np
import pandas as pd

from src.utils.config import TRADING_DAYS, INITIAL_CAPITAL


@dataclass
class TradeRecord:
    """Individual trade record."""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_percent: float
    direction: str = "long"
    
    @property
    def duration(self) -> pd.Timedelta:
        """Trade duration."""
        return pd.Timestamp(self.exit_time) - pd.Timestamp(self.entry_time)
    
    @property
    def is_win(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl > 0


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics container."""
    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    
    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # PnL statistics
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    
    # Additional metrics
    avg_trade_duration: str = ""
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    recovery_factor: float = 0.0


class PerformanceReporter:
    """
    Generates comprehensive performance reports from trading results.
    
    Features:
    - Complete performance metrics calculation
    - Trade-level analysis
    - Equity curve analysis
    - Risk-adjusted return metrics
    - Monthly/yearly breakdown
    """
    
    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        risk_free_rate: float = 0.05,  # 5% annual
        trading_days: int = TRADING_DAYS,
    ):
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        
        self._trades: list[TradeRecord] = []
        self._equity_curve: list[float] = []
        self._dates: list[datetime] = []
    
    def add_trade(
        self,
        entry_time: datetime,
        exit_time: datetime,
        entry_price: float,
        exit_price: float,
        quantity: int,
        direction: str = "long",
    ):
        """Add a completed trade to the record."""
        if direction == "long":
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity
        
        pnl_percent = (exit_price - entry_price) / entry_price
        
        trade = TradeRecord(
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=pnl,
            pnl_percent=pnl_percent,
            direction=direction,
        )
        self._trades.append(trade)
    
    def set_equity_curve(
        self,
        equity: list[float],
        dates: Optional[list[datetime]] = None,
    ):
        """Set the equity curve data."""
        self._equity_curve = equity
        self._dates = dates or list(range(len(equity)))
    
    def calculate_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        metrics = PerformanceMetrics()
        
        if not self._equity_curve:
            return metrics
        
        equity = pd.Series(self._equity_curve)
        returns = equity.pct_change().dropna()
        
        # Basic returns
        metrics.total_return = (equity.iloc[-1] - self.initial_capital) / self.initial_capital
        
        # Annualized return
        n_periods = len(returns)
        if n_periods > 0:
            total_factor = equity.iloc[-1] / self.initial_capital
            years = n_periods / self.trading_days
            metrics.annualized_return = (total_factor ** (1 / max(years, 0.01))) - 1 if years > 0 else 0
        
        # Risk metrics
        metrics.sharpe_ratio = self._calculate_sharpe(returns)
        metrics.sortino_ratio = self._calculate_sortino(returns)
        metrics.max_drawdown = self._calculate_max_drawdown(equity)
        metrics.avg_drawdown = self._calculate_avg_drawdown(equity)
        
        # Calmar ratio
        if metrics.max_drawdown != 0:
            metrics.calmar_ratio = metrics.annualized_return / abs(metrics.max_drawdown)
        
        # Trade statistics
        if self._trades:
            metrics.total_trades = len(self._trades)
            metrics.winning_trades = sum(1 for t in self._trades if t.is_win)
            metrics.losing_trades = metrics.total_trades - metrics.winning_trades
            metrics.win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0
            
            # PnL statistics
            wins = [t.pnl for t in self._trades if t.pnl > 0]
            losses = [t.pnl for t in self._trades if t.pnl < 0]
            
            metrics.avg_win = np.mean(wins) if wins else 0
            metrics.avg_loss = np.mean(losses) if losses else 0
            
            total_wins = sum(wins) if wins else 0
            total_losses = abs(sum(losses)) if losses else 0
            metrics.profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")
            
            # Expectancy
            metrics.expectancy = (
                metrics.win_rate * metrics.avg_win + 
                (1 - metrics.win_rate) * metrics.avg_loss
            )
            
            # Trade duration
            durations = [t.duration for t in self._trades]
            avg_duration = pd.Timedelta(np.mean([d.total_seconds() for d in durations]), unit='s')
            metrics.avg_trade_duration = str(avg_duration)
            
            # Consecutive wins/losses
            metrics.max_consecutive_wins = self._max_consecutive(True)
            metrics.max_consecutive_losses = self._max_consecutive(False)
        
        # Recovery factor
        if metrics.max_drawdown != 0:
            total_profit = equity.iloc[-1] - self.initial_capital
            metrics.recovery_factor = total_profit / (abs(metrics.max_drawdown) * self.initial_capital)
        
        return metrics
    
    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if returns.std() == 0 or len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (self.risk_free_rate / self.trading_days)
        return np.sqrt(self.trading_days) * excess_returns.mean() / returns.std()
    
    def _calculate_sortino(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (penalizes only downside volatility)."""
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0 or negative_returns.std() == 0:
            return 0.0
        
        excess_return = returns.mean() - (self.risk_free_rate / self.trading_days)
        downside_std = negative_returns.std()
        
        return np.sqrt(self.trading_days) * excess_return / downside_std
    
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        return drawdown.min()
    
    def _calculate_avg_drawdown(self, equity: pd.Series) -> float:
        """Calculate average drawdown."""
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        return drawdown.mean()
    
    def _max_consecutive(self, wins: bool) -> int:
        """Calculate max consecutive wins or losses."""
        if not self._trades:
            return 0
        
        max_streak = 0
        current_streak = 0
        
        for trade in self._trades:
            if trade.is_win == wins:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def get_monthly_returns(self) -> pd.DataFrame:
        """Get monthly returns breakdown."""
        if not self._equity_curve or not self._dates:
            return pd.DataFrame()
        
        equity_df = pd.DataFrame({
            "date": pd.to_datetime(self._dates),
            "equity": self._equity_curve,
        }).set_index("date")
        
        equity_df["returns"] = equity_df["equity"].pct_change()
        
        monthly = equity_df.resample("M")["returns"].apply(
            lambda x: (1 + x).prod() - 1
        )
        
        return monthly.to_frame("return")
    
    def generate_report(self) -> str:
        """Generate a text performance report."""
        metrics = self.calculate_metrics()
        
        report = []
        report.append("=" * 60)
        report.append("TRADING PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append("")
        
        report.append("📊 RETURNS")
        report.append("-" * 40)
        report.append(f"  Total Return:      {metrics.total_return:>10.2%}")
        report.append(f"  Annualized Return: {metrics.annualized_return:>10.2%}")
        report.append("")
        
        report.append("⚠️  RISK METRICS")
        report.append("-" * 40)
        report.append(f"  Sharpe Ratio:      {metrics.sharpe_ratio:>10.2f}")
        report.append(f"  Sortino Ratio:     {metrics.sortino_ratio:>10.2f}")
        report.append(f"  Calmar Ratio:      {metrics.calmar_ratio:>10.2f}")
        report.append(f"  Max Drawdown:      {metrics.max_drawdown:>10.2%}")
        report.append(f"  Avg Drawdown:      {metrics.avg_drawdown:>10.2%}")
        report.append("")
        
        report.append("📈 TRADE STATISTICS")
        report.append("-" * 40)
        report.append(f"  Total Trades:      {metrics.total_trades:>10}")
        report.append(f"  Winning Trades:    {metrics.winning_trades:>10}")
        report.append(f"  Losing Trades:     {metrics.losing_trades:>10}")
        report.append(f"  Win Rate:          {metrics.win_rate:>10.2%}")
        report.append("")
        
        report.append("💰 PROFIT/LOSS")
        report.append("-" * 40)
        report.append(f"  Avg Win:           {metrics.avg_win:>10.2f}")
        report.append(f"  Avg Loss:          {metrics.avg_loss:>10.2f}")
        report.append(f"  Profit Factor:     {metrics.profit_factor:>10.2f}")
        report.append(f"  Expectancy:        {metrics.expectancy:>10.2f}")
        report.append("")
        
        report.append("🔄 STREAKS")
        report.append("-" * 40)
        report.append(f"  Max Cons. Wins:    {metrics.max_consecutive_wins:>10}")
        report.append(f"  Max Cons. Losses:  {metrics.max_consecutive_losses:>10}")
        report.append(f"  Recovery Factor:   {metrics.recovery_factor:>10.2f}")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def to_dict(self) -> dict:
        """Export metrics as dictionary."""
        metrics = self.calculate_metrics()
        return {
            "total_return": metrics.total_return,
            "annualized_return": metrics.annualized_return,
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "calmar_ratio": metrics.calmar_ratio,
            "max_drawdown": metrics.max_drawdown,
            "avg_drawdown": metrics.avg_drawdown,
            "total_trades": metrics.total_trades,
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor,
            "expectancy": metrics.expectancy,
            "max_consecutive_wins": metrics.max_consecutive_wins,
            "max_consecutive_losses": metrics.max_consecutive_losses,
            "recovery_factor": metrics.recovery_factor,
        }
