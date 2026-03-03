"""
Realistic backtesting engine with proper execution.
"""

import pandas as pd
import numpy as np
from src.utils.config import (
    INITIAL_CAPITAL,
    COMMISSION_RATE,
    RISK_PER_TRADE,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
)

class ProperBacktester:

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        commission: float = COMMISSION_RATE,
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.reset()

    def reset(self):
        self.capital = self.initial_capital
        self.position = 0
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.trades = []
        self.equity_curve = []

    # ---------- Position sizing ----------
    def position_size(self, entry, stop):
        risk_amount = self.capital * RISK_PER_TRADE
        risk_per_share = abs(entry - stop)
        if risk_per_share <= 0:
            return 0
        return int(risk_amount / risk_per_share)

    # ---------- Backtest ----------
    def backtest(
        self,
        data: pd.DataFrame,
        signal_col: str = "combined_signal",
    ) -> dict:

        self.reset()
        data = data.copy()

        # Prevent lookahead bias
        data[signal_col] = data[signal_col].shift(1).fillna(0)

        for i in range(1, len(data)):
            row = data.iloc[i]
            prev = data.iloc[i - 1]

            price_open = row["Open"]
            price_high = row["High"]
            price_low = row["Low"]
            signal = int(prev[signal_col])

            # ---------- ENTRY ----------
            if self.position == 0 and signal == 1:
                entry = price_open
                sl = entry * (1 - STOP_LOSS_PCT)
                tp = entry * (1 + TAKE_PROFIT_PCT)

                qty = self.position_size(entry, sl)
                if qty > 0:
                    cost = qty * entry * (1 + self.commission)
                    if cost <= self.capital:
                        self.capital -= cost
                        self.position = qty
                        self.entry_price = entry
                        self.stop_loss = sl
                        self.take_profit = tp

            # ---------- EXIT ----------
            if self.position > 0:
                exit_price = None

                if price_low <= self.stop_loss:
                    exit_price = self.stop_loss
                elif price_high >= self.take_profit:
                    exit_price = self.take_profit
                elif signal == -1:
                    exit_price = price_open

                if exit_price:
                    proceeds = self.position * exit_price * (1 - self.commission)
                    pnl = proceeds - (self.position * self.entry_price)
                    self.capital += proceeds

                    self.trades.append(pnl)

                    self.position = 0
                    self.entry_price = None

            # ---------- Equity ----------
            equity = self.capital + self.position * price_open
            self.equity_curve.append(equity)

        return self.metrics()

    # ---------- Metrics ----------
    def metrics(self):
        equity = pd.Series(self.equity_curve)
        returns = equity.pct_change().dropna()

        max_dd = ((equity - equity.cummax()) / equity.cummax()).min()

        return {
            "final_equity": equity.iloc[-1],
            "total_return": (equity.iloc[-1] - self.initial_capital) / self.initial_capital,
            "win_rate": sum(1 for p in self.trades if p > 0) / len(self.trades) if self.trades else 0,
            "max_drawdown": max_dd,
            "total_trades": len(self.trades),
            "sharpe": (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
        }
