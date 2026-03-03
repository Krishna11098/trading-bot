"""
Backtesting script for evaluating strategies on 2024 test data.
Train period: 2015-2023
Test period: 2024
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.logger import get_logger
from src.utils.config import (
    DEFAULT_TICKERS,
    INDICATORS_DIR,
    MODELS_DIR,
    TRAIN_START,
    TRAIN_END,
    TEST_START,
    TEST_END,
)
from src.utils.data_split import split_data_by_date
from src.data_collection.load_kaggle_data import load_kaggle_data
from src.preprocessing.clean_data import clean_ohlcv_data
from src.strategy.backtest import ProperBacktester
from src.strategy.combined_strategy import generate_combined_strategy

logger = get_logger(__name__)


def backtest_ticker_on_2024(ticker: str) -> dict:
    """
    Backtest a strategy on 2024 data for a single ticker.
    Model is trained on 2015-2023 data.
    
    Returns:
        Dictionary with backtest metrics
    """
    try:
        logger.info(f"\n📊 Backtesting {ticker} on 2024 data")
        
        # Load and clean data
        raw_data = load_kaggle_data(ticker)
        cleaned_data = clean_ohlcv_data(raw_data)
        
        # Split into train (2015-2023) and test (2024)
        train_data, test_data = split_data_by_date(
            cleaned_data,
            train_start=TRAIN_START,
            train_end=TRAIN_END,
            test_start=TEST_START,
            test_end=TEST_END
        )
        
        if len(test_data) == 0:
            logger.warning(f"No 2024 data available for {ticker}, skipping")
            return {
                "ticker": ticker,
                "status": "no_data",
                "error": "No 2024 test data"
            }
        
        # Add basic technical indicators for strategy
        test_data = add_basic_features(test_data)
        
        # Generate trading signals (combined ML + scalping)
        test_data_signals = generate_combined_strategy(ticker, test_data)
        
        # Run backtest
        backtester = ProperBacktester()
        metrics = backtester.backtest(test_data_signals, signal_col="combined_signal")
        
        metrics["ticker"] = ticker
        metrics["status"] = "success"
        metrics["test_period"] = f"{test_data.index[0].date()} to {test_data.index[-1].date()}"
        metrics["test_samples"] = len(test_data)
        
        logger.info(f"✅ {ticker} Backtest Results:")
        logger.info(f"   Final Equity: ${metrics['final_equity']:.2f}")
        logger.info(f"   Total Return: {metrics['total_return']*100:.2f}%")
        logger.info(f"   Win Rate: {metrics['win_rate']*100:.2f}%")
        logger.info(f"   Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        logger.info(f"   Total Trades: {metrics['total_trades']}")
        logger.info(f"   Sharpe Ratio: {metrics['sharpe']:.4f}")
        
        return metrics
    
    except Exception as e:
        logger.error(f"❌ Backtest failed for {ticker}: {e}")
        return {
            "ticker": ticker,
            "status": "failed",
            "error": str(e)
        }


def backtest_all_tickers_on_2024() -> dict:
    """
    Run backtests for all tickers on 2024 test data.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"BACKTESTING ALL TICKERS ON 2024 DATA")
    logger.info(f"Train Period: {TRAIN_START} to {TRAIN_END}")
    logger.info(f"Test Period: {TEST_START} to {TEST_END}")
    logger.info(f"{'='*60}\n")
    
    results = {}
    
    for ticker in DEFAULT_TICKERS:
        metrics = backtest_ticker_on_2024(ticker)
        results[ticker] = metrics
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("BACKTEST SUMMARY")
    logger.info(f"{'='*60}")
    
    summary_df = pd.DataFrame(results).T
    logger.info(f"\n{summary_df}")
    
    # Calculate aggregate metrics
    successful_backtests = [m for m in results.values() if m.get("status") == "success"]
    
    if successful_backtests:
        avg_return = np.mean([m["total_return"] for m in successful_backtests])
        avg_sharpe = np.mean([m["sharpe"] for m in successful_backtests])
        avg_win_rate = np.mean([m["win_rate"] for m in successful_backtests])
        
        logger.info(f"\nAggregate Metrics (across successful backtests):")
        logger.info(f"   Average Return: {avg_return*100:.2f}%")
        logger.info(f"   Average Sharpe Ratio: {avg_sharpe:.4f}")
        logger.info(f"   Average Win Rate: {avg_win_rate*100:.2f}%")
    
    return results


def add_basic_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic technical indicators for strategy signals.
    """
    df = data.copy()
    
    if "Close" in df.columns:
        # Moving averages for scalping
        if "SMA_20" not in df.columns:
            df["SMA_20"] = df["Close"].rolling(window=20).mean()
        if "SMA_50" not in df.columns:
            df["SMA_50"] = df["Close"].rolling(window=50).mean()
        
        # RSI for scalping
        if "RSI" not in df.columns:
            df["RSI"] = calculate_rsi(df["Close"])
        
        # Volume for confirmation
        if "Volume_SMA_20" not in df.columns and "Volume" in df.columns:
            df["Volume_SMA_20"] = df["Volume"].rolling(window=20).mean()
    
    return df


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index).
    """
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


if __name__ == "__main__":
    results = backtest_all_tickers_on_2024()
