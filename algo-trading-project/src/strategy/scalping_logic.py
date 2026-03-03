"""
Improved scalping signal generation using vote-based system.

Combines multiple technical indicators to generate high-confidence
scalping signals for short-term trading.
"""
from typing import Tuple
import pandas as pd
import numpy as np

from src.utils.config import RSI_OVERSOLD, RSI_OVERBOUGHT
from src.utils.enums import Signal


def calculate_rsi_signal(df: pd.DataFrame) -> pd.Series:
    """
    Calculate RSI-based signal component.
    
    Args:
        df: DataFrame with RSI column
    
    Returns:
        Series with RSI signal scores (-1, 0, 1)
    """
    signal = pd.Series(0, index=df.index)
    
    if "RSI" not in df.columns:
        return signal
    
    signal.loc[df["RSI"] <= RSI_OVERSOLD] = 1   # Oversold = bullish
    signal.loc[df["RSI"] >= RSI_OVERBOUGHT] = -1  # Overbought = bearish
    
    return signal


def calculate_bollinger_signal(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Bollinger Bands signal component.
    
    Args:
        df: DataFrame with Bollinger Band columns
    
    Returns:
        Series with BB signal scores (-1, 0, 1)
    """
    signal = pd.Series(0, index=df.index)
    
    required_cols = {"BBL_20_2.0", "BBU_20_2.0"}
    if not required_cols.issubset(df.columns):
        return signal
    
    signal.loc[df["Close"] <= df["BBL_20_2.0"]] = 1   # Below lower band = bullish
    signal.loc[df["Close"] >= df["BBU_20_2.0"]] = -1  # Above upper band = bearish
    
    return signal


def calculate_macd_signal(df: pd.DataFrame) -> pd.Series:
    """
    Calculate MACD crossover signal component.
    
    Args:
        df: DataFrame with MACD columns
    
    Returns:
        Series with MACD signal scores (-1, 0, 1)
    """
    signal = pd.Series(0, index=df.index)
    
    required_cols = {"MACD_12_26_9", "MACDs_12_26_9"}
    if not required_cols.issubset(df.columns):
        return signal
    
    macd = df["MACD_12_26_9"]
    macd_signal = df["MACDs_12_26_9"]
    
    # Bullish crossover: MACD crosses above signal line
    bullish_cross = (macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1))
    
    # Bearish crossover: MACD crosses below signal line
    bearish_cross = (macd < macd_signal) & (macd.shift(1) >= macd_signal.shift(1))
    
    signal.loc[bullish_cross] = 1
    signal.loc[bearish_cross] = -1
    
    return signal


def calculate_stochastic_signal(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Stochastic oscillator signal component.
    
    Args:
        df: DataFrame with Stochastic columns
    
    Returns:
        Series with Stochastic signal scores (-1, 0, 1)
    """
    signal = pd.Series(0, index=df.index)
    
    # Check for stochastic columns (pandas_ta names)
    stoch_k = None
    stoch_d = None
    
    for col in df.columns:
        if col.startswith("STOCHk_"):
            stoch_k = col
        if col.startswith("STOCHd_"):
            stoch_d = col
    
    if stoch_k is None or stoch_d is None:
        return signal
    
    # Oversold crossover
    signal.loc[(df[stoch_k] < 20) & (df[stoch_k] > df[stoch_d])] = 1
    # Overbought crossover
    signal.loc[(df[stoch_k] > 80) & (df[stoch_k] < df[stoch_d])] = -1
    
    return signal


def calculate_scalping_signals(
    df: pd.DataFrame,
    min_score: int = 2,
) -> pd.DataFrame:
    """
    Generate scalping signals using vote-based indicator system.
    
    Combines RSI, Bollinger Bands, MACD, and Stochastic signals
    using a voting mechanism. A trade signal is only generated
    when multiple indicators agree.
    
    Args:
        df: DataFrame with OHLCV and indicator data
        min_score: Minimum score needed to generate signal (default: 2)
    
    Returns:
        DataFrame with scalp_score and scalp_signal columns
    """
    df = df.copy()
    
    # Calculate individual indicator signals
    rsi_signal = calculate_rsi_signal(df)
    bb_signal = calculate_bollinger_signal(df)
    macd_signal = calculate_macd_signal(df)
    stoch_signal = calculate_stochastic_signal(df)
    
    # Combine signals using voting
    df["scalp_score"] = rsi_signal + bb_signal + macd_signal + stoch_signal
    
    # Generate final signal based on minimum score threshold
    df["scalp_signal"] = np.where(
        df["scalp_score"] >= min_score, Signal.BUY.value,
        np.where(df["scalp_score"] <= -min_score, Signal.SELL.value, Signal.HOLD.value)
    )
    
    # Add signal strength for position sizing
    max_possible_score = 4  # 4 indicators
    df["scalp_strength"] = (abs(df["scalp_score"]) / max_possible_score).clip(0, 1)
    
    return df


def get_signal_breakdown(df: pd.DataFrame, idx: int) -> dict:
    """
    Get detailed breakdown of signals at a specific index.
    
    Useful for debugging and understanding why a signal was generated.
    
    Args:
        df: DataFrame with indicator data
        idx: Index position to analyze
    
    Returns:
        Dictionary with signal breakdown
    """
    row = df.iloc[[idx]]
    
    return {
        "rsi": calculate_rsi_signal(row).iloc[0],
        "bollinger": calculate_bollinger_signal(row).iloc[0],
        "macd": calculate_macd_signal(row).iloc[0],
        "stochastic": calculate_stochastic_signal(row).iloc[0],
        "total_score": row["scalp_score"].iloc[0] if "scalp_score" in row.columns else None,
        "final_signal": row["scalp_signal"].iloc[0] if "scalp_signal" in row.columns else None,
    }
