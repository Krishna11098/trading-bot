"""
Helper utility functions.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def get_trading_dates(start_date: str, end_date: str) -> list:
    """
    Get list of trading dates between start and end.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        List of trading dates
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Generate business days (excludes weekends)
    dates = pd.bdate_range(start=start, end=end)
    return dates.tolist()


def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate simple returns from price series.
    
    Args:
        prices: Price series
    
    Returns:
        Returns series
    """
    return prices.pct_change()


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate log returns from price series.
    
    Args:
        prices: Price series
    
    Returns:
        Log returns series
    """
    return np.log(prices / prices.shift(1))


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Returns series
        risk_free_rate: Annual risk-free rate
    
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - (risk_free_rate / 252)
    if returns.std() == 0:
        return 0
    return np.sqrt(252) * excess_returns.mean() / returns.std()


def calculate_max_drawdown(prices: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        prices: Price series
    
    Returns:
        Maximum drawdown (negative value)
    """
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    return drawdown.min()


def normalize_data(data: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Normalize data using min-max scaling.
    
    Args:
        data: DataFrame to normalize
        columns: Columns to normalize (if None, all numeric columns)
    
    Returns:
        Normalized DataFrame
    """
    df = data.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        
        if max_val - min_val != 0:
            df[col] = (df[col] - min_val) / (max_val - min_val)
    
    return df


def save_json(data: dict, filepath: str):
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to JSON file
    """
    import json
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4, default=str)


def load_json(filepath: str) -> dict:
    """
    Load JSON file to dictionary.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        Dictionary
    """
    import json
    
    with open(filepath, 'r') as f:
        return json.load(f)


def format_currency(value: float) -> str:
    """
    Format value as currency.
    
    Args:
        value: Numeric value
    
    Returns:
        Formatted currency string
    """
    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """
    Format value as percentage.
    
    Args:
        value: Numeric value (e.g., 0.15 for 15%)
    
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.2f}%"

