"""
Train/test data splitting by date for time series.
"""
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import get_logger
from utils.config import TRAIN_START, TRAIN_END, TEST_START, TEST_END

logger = get_logger(__name__)


def split_data_by_date(
    data: pd.DataFrame,
    train_start: str = TRAIN_START,
    train_end: str = TRAIN_END,
    test_start: str = TEST_START,
    test_end: str = TEST_END,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train and test sets based on date ranges.
    
    Args:
        data: DataFrame with datetime index
        train_start: Training period start date (YYYY-MM-DD)
        train_end: Training period end date (YYYY-MM-DD)
        test_start: Testing period start date (YYYY-MM-DD)
        test_end: Testing period end date (YYYY-MM-DD)
    
    Returns:
        Tuple of (train_data, test_data)
    """
    df = data.copy()
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Convert string dates to datetime
    train_start_dt = pd.to_datetime(train_start)
    train_end_dt = pd.to_datetime(train_end)
    test_start_dt = pd.to_datetime(test_start)
    test_end_dt = pd.to_datetime(test_end)
    
    # Filter training data
    train_data = df[
        (df.index >= train_start_dt) & 
        (df.index <= train_end_dt)
    ]
    
    # Filter testing data
    test_data = df[
        (df.index >= test_start_dt) & 
        (df.index <= test_end_dt)
    ]
    
    logger.info(
        f"Data split: Train {len(train_data)} rows ({train_start} to {train_end}) | "
        f"Test {len(test_data)} rows ({test_start} to {test_end})"
    )
    
    return train_data, test_data


def get_date_range(data: pd.DataFrame) -> tuple[datetime, datetime]:
    """
    Get the date range of a DataFrame.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    return data.index[0], data.index[-1]
