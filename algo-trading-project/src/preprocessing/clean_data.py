"""
Clean and preprocess OHLCV stock data from Kaggle.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Make project root importable when running this script directly
# (file is at src/preprocessing/clean_data.py -> parents[2] is project root)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.logger import get_logger
from src.utils.config import (
    DEFAULT_TICKERS,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)
from src.data_collection.load_kaggle_data import load_kaggle_data

logger = get_logger(__name__)


def clean_ohlcv_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean OHLCV data by handling duplicates, missing values, and anomalies.
    """
    df = data.copy()

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Remove duplicate timestamps
    df = df[~df.index.duplicated(keep="first")]

    # Force numeric columns
    price_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in price_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Handle missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        logger.warning(f"Found {missing} missing values — applying ffill/bfill")
        df = df.ffill().bfill()

    # Remove non-trading rows (Volume > 0)
    if "Volume" in df.columns:
        # Some Kaggle minute datasets use 0 for volume; only filter by volume
        # when it's informative. If most rows have zero volume, skip this
        # filter to avoid dropping all data.
        initial_rows = len(df)
        zero_frac = (df["Volume"] == 0).mean()
        if zero_frac < 0.9:
            df = df[df["Volume"] > 0]
            removed = initial_rows - len(df)
            if removed > 0:
                logger.info(f"Removed {removed} zero-volume rows")
        else:
            logger.info("Volume column largely zero — skipping volume filter")

    # Remove extreme price anomalies (robust quantile filter)
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            # Skip quantile-based filtering if dataframe became empty
            if df.empty:
                break

            q_low, q_high = df[col].quantile([0.01, 0.99])
            initial_rows = len(df)
            df = df[(df[col] >= q_low) & (df[col] <= q_high)]
            removed = initial_rows - len(df)
            if removed > 0:
                logger.info(f"Removed {removed} outliers from {col}")

    if df.empty:
        logger.warning("Dataframe empty after cleaning steps — returning empty DataFrame")
        return df

    logger.info(f"Cleaned OHLCV data → {len(df)} rows | {df.index[0]} to {df.index[-1]}")
    return df


def resample_data(data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a higher timeframe.
    """
    resampled = (
        data.resample(timeframe)
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
    )

    logger.info(f"Resampled to {timeframe} → {len(resampled)} rows")
    return resampled


def clean_all_tickers(raw_dir: Path = RAW_DATA_DIR, processed_dir: Path = PROCESSED_DATA_DIR):
    """
    Clean raw OHLCV data for all configured tickers from Kaggle CSV files.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    for ticker in DEFAULT_TICKERS:
        logger.info(f"Cleaning data for {ticker}")
        
        try:
            # Load raw Kaggle data
            data = load_kaggle_data(ticker, raw_dir)
            
            # Clean the data
            cleaned = clean_ohlcv_data(data)
            
            # Save cleaned data (skip if empty)
            if cleaned.empty:
                logger.warning(f"Cleaned data for {ticker} is empty — skipping save")
            else:
                output_path = processed_dir / f"{ticker}_cleaned.csv"
                cleaned.to_csv(output_path)
                logger.info(f"Saved cleaned data → {output_path}")
        
        except Exception as e:
            logger.error(f"Failed to clean {ticker}: {e}")
            continue


if __name__ == "__main__":
    clean_all_tickers()
