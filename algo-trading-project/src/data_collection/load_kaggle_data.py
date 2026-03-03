"""
Load historical stock data from Kaggle CSV files.
"""
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import get_logger
from utils.config import DEFAULT_TICKERS, RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = get_logger(__name__)


def load_kaggle_data(
    ticker: str,
    raw_dir: Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """
    Load stock data from Kaggle CSV file.
    
    Expected CSV format: date, open, high, low, close, volume
    """
    try:
        # Handle ticker name with spaces (e.g., "NIFTY BANK" -> "NIFTY BANK_minute.csv")
        file_path = raw_dir / f"{ticker}_minute.csv"
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"No data file for {ticker}")
        
        logger.info(f"Loading {ticker} from {file_path}")
        
        # Load CSV with proper date parsing
        df = pd.read_csv(
            file_path,
            parse_dates=['date'],
            index_col='date'
        )
        
        # Standardize column names to uppercase
        df.columns = [col.strip().capitalize() for col in df.columns]
        
        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                raise ValueError(f"Missing column {col} in {ticker} data")
        
        # Convert to numeric and handle any errors
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"Loaded {len(df)} rows for {ticker} from {df.index[0]} to {df.index[-1]}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data for {ticker}: {e}")
        raise


def load_all_tickers(raw_dir: Path = RAW_DATA_DIR) -> dict:
    """
    Load data for all configured tickers.
    
    Returns:
        Dictionary with ticker as key and DataFrame as value
    """
    data = {}
    
    for ticker in DEFAULT_TICKERS:
        try:
            df = load_kaggle_data(ticker, raw_dir)
            data[ticker] = df
        except Exception as e:
            logger.warning(f"Failed to load {ticker}: {e}")
    
    if not data:
        logger.error("No data loaded for any ticker")
        raise RuntimeError("Failed to load data for all tickers")
    
    logger.info(f"Successfully loaded {len(data)} tickers")
    return data


if __name__ == "__main__":
    # Test loading all tickers
    all_data = load_all_tickers()
    for ticker, df in all_data.items():
        print(f"\n{ticker}: {len(df)} rows | {df.index[0]} to {df.index[-1]}")
        print(df.head())
