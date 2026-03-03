"""
Download historical stock data using yfinance.
"""
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import get_logger
from utils.config import DEFAULT_TICKERS, RAW_DATA_DIR

logger = get_logger(__name__)


def download_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Download stock data from Yahoo Finance.
    """
    try:
        logger.info(f"Downloading {ticker} data from {start_date} to {end_date}")
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False
        )
        logger.info(f"Downloaded {len(data)} rows for {ticker}")
        return data
    except Exception as e:
        logger.error(f"Error downloading data for {ticker}: {e}")
        raise


def save_data(data: pd.DataFrame, ticker: str, interval: str = "1d", output_dir = None):
    """
    Save downloaded data to CSV file.
    """
    if output_dir is None:
        output_dir = RAW_DATA_DIR
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    suffix = f"_{interval}" if interval != "1d" else ""
    filepath = Path(output_dir) / f"{ticker}{suffix}.csv"
    data.to_csv(filepath)
    logger.info(f"Saved {len(data)} rows to {filepath}")


def download_all_tickers(
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-12",
    intervals: list = None
):
    """
    Download and save data for all configured tickers and intervals.
    Yahoo Finance limits:
    - 1d: unlimited history
    - 5m: 60 days max
    - 1m: 7 days max
    """
    if intervals is None:
        intervals = ["1d"]
    
    today = datetime.now()
    
    for ticker in DEFAULT_TICKERS:
        for interval in intervals:
            # Adjust date range based on interval
            if interval == "1m":
                # 1m data: last 7 days only
                adj_start = (today - timedelta(days=7)).strftime("%Y-%m-%d")
                adj_end = today.strftime("%Y-%m-%d")
            elif interval == "5m":
                # 5m data: last 30 days only (safer range for Indian stocks)
                adj_start = (today - timedelta(days=30)).strftime("%Y-%m-%d")
                adj_end = today.strftime("%Y-%m-%d")
            else:
                # 1d data: use provided date range
                adj_start = start_date
                adj_end = end_date
            
            logger.info(f"Starting download for {ticker} ({interval}) from {adj_start} to {adj_end}")
            try:
                data = download_stock_data(ticker, adj_start, adj_end, interval)

                if not data.empty:
                    save_data(data, ticker, interval)
                else:
                    logger.warning(f"No data returned for {ticker} ({interval})")
            except Exception as e:
                logger.error(f"Failed to download {ticker} ({interval}): {e}")


if __name__ == "__main__":
    # Download 1d, 5m, and 1m data for all tickers
    # Note: 5m and 1m data has limited history (60 days and 7 days respectively)
    download_all_tickers(intervals=["1d", "5m", "1m"])
