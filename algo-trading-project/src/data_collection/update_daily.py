"""
Update stock data with latest prices (daily, 1m, 5m).
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


# Yahoo Finance interval limits
INTERVAL_LIMITS = {
    "1m": 7,     # days
    "5m": 60,    # days
    "1d": 3650   # ~10 years (safe)
}


def update_ticker_data(
    ticker: str,
    interval: str = "1d",
    data_dir: str = None
) -> pd.DataFrame | None:
    """
    Update existing data file with latest prices.
    Supports 1m, 5m, and 1d intervals.
    """
    if data_dir is None:
        data_dir = str(RAW_DATA_DIR)
    
    suffix = f"_{interval}"
    filepath = Path(data_dir) / f"{ticker}{suffix}.csv"

    today = datetime.now()

    # If file does not exist → fresh download (within allowed window)
    if not filepath.exists():
        logger.info(f"No existing {interval} data for {ticker}. Creating fresh file.")
        start_date = today - timedelta(days=INTERVAL_LIMITS[interval])
        data = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            interval=interval,
            progress=False
        )
        if not data.empty:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(filepath)
            logger.info(f"Saved {len(data)} rows for {ticker} ({interval})")
        return data

    # Load existing data
    existing_data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    last_date = pd.to_datetime(existing_data.index[-1])

    # Compute allowed start date
    max_lookback = today - timedelta(days=INTERVAL_LIMITS[interval])
    start_date = max(last_date + timedelta(minutes=1), max_lookback)

    if start_date >= today:
        logger.info(f"{ticker} ({interval}) is already up to date")
        return existing_data

    logger.info(
        f"Updating {ticker} ({interval}) from {start_date} to {today}"
    )

    new_data = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        interval=interval,
        progress=False
    )

    if not new_data.empty:
        updated_data = pd.concat([existing_data, new_data])
        updated_data = updated_data[~updated_data.index.duplicated(keep="last")]
        filepath.parent.mkdir(parents=True, exist_ok=True)
        updated_data.to_csv(filepath)
        logger.info(f"Added {len(new_data)} new rows for {ticker} ({interval})")
        return updated_data

    logger.info(f"No new data for {ticker} ({interval})")
    return existing_data


if __name__ == "__main__":
    # Update all intervals for all tickers
    intervals = ["1d", "5m", "1m"]

    for ticker in DEFAULT_TICKERS:
        for interval in intervals:
            try:
                update_ticker_data(ticker, interval)
            except Exception as e:
                logger.error(f"Failed to update {ticker} ({interval}): {e}")
