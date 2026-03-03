"""
Feature engineering: technical indicators and derived features.
"""
import pandas as pd
import numpy as np
import pandas_ta as ta
from pathlib import Path
import sys

# Make project root importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.logger import get_logger
from src.utils.config import (
    DEFAULT_TICKERS,
    PROCESSED_DATA_DIR,
    INDICATORS_DIR,
)

logger = get_logger(__name__)


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to OHLCV data.
    """
    df = data.copy()

    # Moving averages
    df["SMA_20"] = ta.sma(df["Close"], length=20)
    df["SMA_50"] = ta.sma(df["Close"], length=50)
    df["EMA_12"] = ta.ema(df["Close"], length=12)
    df["EMA_26"] = ta.ema(df["Close"], length=26)

    # RSI
    df["RSI"] = ta.rsi(df["Close"], length=14)

    # MACD
    macd = ta.macd(df["Close"])
    df = pd.concat([df, macd], axis=1)

    # Bollinger Bands
    bb = ta.bbands(df["Close"], length=20)
    df = pd.concat([df, bb], axis=1)

    # ATR
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    # Volume indicators
    df["Volume_SMA_20"] = ta.sma(df["Volume"], length=20)

    # Stochastic Oscillator
    stoch = ta.stoch(df["High"], df["Low"], df["Close"])
    df = pd.concat([df, stoch], axis=1)

    logger.info(f"Added technical indicators → {len(df.columns)} columns")
    return df


def add_price_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add price-based statistical features.
    """
    df = data.copy()

    # Returns
    df["returns"] = df["Close"].pct_change()
    df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))

    # Price range
    df["range"] = df["High"] - df["Low"]
    df["range_pct"] = df["range"] / df["Close"]

    # Gap features
    df["gap"] = df["Open"] - df["Close"].shift(1)
    df["gap_pct"] = df["gap"] / df["Close"].shift(1)

    # Rolling volatility
    df["volatility_20"] = df["returns"].rolling(window=20).std()

    logger.info("Added price-derived features")
    return df


def create_lag_features(
    data: pd.DataFrame,
    columns: list[str],
    lags: int = 3,
) -> pd.DataFrame:
    """
    Create lagged features for selected columns.
    """
    df = data.copy()

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found — skipping lags")
            continue

        for lag in range(1, lags + 1):
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    logger.info(f"Created lag features → {columns} | lags={lags}")
    return df


def generate_features_for_all_tickers():
    """
    Generate features for all configured tickers.
    """
    INDICATORS_DIR.mkdir(parents=True, exist_ok=True)

    for ticker in DEFAULT_TICKERS:
        logger.info(f"⚙️ Feature engineering for {ticker}")

        # Process multiple intervals: 1d (no suffix), 5m (_5m), 1m (_1m)
        intervals = [("1d", ""), ("5m", "_5m"), ("1m", "_1m")]

        for interval_name, suffix in intervals:
            input_path = PROCESSED_DATA_DIR / f"{ticker}{suffix}_cleaned.csv"
            if not input_path.exists():
                logger.warning(f"Missing cleaned data for {ticker}{suffix}, skipping {interval_name}")
                continue

            df = pd.read_csv(
                input_path,
                index_col=0,
                parse_dates=True,
            )

            df = add_technical_indicators(df)
            df = add_price_features(df)
            df = create_lag_features(
                df,
                columns=["Close", "Volume", "RSI"],
                lags=3,
            )

            output_path = INDICATORS_DIR / f"{ticker}{suffix}_features.csv"
            df.to_csv(output_path)

            logger.info(f"Saved features → {output_path} ({interval_name})")


if __name__ == "__main__":
    generate_features_for_all_tickers()
