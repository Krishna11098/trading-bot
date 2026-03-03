"""
Combined Strategy: ML Predictions + Technical Scalping Signals.

This module implements an ensemble approach that combines:
1. Machine Learning predictions (probability-based)
2. Technical indicator scalping signals (vote-based)

The weighting between ML and scalping adapts based on ML confidence.
"""
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from src.strategy.scalping_logic import calculate_scalping_signals
from src.utils.config import (
    ML_DEFAULT_WEIGHT,
    MIN_PROB_BUY,
    MAX_PROB_SELL,
)
from src.utils.enums import Signal
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Import ML utilities with fallback
try:
    from src.utils import load_model, predict
    ML_AVAILABLE = True
except ImportError:
    load_model = None
    predict = None
    ML_AVAILABLE = False


def combine_signals(
    ml_signal: int,
    scalp_signal: int,
    ml_weight: float = ML_DEFAULT_WEIGHT,
    threshold: float = 0.5,
) -> int:
    """
    Combine ML and scalping signals using weighted average.
    
    Args:
        ml_signal: ML model signal (-1, 0, 1)
        scalp_signal: Scalping signal (-1, 0, 1)
        ml_weight: Weight for ML signal (0-1)
        threshold: Minimum score to generate signal
    
    Returns:
        Combined signal (-1, 0, 1)
    """
    scalp_weight = 1.0 - ml_weight
    score = (ml_signal * ml_weight) + (scalp_signal * scalp_weight)

    if score >= threshold:
        return Signal.BUY.value
    elif score <= -threshold:
        return Signal.SELL.value
    return Signal.HOLD.value


def calculate_ml_signals(
    df: pd.DataFrame,
    ticker: str,
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate ML-based trading signals.
    
    Args:
        df: DataFrame with feature data
        ticker: Ticker symbol for model lookup
    
    Returns:
        Tuple of (ml_signals, ml_weights)
    """
    ml_signals = pd.Series(0, index=df.index)
    ml_weights = pd.Series(0.0, index=df.index)
    
    if not ML_AVAILABLE or load_model is None:
        logger.warning("ML models not available, using scalping-only signals")
        return ml_signals, ml_weights
    
    try:
        model, scaler, features = load_model(ticker)
        preds, probs = predict(model, scaler, features, df)
        
        if probs is None:
            # Model doesn't support probabilities
            ml_signals = pd.Series(preds, index=df.index)
            ml_weights = pd.Series(ML_DEFAULT_WEIGHT, index=df.index)
        else:
            # Handle multi-class probabilities
            if len(probs.shape) > 1:
                probs = probs[:, 1]  # Use positive class probability
            
            probs_series = pd.Series(probs, index=df.index)
            
            # Generate signals based on probability thresholds
            ml_signals = np.where(
                probs_series >= MIN_PROB_BUY, Signal.BUY.value,
                np.where(probs_series <= MAX_PROB_SELL, Signal.SELL.value, Signal.HOLD.value)
            )
            ml_signals = pd.Series(ml_signals, index=df.index)
            
            # Dynamic weight based on confidence
            confidence = abs(probs_series - 0.5) * 2
            ml_weights = confidence.clip(0.3, 0.8)
        
        logger.info(f"ML signals generated for {ticker}")
        
    except FileNotFoundError:
        logger.warning(f"No model found for {ticker}, using scalping-only signals")
    except Exception as e:
        logger.error(f"Error generating ML signals: {e}")
    
    return ml_signals, ml_weights


def generate_combined_strategy(
    ticker: str,
    data: pd.DataFrame,
    ml_weight: float = ML_DEFAULT_WEIGHT,
    use_dynamic_weight: bool = True,
) -> pd.DataFrame:
    """
    Generate combined ML + scalping trading signals.
    
    Args:
        ticker: Symbol for ML model lookup
        data: DataFrame with OHLCV and indicator data
        ml_weight: Base weight for ML signals (0-1)
        use_dynamic_weight: Whether to adjust weights based on ML confidence
    
    Returns:
        DataFrame with signal columns added:
        - scalp_signal: Technical indicator signal
        - ml_signal: ML model signal
        - ml_weight: Weight applied to ML signal
        - combined_signal: Final trading signal
        - signal_strength: Confidence level of signal
    """
    df = data.copy()
    
    # Generate scalping signals
    df = calculate_scalping_signals(df)
    if "scalp_signal" not in df.columns:
        df["scalp_signal"] = Signal.HOLD.value
    
    # Generate ML signals
    df["ml_signal"], df["ml_weight"] = calculate_ml_signals(df, ticker)
    
    # Use fixed weight if dynamic not enabled
    if not use_dynamic_weight:
        df["ml_weight"] = ml_weight
    
    # Combine signals
    df["combined_signal"] = df.apply(
        lambda row: combine_signals(
            int(row["ml_signal"]),
            int(row["scalp_signal"]),
            float(row["ml_weight"]),
        ),
        axis=1,
    ).astype(int)
    
    # Calculate signal strength (useful for position sizing)
    df["signal_strength"] = (
        abs(df["ml_signal"] * df["ml_weight"]) +
        abs(df["scalp_signal"] * (1 - df["ml_weight"]))
    ).clip(0, 1)
    
    # Prevent lookahead bias by shifting signals
    df["combined_signal"] = df["combined_signal"].shift(1).fillna(0).astype(int)
    df["signal_strength"] = df["signal_strength"].shift(1).fillna(0)
    
    logger.info(
        f"Generated signals for {ticker}: "
        f"Buy={sum(df['combined_signal'] == 1)}, "
        f"Sell={sum(df['combined_signal'] == -1)}, "
        f"Hold={sum(df['combined_signal'] == 0)}"
    )
    
    return df


class CombinedStrategy:
    """
    Object-oriented wrapper for the combined strategy.
    
    Allows for more flexible configuration and state management.
    """
    
    def __init__(
        self,
        ml_weight: float = ML_DEFAULT_WEIGHT,
        use_dynamic_weight: bool = True,
        signal_threshold: float = 0.5,
    ):
        self.ml_weight = ml_weight
        self.use_dynamic_weight = use_dynamic_weight
        self.signal_threshold = signal_threshold
        self._last_signals: Optional[pd.DataFrame] = None
    
    def generate_signals(
        self,
        ticker: str,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate trading signals for the given data."""
        result = generate_combined_strategy(
            ticker=ticker,
            data=data,
            ml_weight=self.ml_weight,
            use_dynamic_weight=self.use_dynamic_weight,
        )
        self._last_signals = result
        return result
    
    def get_current_signal(self) -> Optional[int]:
        """Get the most recent signal."""
        if self._last_signals is None or len(self._last_signals) == 0:
            return None
        return int(self._last_signals["combined_signal"].iloc[-1])
    
    def get_signal_stats(self) -> dict:
        """Get statistics about generated signals."""
        if self._last_signals is None:
            return {}
        
        signals = self._last_signals["combined_signal"]
        return {
            "total_signals": len(signals),
            "buy_signals": int((signals == 1).sum()),
            "sell_signals": int((signals == -1).sum()),
            "hold_signals": int((signals == 0).sum()),
            "buy_ratio": (signals == 1).mean(),
            "avg_strength": self._last_signals["signal_strength"].mean(),
        }
