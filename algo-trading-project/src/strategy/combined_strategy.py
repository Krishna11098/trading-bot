"""
Combine ML predictions with scalping signals (proper version).
"""

import numpy as np
import pandas as pd

from src.strategy.scalping_logic import calculate_scalping_signals
from src.utils.config import (
    ML_DEFAULT_WEIGHT,
    MIN_PROB_BUY,
    MAX_PROB_SELL,
)

try:
    from src.utils import load_model, predict
except Exception:
    load_model = None
    predict = None


# --------------------------------------------------
# SIGNAL COMBINATION
# --------------------------------------------------
def combine_signals(
    ml_signal: int,
    scalp_signal: int,
    ml_weight: float,
) -> int:
    scalp_weight = 1.0 - ml_weight
    score = (ml_signal * ml_weight) + (scalp_signal * scalp_weight)

    if score >= 0.5:
        return 1
    elif score <= -0.5:
        return -1
    else:
        return 0


# --------------------------------------------------
# MAIN STRATEGY
# --------------------------------------------------
def generate_combined_strategy(
    ticker: str,
    data: pd.DataFrame,
    ml_weight: float = ML_DEFAULT_WEIGHT,
) -> pd.DataFrame:
    """
    Generate combined ML + scalping signals.
    """

    df = data.copy()

    # ---------- SCALPING ----------
    df = calculate_scalping_signals(df)

    if "scalp_signal" not in df.columns:
        df["scalp_signal"] = 0

    # ---------- ML ----------
    df["ml_signal"] = 0

    if load_model and predict:
        try:
            model, scaler, features = load_model(ticker)
            preds, probs = predict(model, scaler, features, df)

            # ML confidence-based signal
            df["ml_signal"] = np.where(
                probs >= MIN_PROB_BUY, 1,
                np.where(probs <= MAX_PROB_SELL, -1, 0)
            )

            # Dynamic ML weight (stronger confidence → more weight)
            confidence = abs(probs - 0.5) * 2
            df["ml_weight"] = np.clip(confidence, 0.3, 0.8)

        except Exception:
            df["ml_signal"] = 0
            df["ml_weight"] = 0.0
    else:
        df["ml_weight"] = 0.0

    # ---------- COMBINE ----------
    df["combined_signal"] = df.apply(
        lambda r: combine_signals(
            r["ml_signal"],
            r["scalp_signal"],
            r["ml_weight"] if "ml_weight" in df.columns else ml_weight,
        ),
        axis=1,
    )

    # Prevent lookahead bias
    df["combined_signal"] = df["combined_signal"].shift(1).fillna(0)

    return df
