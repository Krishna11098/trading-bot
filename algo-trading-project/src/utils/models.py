"""
ML model helpers for loading and prediction.

This module wraps the API's model loader and provides a `predict`
helper that works with DataFrame inputs (used by strategies/backtester).
"""
from typing import Any, Tuple, List
import numpy as np
import pandas as pd

try:
    from src.api.model_loader import load_model as _api_load_model
except Exception:
    # Fallback for notebook / import contexts where src package path differs
    from api.model_loader import load_model as _api_load_model


def load_model(ticker: str) -> Tuple[Any, Any, List[str]]:
    """Load (model, scaler, features) for a ticker using the global cache.

    Raises FileNotFoundError if model files are missing.
    """
    return _api_load_model(ticker)


def predict(model: Any, scaler: Any, features: list, df: pd.DataFrame) -> Tuple[np.ndarray, Any]:
    """Run model prediction on `df` using the provided `features` and `scaler`.

    Returns (preds, probs) where `probs` is None if `predict_proba` is not
    supported by the estimator.
    """
    if len(df) == 0:
        return np.array([], dtype=int), None

    # Ensure features exist
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[features].copy()
    # Basic clean: forward-fill then fill remaining NA with 0
    X = X.fillna(method="ffill").fillna(0)

    # Scale
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)

    probs = None
    try:
        probs = model.predict_proba(X_scaled)
    except Exception:
        # Some models don't implement predict_proba (e.g., certain SKLearn wrappers)
        probs = None

    return preds, probs
