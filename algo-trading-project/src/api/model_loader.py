"""
Model loader and cache for API.
"""
import joblib
from pathlib import Path
from typing import Dict, Tuple, Any
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelCache:
    """
    Cache for loaded models to avoid repeated disk I/O.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize model cache.
        
        Args:
            models_dir: Directory containing saved models
        """
        self.models_dir = Path(__file__).resolve().parents[2] / models_dir

        self.cache: Dict[str, Tuple[Any, Any, list]] = {}
    
    def load_model(self, ticker: str) -> Tuple[Any, Any, list]:
        """
        Load model from cache or disk.
        
        Args:
            ticker: Stock ticker
        
        Returns:
            model, scaler, features
        """
        # Check cache first
        if ticker in self.cache:
            logger.info(f"Loading {ticker} model from cache")
            return self.cache[ticker]
        
        # Load from disk
        logger.info(f"Loading {ticker} model from disk")
        model_path = self.models_dir / f"{ticker}_model.pkl"
        scaler_path = self.models_dir / f"{ticker}_scaler.pkl"
        features_path = self.models_dir / f"{ticker}_features.pkl"
        
        if not all([model_path.exists(), scaler_path.exists(), features_path.exists()]):
            raise FileNotFoundError(f"Model files not found for {ticker}")
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)
        
        # Store in cache
        self.cache[ticker] = (model, scaler, features)
        
        return model, scaler, features
    
    def preload_models(self, tickers: list):
        """
        Preload multiple models into cache.
        
        Args:
            tickers: List of tickers to preload
        """
        for ticker in tickers:
            try:
                self.load_model(ticker)
                logger.info(f"Preloaded model for {ticker}")
            except FileNotFoundError:
                logger.warning(f"Could not preload model for {ticker}")
    
    def clear_cache(self):
        """Clear the model cache."""
        self.cache.clear()
        logger.info("Model cache cleared")
    
    def get_cached_tickers(self) -> list:
        """
        Get list of tickers currently in cache.
        
        Returns:
            List of ticker symbols
        """
        return list(self.cache.keys())


# Global model cache instance
_model_cache = None


def get_model_cache(models_dir: str = "models") -> ModelCache:
    """
    Get global model cache instance.
    
    Args:
        models_dir: Directory containing saved models
    
    Returns:
        ModelCache instance
    """
    global _model_cache
    
    if _model_cache is None:
        _model_cache = ModelCache(models_dir)
    
    return _model_cache
def load_model(ticker: str):
    """
    Public helper to load a model for a ticker.
    Used by strategy / backtest modules.
    """
    cache = get_model_cache()
    return cache.load_model(ticker)


