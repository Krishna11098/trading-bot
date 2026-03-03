"""Tests for ML models."""
import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.modeling.train_model import prepare_features, train_random_forest


class TestModel(unittest.TestCase):
    """Test cases for ML models."""
    
    def setUp(self):
        """Setup test data."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200)
        self.test_data = pd.DataFrame({
            'Close': np.random.rand(200) * 100 + 100,
            'Volume': np.random.rand(200) * 1000000,
            'RSI': np.random.rand(200) * 100,
            'SMA_20': np.random.rand(200) * 100 + 100,
            'SMA_50': np.random.rand(200) * 100 + 100
        }, index=dates)
    
    def test_prepare_features(self):
        """Test feature preparation."""
        X_train, X_test, y_train, y_test, scaler, features = prepare_features(self.test_data)
        
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(features), 0)
    
    def test_train_model(self):
        """Test model training."""
        X_train, X_test, y_train, y_test, scaler, features = prepare_features(self.test_data)
        
        model = train_random_forest(X_train, y_train, n_estimators=10)
        self.assertIsNotNone(model)
        
        # Test prediction
        predictions = model.predict(X_test)
        self.assertEqual(len(predictions), len(X_test))


if __name__ == '__main__':
    unittest.main()

