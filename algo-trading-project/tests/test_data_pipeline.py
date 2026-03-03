"""Tests for data collection and preprocessing pipeline."""
import unittest
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.data_collection.download_data import download_stock_data
from src.preprocessing.clean_data import clean_ohlcv_data
from src.preprocessing.feature_engineering import add_technical_indicators


class TestDataPipeline(unittest.TestCase):
    """Test cases for data pipeline."""
    
    def test_download_data(self):
        """Test data download functionality."""
        # This test requires internet connection
        try:
            data = download_stock_data('AAPL', '2024-01-01', '2024-01-31')
            self.assertIsInstance(data, pd.DataFrame)
            self.assertGreater(len(data), 0)
            self.assertTrue('Close' in data.columns)
        except Exception as e:
            self.skipTest(f"Skipping due to network issue: {e}")
    
    def test_clean_data(self):
        """Test data cleaning."""
        # Create sample data
        data = pd.DataFrame({
            'Open': [100, 101, None, 103],
            'High': [102, 103, 104, 105],
            'Low': [99, 100, 101, 102],
            'Close': [101, 102, 103, 104],
            'Volume': [1000, 1100, 1200, 1300]
        })
        
        cleaned = clean_ohlcv_data(data)
        self.assertFalse(cleaned.isnull().any().any())
    
    def test_add_indicators(self):
        """Test technical indicator calculation."""
        # Create sample data with enough rows for indicators
        dates = pd.date_range('2023-01-01', periods=60)
        data = pd.DataFrame({
            'Open': range(100, 160),
            'High': range(102, 162),
            'Low': range(99, 159),
            'Close': range(101, 161),
            'Volume': [1000] * 60
        }, index=dates)
        
        with_indicators = add_technical_indicators(data)
        self.assertTrue('RSI' in with_indicators.columns)
        self.assertTrue('SMA_20' in with_indicators.columns)


if __name__ == '__main__':
    unittest.main()

