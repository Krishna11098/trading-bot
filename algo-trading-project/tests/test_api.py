"""Tests for FastAPI application."""
import unittest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.api.main import app


class TestAPI(unittest.TestCase):
    """Test cases for API endpoints."""
    
    def setUp(self):
        """Setup test client."""
        self.client = TestClient(app)
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('message', data)
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'healthy')
    
    def test_get_tickers(self):
        """Test get tickers endpoint."""
        response = self.client.get("/tickers")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('tickers', data)
        self.assertIsInstance(data['tickers'], list)
    
    def test_list_models(self):
        """Test list models endpoint."""
        response = self.client.get("/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('available_models', data)


if __name__ == '__main__':
    unittest.main()

