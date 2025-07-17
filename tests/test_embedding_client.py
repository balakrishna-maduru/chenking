import unittest
import logging
import time
from unittest.mock import patch, Mock, MagicMock
import requests
from chenking.embedding_client import EmbeddingClient


class TestEmbeddingClient(unittest.TestCase):
    """Test cases for EmbeddingClient class."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        self.api_url = "https://api.example.com/embeddings"
        self.client = EmbeddingClient(self.api_url)
        
        self.sample_check_data = {
            "word_count": 25,
            "char_count": 150,
            "has_content": True
        }

    def test_init_default_values(self) -> None:
        """Test initialization with default values."""
        client = EmbeddingClient("http://test.com")
        
        self.assertEqual(client.api_url, "http://test.com")
        self.assertEqual(client.timeout, 30)
        self.assertEqual(client.max_retries, 3)
        self.assertIsNotNone(client.logger)

    def test_init_custom_values(self) -> None:
        """Test initialization with custom values."""
        client = EmbeddingClient(
            "http://custom.com", 
            timeout=60, 
            max_retries=5
        )
        
        self.assertEqual(client.api_url, "http://custom.com")
        self.assertEqual(client.timeout, 60)
        self.assertEqual(client.max_retries, 5)

    @patch('chenking.embedding_client.requests.post')
    def test_get_embedding_success(self, mock_post: Mock) -> None:
        """Test successful embedding generation."""
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
            "vector": [0.6, 0.7, 0.8, 0.9, 1.0]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        result = self.client.get_embedding(self.sample_check_data)
        
        # Verify request was made correctly
        mock_post.assert_called_once_with(
            self.api_url,
            json=self.sample_check_data,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        # Verify response structure
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["embedding"], [0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertEqual(result["vector"], [0.6, 0.7, 0.8, 0.9, 1.0])
        self.assertIn("request_time", result)
        self.assertIsInstance(result["request_time"], float)

    @patch('chenking.embedding_client.requests.post')
    def test_get_embedding_timeout(self, mock_post: Mock) -> None:
        """Test embedding request timeout handling."""
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
        
        result = self.client.get_embedding(self.sample_check_data)
        
        # Should retry max_retries + 1 times
        self.assertEqual(mock_post.call_count, self.client.max_retries + 1)
        
        # Verify error response
        self.assertEqual(result["status"], "timeout")
        self.assertIsNone(result["embedding"])
        self.assertIsNone(result["vector"])
        self.assertIn("Timeout after", result["error"])

    @patch('chenking.embedding_client.requests.post')
    def test_get_embedding_request_error(self, mock_post: Mock) -> None:
        """Test embedding request error handling."""
        mock_post.side_effect = requests.exceptions.RequestException("Connection error")
        
        result = self.client.get_embedding(self.sample_check_data)
        
        # Should retry max_retries + 1 times
        self.assertEqual(mock_post.call_count, self.client.max_retries + 1)
        
        # Verify error response
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["embedding"])
        self.assertIsNone(result["vector"])
        self.assertIn("Request failed after", result["error"])

    @patch('chenking.embedding_client.requests.post')
    def test_get_embedding_http_error(self, mock_post: Mock) -> None:
        """Test embedding HTTP error handling."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_post.return_value = mock_response
        
        result = self.client.get_embedding(self.sample_check_data)
        
        # Should retry max_retries + 1 times
        self.assertEqual(mock_post.call_count, self.client.max_retries + 1)
        
        # Verify error response
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["embedding"])
        self.assertIsNone(result["vector"])
        self.assertIn("Request failed after", result["error"])

    @patch('chenking.embedding_client.requests.post')
    def test_get_embedding_unexpected_error(self, mock_post: Mock) -> None:
        """Test handling of unexpected errors."""
        mock_post.side_effect = ValueError("Unexpected error")
        
        result = self.client.get_embedding(self.sample_check_data)
        
        # Should not retry for unexpected errors
        self.assertEqual(mock_post.call_count, 1)
        
        # Verify error response
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["embedding"])
        self.assertIsNone(result["vector"])
        self.assertIn("Unexpected error", result["error"])

    @patch('chenking.embedding_client.requests.post')
    @patch('chenking.embedding_client.time.sleep')
    def test_get_embedding_retry_with_backoff(self, mock_sleep: Mock, mock_post: Mock) -> None:
        """Test retry mechanism with exponential backoff."""
        # First two calls fail, third succeeds
        mock_response_success = Mock()
        mock_response_success.json.return_value = {"embedding": [0.1], "vector": [0.2]}
        mock_response_success.raise_for_status.return_value = None
        
        mock_post.side_effect = [
            requests.exceptions.RequestException("Error 1"),
            requests.exceptions.RequestException("Error 2"),
            mock_response_success
        ]
        
        result = self.client.get_embedding(self.sample_check_data)
        
        # Should make 3 calls (2 failures + 1 success)
        self.assertEqual(mock_post.call_count, 3)
        
        # Should sleep twice with exponential backoff
        self.assertEqual(mock_sleep.call_count, 2)
        mock_sleep.assert_any_call(1)  # 2^0
        mock_sleep.assert_any_call(2)  # 2^1
        
        # Should eventually succeed
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["embedding"], [0.1])

    @patch('chenking.embedding_client.requests.post')
    def test_get_embedding_partial_response(self, mock_post: Mock) -> None:
        """Test handling of partial API response."""
        # Mock response with only embedding, missing vector
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        result = self.client.get_embedding(self.sample_check_data)
        
        # Should handle missing vector gracefully
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["embedding"], [0.1, 0.2, 0.3])
        self.assertIsNone(result["vector"])

    @patch('chenking.embedding_client.requests.get')
    def test_health_check_success(self, mock_get: Mock) -> None:
        """Test successful health check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = self.client.health_check()
        
        self.assertTrue(result)
        mock_get.assert_called_once_with(
            f"{self.api_url}/health",
            timeout=30
        )

    @patch('chenking.embedding_client.requests.get')
    def test_health_check_failure(self, mock_get: Mock) -> None:
        """Test failed health check."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        result = self.client.health_check()
        
        self.assertFalse(result)

    @patch('chenking.embedding_client.requests.get')
    def test_health_check_exception(self, mock_get: Mock) -> None:
        """Test health check with exception."""
        mock_get.side_effect = requests.exceptions.RequestException("Connection failed")
        
        result = self.client.health_check()
        
        self.assertFalse(result)

    @patch('chenking.embedding_client.requests.get')
    def test_health_check_with_trailing_slash(self, mock_get: Mock) -> None:
        """Test health check URL handling with trailing slash."""
        client = EmbeddingClient("http://test.com/")
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = client.health_check()
        
        # Should strip trailing slash
        mock_get.assert_called_once_with(
            "http://test.com/health",
            timeout=30
        )
        self.assertTrue(result)

    def test_custom_timeout_and_retries(self) -> None:
        """Test client with custom timeout and retry settings."""
        client = EmbeddingClient(
            "http://test.com",
            timeout=45,
            max_retries=2
        )
        
        self.assertEqual(client.timeout, 45)
        self.assertEqual(client.max_retries, 2)

    @patch('chenking.embedding_client.requests.post')
    def test_logging_debug_messages(self, mock_post: Mock) -> None:
        """Test that debug logging messages are generated."""
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": [0.1], "vector": [0.2]}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        with patch.object(self.client.logger, 'debug') as mock_debug:
            self.client.get_embedding(self.sample_check_data)
            
            # Should log debug messages
            self.assertTrue(mock_debug.called)
            # Check for specific debug messages
            debug_calls = [call[0][0] for call in mock_debug.call_args_list]
            self.assertTrue(any("Requesting embedding" in msg for msg in debug_calls))
            self.assertTrue(any("completed in" in msg for msg in debug_calls))

    @patch('chenking.embedding_client.requests.post')
    def test_empty_check_data(self, mock_post: Mock) -> None:
        """Test embedding generation with empty check data."""
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": [], "vector": []}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        result = self.client.get_embedding({})
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["embedding"], [])
        self.assertEqual(result["vector"], [])

    @patch('chenking.embedding_client.requests.post')
    def test_large_check_data(self, mock_post: Mock) -> None:
        """Test embedding generation with large check data."""
        large_data = {f"key_{i}": f"value_{i}" for i in range(1000)}
        
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": [0.1] * 100, "vector": [0.2] * 100}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        result = self.client.get_embedding(large_data)
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["embedding"]), 100)
        
        # Verify large data was sent
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(call_args[1]["json"], large_data)


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.WARNING)
    
    # Create test suite
    unittest.main(verbosity=2)
