import unittest
import logging
import time
from unittest.mock import patch, MagicMock, Mock
from chenking.document_processor import DocumentProcessor
from chenking.document_checker import DocumentChecker
from chenking.embedding_client import EmbeddingClient


class TestDocumentProcessor(unittest.TestCase):
    """Test cases for DocumentProcessor class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.api_url = "https://api.example.com/embeddings"
        self.processor = DocumentProcessor(self.api_url)
        
        # Sample documents for testing
        self.valid_document = {
            "id": "doc_001",
            "content": "This is a comprehensive test document with sufficient content for validation. It contains multiple sentences and demonstrates the document processing functionality.",
            "title": "Test Document",
            "author": "Test Author",
            "format": "txt",
            "metadata": {"category": "test", "version": "1.0"}
        }
        
        self.minimal_document = {
            "content": "Minimal test content for basic validation."
        }
        
        self.invalid_document = {
            "title": "No Content Document"
            # Missing content field
        }

    def test_init_default_values(self):
        """Test initialization with default DocumentChecker configuration."""
        processor = DocumentProcessor("http://test.com")
        
        self.assertEqual(processor.api_url, "http://test.com")
        self.assertIsInstance(processor.checker, DocumentChecker)
        self.assertIsInstance(processor.embedder, EmbeddingClient)
        self.assertEqual(processor.checker.min_word_count, 10)  # Default value

    def test_init_custom_checker_config(self):
        """Test initialization with custom DocumentChecker configuration."""
        processor = DocumentProcessor(
            "http://test.com",
            min_word_count=5,
            max_word_count=500,
            enable_detailed_logging=True
        )
        
        self.assertEqual(processor.checker.min_word_count, 5)
        self.assertEqual(processor.checker.max_word_count, 500)
        self.assertTrue(processor.checker.enable_detailed_logging)

    @patch('chenking.embedding_client.requests.post')
    def test_process_valid_document(self, mock_post):
        """Test processing a valid document successfully."""
        # Mock embedding API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "embedding": [0.1, 0.2, 0.3],
            "vector": [0.4, 0.5, 0.6]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        result = self.processor.process(self.valid_document)
        
        # Verify structure
        self.assertIn("id", result)
        self.assertIn("content", result)
        self.assertIn("metadata", result)
        self.assertIn("chenkings", result)
        self.assertIn("processing_info", result)
        
        # Verify content
        self.assertEqual(result["id"], "doc_001")
        self.assertEqual(result["content"], self.valid_document["content"])
        
        # Verify processing info
        self.assertIn("timestamp", result["processing_info"])
        self.assertIn("total_processing_time", result["processing_info"])
        self.assertIn("checks_completed", result["processing_info"])
        self.assertIn("successful_checks", result["processing_info"])
        
        # Verify checks were run
        self.assertGreater(len(result["chenkings"]), 0)
        
        # Verify embedding was called for successful checks
        for check_name, check_result in result["chenkings"].items():
            if check_result["status"] == "success":
                self.assertIsNotNone(check_result.get("embedding"))

    def test_process_invalid_document_type(self):
        """Test processing with invalid document type."""
        with self.assertRaises(ValueError) as context:
            self.processor.process("not a dictionary")
        
        self.assertIn("Document must be a dictionary", str(context.exception))

    def test_process_document_missing_content(self):
        """Test processing document without content field."""
        with self.assertRaises(ValueError) as context:
            self.processor.process(self.invalid_document)
        
        self.assertIn("Document must contain 'content' field", str(context.exception))

    def test_process_document_empty_content(self):
        """Test processing document with empty content."""
        empty_doc = {"content": ""}
        
        with self.assertRaises(ValueError) as context:
            self.processor.process(empty_doc)
        
        self.assertIn("Document must contain 'content' field", str(context.exception))

    @patch('chenking.embedding_client.requests.post')
    def test_process_with_embedding_error(self, mock_post):
        """Test processing when embedding API fails."""
        # Mock embedding API error
        mock_post.side_effect = Exception("API Error")
        
        result = self.processor.process(self.minimal_document)
        
        # Should still return result structure
        self.assertIn("chenkings", result)
        
        # Check that embedding errors are handled
        for check_name, check_result in result["chenkings"].items():
            if check_result["status"] == "success":
                self.assertIsNone(check_result.get("embedding"))
                self.assertIn("embedding_error", check_result)

    @patch('chenking.document_checker.DocumentChecker.run_checks')
    def test_process_with_checker_error(self, mock_run_checks):
        """Test processing when document checker fails."""
        # Mock checker error
        mock_run_checks.side_effect = Exception("Checker Error")
        
        result = self.processor.process(self.valid_document)
        
        # Should handle error gracefully
        self.assertIn("processing_info", result)
        self.assertIn("error", result["processing_info"])
        self.assertEqual(result["processing_info"]["status"], "failed")

    @patch('chenking.embedding_client.requests.post')
    def test_process_with_failed_checks(self, mock_post):
        """Test processing when some checks fail."""
        # Mock embedding response (won't be called for failed checks)
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2], "vector": [0.3, 0.4]}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        # Create processor with strict limits to force check failures
        processor = DocumentProcessor(
            self.api_url,
            min_word_count=100,  # Very high limit to force failure
            max_word_count=200
        )
        
        result = processor.process(self.minimal_document)
        
        # Should handle failed checks
        for check_name, check_result in result["chenkings"].items():
            if check_result["status"] == "error":
                self.assertIsNone(check_result.get("embedding"))
                self.assertEqual(
                    check_result.get("embedding_error"), 
                    "Check failed, no embedding generated"
                )

    @patch('chenking.embedding_client.requests.post')
    def test_process_batch_success(self, mock_post):
        """Test batch processing of multiple documents."""
        # Mock embedding API response
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2], "vector": [0.3, 0.4]}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        documents = [
            {"id": "doc1", "content": "First test document content."},
            {"id": "doc2", "content": "Second test document content."},
            {"id": "doc3", "content": "Third test document content."}
        ]
        
        results = self.processor.process_batch(documents)
        
        self.assertEqual(len(results), 3)
        for i, result in enumerate(results):
            self.assertEqual(result["id"], f"doc{i+1}")
            self.assertIn("chenkings", result)
            self.assertIn("processing_info", result)

    @patch('chenking.embedding_client.requests.post')
    def test_process_batch_with_errors(self, mock_post):
        """Test batch processing with some document errors."""
        # Mock embedding API response for successful documents
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2], "vector": [0.3, 0.4]}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        documents = [
            {"id": "doc1", "content": "Valid document content."},
            {"id": "doc2"},  # Missing content
            {"id": "doc3", "content": "Another valid document."}
        ]
        
        results = self.processor.process_batch(documents)
        
        self.assertEqual(len(results), 3)
        
        # First and third should succeed, second should fail
        self.assertNotIn("error", results[0].get("processing_info", {}))
        self.assertIn("error", results[1]["processing_info"])
        self.assertEqual(results[1]["processing_info"]["status"], "failed")
        self.assertNotIn("error", results[2].get("processing_info", {}))

    @patch('chenking.embedding_client.requests.post')
    def test_get_processing_stats(self, mock_post):
        """Test generation of processing statistics."""
        # Mock embedding API response
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2], "vector": [0.3, 0.4]}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        documents = [
            {"id": "doc1", "content": "First test document content."},
            {"id": "doc2", "content": "Second test document content."}
        ]
        
        processed_docs = self.processor.process_batch(documents)
        stats = self.processor.get_processing_stats(processed_docs)
        
        # Verify statistics structure
        self.assertIn("total_documents", stats)
        self.assertIn("successful_documents", stats)
        self.assertIn("failed_documents", stats)
        self.assertIn("success_rate", stats)
        self.assertIn("total_processing_time", stats)
        self.assertIn("average_processing_time", stats)
        self.assertIn("check_statistics", stats)
        
        # Verify values
        self.assertEqual(stats["total_documents"], 2)
        self.assertEqual(stats["successful_documents"], 2)
        self.assertEqual(stats["failed_documents"], 0)
        self.assertEqual(stats["success_rate"], 1.0)
        self.assertGreater(stats["total_processing_time"], 0)

    def test_get_processing_stats_empty_list(self):
        """Test statistics generation with empty document list."""
        stats = self.processor.get_processing_stats([])
        
        self.assertEqual(stats["total_documents"], 0)

    def test_get_processing_stats_with_failures(self):
        """Test statistics generation with failed documents."""
        # Create mock processed documents with failures
        processed_docs = [
            {
                "id": "doc1",
                "chenkings": {
                    "basic_check": {"status": "success"},
                    "length_check": {"status": "success"}
                },
                "processing_info": {"total_processing_time": 0.1}
            },
            {
                "id": "doc2", 
                "chenkings": {
                    "basic_check": {"status": "error"},
                    "length_check": {"status": "success"}
                },
                "processing_info": {
                    "total_processing_time": 0.05,
                    "status": "failed"
                }
            }
        ]
        
        stats = self.processor.get_processing_stats(processed_docs)
        
        self.assertEqual(stats["total_documents"], 2)
        self.assertEqual(stats["successful_documents"], 1)
        self.assertEqual(stats["failed_documents"], 1)
        self.assertEqual(stats["success_rate"], 0.5)
        
        # Check individual check statistics
        self.assertEqual(stats["check_statistics"]["basic_check"]["success"], 1)
        self.assertEqual(stats["check_statistics"]["basic_check"]["failed"], 1)
        self.assertEqual(stats["check_statistics"]["length_check"]["success"], 2)
        self.assertEqual(stats["check_statistics"]["length_check"]["failed"], 0)

    def test_logging_setup(self):
        """Test that logging is properly configured."""
        processor = DocumentProcessor("http://test.com", enable_detailed_logging=True)
        self.assertIsNotNone(processor.logger)
        self.assertEqual(processor.logger.name, "DocumentProcessor")

    @patch('chenking.embedding_client.requests.post')
    def test_integration_with_custom_config(self, mock_post):
        """Test end-to-end processing with custom configuration."""
        # Mock embedding response
        mock_response = Mock()
        mock_response.json.return_value = {
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
            "vector": [0.6, 0.7, 0.8, 0.9, 1.0]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        # Create processor with custom configuration
        processor = DocumentProcessor(
            "http://custom.api.com/embed",
            min_word_count=3,
            max_word_count=50,
            supported_formats=["txt", "md"],
            required_fields=["content", "title"],
            enable_detailed_logging=True
        )
        
        document = {
            "id": "custom_test",
            "content": "Short but valid content for testing.",
            "title": "Custom Test",
            "format": "txt"
        }
        
        result = processor.process(document)
        
        # Verify the processor used custom configuration
        self.assertEqual(result["id"], "custom_test")
        self.assertIn("chenkings", result)
        
        # Should have successful checks due to relaxed word count
        successful_checks = sum(
            1 for check in result["chenkings"].values() 
            if check["status"] == "success"
        )
        self.assertGreater(successful_checks, 0)


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.WARNING)
    
    # Create test suite
    unittest.main(verbosity=2)
