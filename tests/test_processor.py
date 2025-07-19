import unittest
import logging
import time
from unittest.mock import patch, MagicMock, Mock
from chenking.processor import Processor
from chenking.chenker import Chenker
from chenking.embedding_client import EmbeddingClient


class TestProcessor(unittest.TestCase):
    """Test cases for Processor class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.api_url = "https://api.example.com/embeddings"
        self.processor = Processor(self.api_url)
        
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
        """Test initialization with default Chenker configuration."""
        processor = Processor("http://test.com")
        
        self.assertEqual(processor.api_url, "http://test.com")
        self.assertIsInstance(processor.checker, Chenker)
        self.assertIsInstance(processor.embedder, EmbeddingClient)
        self.assertEqual(processor.checker.min_word_count, 10)  # Default value

    def test_init_custom_checker_config(self):
        """Test initialization with custom Chenker configuration."""
        processor = Processor(
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

    @patch('chenking.chenker.Chenker.run_checks')
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
        processor = Processor(
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
        processor = Processor("http://test.com", enable_detailed_logging=True)
        self.assertIsNotNone(processor.logger)
        self.assertEqual(processor.logger.name, "Processor")

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
        processor = Processor(
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

    def test_failed_check_handling(self):
        """Test handling of failed checks without embeddings."""
        # Create a document that will fail validation
        failing_document = {
            "content": "Short",  # Too short to pass validation
            "id": "failing_doc"
        }
        
        # Mock embedder to simulate failure
        with patch.object(self.processor.embedder, 'get_embedding', return_value=None):
            result = self.processor.process(failing_document)
        
        # Check that failed checks are handled properly
        self.assertEqual(result["id"], "failing_doc")
        self.assertIn("chenkings", result)
        
        # At least one check should fail
        failed_checks = [
            check for check in result["chenkings"].values() 
            if check["status"] == "error"
        ]
        self.assertGreater(len(failed_checks), 0)
        
        # Failed checks should still contain original document context
        for check in failed_checks:
            self.assertIn("check_content", check)
            self.assertIn("check_metadata", check)
            self.assertEqual(check["check_content"], "Short")

    def test_edge_case_no_content_field(self):
        """Test processing document without content field."""
        document_no_content = {
            "id": "no_content_doc",
            "title": "Document without content"
        }
        
        result = self.processor.process(document_no_content)
        
        # Should handle gracefully
        self.assertEqual(result["id"], "no_content_doc")
        self.assertIn("chenkings", result)

    def test_edge_case_empty_metadata(self):
        """Test processing document with empty or missing metadata."""
        document_empty_metadata = {
            "content": "This document has no metadata",
            "id": "empty_meta_doc",
            "metadata": {}
        }
        
        result = self.processor.process(document_empty_metadata)
        
        # Should process successfully
        self.assertEqual(result["id"], "empty_meta_doc")
        self.assertIn("chenkings", result)
        
        # Check that empty metadata is handled in failed checks
        for check in result["chenkings"].values():
            self.assertIn("check_metadata", check)

    def test_embedding_error_propagation(self):
        """Test that embedding errors are properly propagated."""
        document = {
            "content": "This is a test document with sufficient content for validation.",
            "id": "embedding_error_test"
        }
        
        # Mock embedding client to return error
        with patch.object(self.processor.embedder, 'get_embedding') as mock_embed:
            mock_embed.return_value = {"error": "Embedding service unavailable"}
            
            result = self.processor.process(document)
        
        # Check that embedding errors are captured
        self.assertEqual(result["id"], "embedding_error_test")
        
        # Find checks with embeddings
        checks_with_embeddings = [
            check for check in result["chenkings"].values()
            if "embedding_error" in check
        ]
        
        # At least one check should have embedding error
        if checks_with_embeddings:
            self.assertTrue(any(
                check.get("embedding_error") == "Embedding service unavailable"
                for check in checks_with_embeddings
            ))

    def test_concurrent_processing(self):
        """Test concurrent processing of multiple documents."""
        import threading
        
        documents = [
            {
                "content": f"Test document {i} with sufficient content for processing.",
                "id": f"concurrent_doc_{i}"
            }
            for i in range(5)
        ]
        
        results = []
        
        def process_document(doc):
            with patch.object(self.processor.embedder, 'get_embedding') as mock_embed:
                mock_embed.return_value = [0.1, 0.2, 0.3]
                result = self.processor.process(doc)
                results.append(result)
        
        # Process documents concurrently
        threads = []
        for doc in documents:
            thread = threading.Thread(target=process_document, args=(doc,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify all documents were processed
        self.assertEqual(len(results), 5)
        doc_ids = [r["id"] for r in results]
        self.assertEqual(len(set(doc_ids)), 5)  # All unique IDs

    def test_large_document_processing(self):
        """Test processing of large documents."""
        large_content = "This is a sentence. " * 1000  # Large document
        large_document = {
            "content": large_content,
            "id": "large_doc_test"
        }
        
        with patch.object(self.processor.embedder, 'get_embedding') as mock_embed:
            mock_embed.return_value = [0.1] * 768  # Standard embedding size
            
            result = self.processor.process(large_document)
        
        # Should handle large documents successfully
        self.assertEqual(result["id"], "large_doc_test")
        self.assertIn("chenkings", result)
        
        # Should have processed checks
        self.assertGreater(len(result["chenkings"]), 0)


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.WARNING)
    
    # Create test suite
    unittest.main(verbosity=2)
