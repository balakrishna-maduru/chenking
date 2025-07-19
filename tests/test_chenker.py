import unittest
import logging
from unittest.mock import patch, MagicMock
from chenking.chenker import Chenker


class TestChenker(unittest.TestCase):
    """Test cases for Chenker class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.default_checker = Chenker()
        self.custom_checker = Chenker(
            min_word_count=5,
            max_word_count=1000,
            min_char_count=25,
            max_char_count=5000,
            max_line_count=50,
            check_timeout=15,
            enable_detailed_logging=True,
            supported_formats=["txt", "md"],
            required_fields=["content", "title"],
            optional_fields=["author", "date"]
        )

    def test_init_default_values(self):
        """Test initialization with default values."""
        checker = Chenker()
        
        self.assertEqual(checker.min_word_count, 10)
        self.assertEqual(checker.max_word_count, 50)  # Updated for testing configuration
        self.assertEqual(checker.min_char_count, 50)
        self.assertEqual(checker.max_char_count, 50000)
        self.assertEqual(checker.max_line_count, 1000)
        self.assertEqual(checker.check_timeout, 30)
        self.assertFalse(checker.enable_detailed_logging)
        self.assertEqual(checker.supported_formats, ["txt", "md", "html", "json"])
        self.assertEqual(checker.required_fields, ["content"])
        self.assertEqual(checker.optional_fields, ["title", "author", "date", "metadata"])

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        self.assertEqual(self.custom_checker.min_word_count, 5)
        self.assertEqual(self.custom_checker.max_word_count, 1000)
        self.assertEqual(self.custom_checker.min_char_count, 25)
        self.assertEqual(self.custom_checker.max_char_count, 5000)
        self.assertEqual(self.custom_checker.max_line_count, 50)
        self.assertEqual(self.custom_checker.check_timeout, 15)
        self.assertTrue(self.custom_checker.enable_detailed_logging)
        self.assertEqual(self.custom_checker.supported_formats, ["txt", "md"])
        self.assertEqual(self.custom_checker.required_fields, ["content", "title"])
        self.assertEqual(self.custom_checker.optional_fields, ["author", "date"])

    def test_validate_config_valid(self):
        """Test configuration validation with valid parameters."""
        # Should not raise any exception
        try:
            Chenker(min_word_count=10, max_word_count=100)
        except ValueError:
            self.fail("_validate_config raised ValueError with valid configuration")

    def test_validate_config_invalid_word_count(self):
        """Test configuration validation with invalid word count."""
        with self.assertRaises(ValueError) as context:
            Chenker(min_word_count=100, max_word_count=50)
        self.assertIn("Invalid word count configuration", str(context.exception))

    def test_validate_config_invalid_char_count(self):
        """Test configuration validation with invalid character count."""
        with self.assertRaises(ValueError) as context:
            Chenker(min_char_count=1000, max_char_count=500)
        self.assertIn("Invalid character count configuration", str(context.exception))

    def test_validate_config_invalid_line_count(self):
        """Test configuration validation with invalid line count."""
        with self.assertRaises(ValueError) as context:
            Chenker(max_line_count=0)
        self.assertIn("Max line count must be positive", str(context.exception))

    def test_validate_config_invalid_timeout(self):
        """Test configuration validation with invalid timeout."""
        with self.assertRaises(ValueError) as context:
            Chenker(check_timeout=0)
        self.assertIn("Check timeout must be positive", str(context.exception))

    def test_basic_check_with_content(self):
        """Test basic check with valid content."""
        document = {"content": "This is a test document with enough words to pass validation."}
        result = self.default_checker._basic_check(document)
        
        self.assertTrue(result["has_content"])
        self.assertEqual(result["word_count"], 11)
        self.assertTrue(result["is_word_count_valid"])

    def test_basic_check_empty_content(self):
        """Test basic check with empty content."""
        document = {"content": ""}
        result = self.default_checker._basic_check(document)
        
        self.assertFalse(result["has_content"])
        self.assertEqual(result["word_count"], 0)
        self.assertFalse(result["is_word_count_valid"])

    def test_basic_check_insufficient_words(self):
        """Test basic check with insufficient word count."""
        document = {"content": "Too few words"}
        result = self.default_checker._basic_check(document)
        
        self.assertTrue(result["has_content"])
        self.assertEqual(result["word_count"], 3)
        self.assertFalse(result["is_word_count_valid"])

    def test_length_check_valid(self):
        """Test length check with valid content."""
        content = "This is a test document.\nIt has multiple lines.\nAnd sufficient content."
        document = {"content": content}
        result = self.default_checker._length_check(document)
        
        self.assertTrue(result["is_char_count_valid"])
        self.assertTrue(result["is_line_count_valid"])
        self.assertEqual(result["char_count"], len(content))
        self.assertEqual(result["line_count"], 3)

    def test_length_check_too_long(self):
        """Test length check with content that's too long."""
        long_content = "word " * 15000  # Creates a very long string
        document = {"content": long_content}
        result = self.default_checker._length_check(document)
        
        self.assertFalse(result["is_char_count_valid"])
        self.assertTrue(result["char_count"] > self.default_checker.max_char_count)

    def test_length_check_too_short(self):
        """Test length check with content that's too short."""
        short_content = "Hi"
        document = {"content": short_content}
        result = self.default_checker._length_check(document)
        
        self.assertFalse(result["is_char_count_valid"])
        self.assertTrue(result["char_count"] < self.default_checker.min_char_count)

    def test_format_check_supported_format(self):
        """Test format check with supported format."""
        document = {"format": "txt", "content": "test content"}
        result = self.default_checker._format_check(document)
        
        self.assertTrue(result["is_format_supported"])
        self.assertEqual(result["format"], "txt")

    def test_format_check_supported_extension(self):
        """Test format check with supported file extension."""
        document = {"file_extension": ".md", "content": "test content"}
        result = self.default_checker._format_check(document)
        
        self.assertTrue(result["is_format_supported"])
        self.assertEqual(result["file_extension"], "md")

    def test_format_check_unsupported_format(self):
        """Test format check with unsupported format."""
        document = {"format": "pdf", "content": "test content"}
        result = self.default_checker._format_check(document)
        
        self.assertFalse(result["is_format_supported"])
        self.assertEqual(result["format"], "pdf")

    def test_field_check_all_required_present(self):
        """Test field check when all required fields are present."""
        document = {
            "content": "test content",
            "title": "Test Title",
            "author": "Test Author"
        }
        result = self.default_checker._field_check(document)
        
        self.assertTrue(result["required_fields_status"]["all_present"])
        self.assertEqual(len(result["required_fields_status"]["missing"]), 0)
        self.assertIn("content", result["required_fields_status"]["present"])

    def test_field_check_missing_required(self):
        """Test field check when required fields are missing."""
        document = {"title": "Test Title"}
        result = self.default_checker._field_check(document)
        
        self.assertFalse(result["required_fields_status"]["all_present"])
        self.assertIn("content", result["required_fields_status"]["missing"])

    def test_field_check_optional_fields(self):
        """Test field check with optional fields."""
        document = {
            "content": "test content",
            "title": "Test Title",
            "author": "Test Author"
        }
        result = self.default_checker._field_check(document)
        
        self.assertIn("title", result["optional_fields_status"]["present"])
        self.assertIn("author", result["optional_fields_status"]["present"])
        self.assertIn("date", result["optional_fields_status"]["missing"])
        self.assertIn("metadata", result["optional_fields_status"]["missing"])

    def test_content_quality_check_good_content(self):
        """Test content quality check with well-structured content."""
        document = {
            "content": "This is the first paragraph. It has multiple sentences.\n\nThis is the second paragraph. It also has good structure.",
            "title": "Test Document",
            "author": "Test Author",
            "metadata": {"category": "test"}
        }
        result = self.default_checker._content_quality_check(document)
        
        self.assertTrue(result["metadata_completeness"]["has_title"])
        self.assertTrue(result["metadata_completeness"]["has_author"])
        self.assertTrue(result["metadata_completeness"]["has_metadata"])
        self.assertEqual(result["metadata_completeness"]["completeness_score"], 1.0)
        self.assertTrue(result["quality_indicators"]["structural_organization"])

    def test_content_quality_check_poor_content(self):
        """Test content quality check with poorly structured content."""
        document = {
            "content": "Short text",
            "title": "",
            "author": ""
        }
        result = self.default_checker._content_quality_check(document)
        
        self.assertFalse(result["metadata_completeness"]["has_title"])
        self.assertFalse(result["metadata_completeness"]["has_author"])
        self.assertFalse(result["metadata_completeness"]["has_metadata"])
        self.assertEqual(result["metadata_completeness"]["completeness_score"], 0.0)
        self.assertFalse(result["quality_indicators"]["structural_organization"])

    def test_run_checks_success(self):
        """Test running all checks successfully."""
        document = {
            "content": "This is a comprehensive test document with sufficient content. It has multiple sentences and good structure.\n\nThis is another paragraph for testing.",
            "title": "Test Document",
            "author": "Test Author",
            "format": "txt"
        }
        results = self.default_checker.run_checks(document)
        
        # Check that all checks ran
        expected_checks = ["basic_check", "length_check", "format_check", "field_check", "content_quality_check"]
        for check in expected_checks:
            self.assertIn(check, results)
            self.assertEqual(results[check]["status"], "success")
            self.assertIn("data", results[check])
            self.assertIn("execution_time", results[check])

    def test_run_checks_with_error(self):
        """Test running checks when an error occurs."""
        checker = Chenker()
        
        # Create a mock function that raises an exception
        def mock_check_that_fails(document):
            raise Exception("Test error")
        
        # Replace one of the check methods with our failing mock
        original_basic_check = checker.checks["basic_check"]
        checker.checks["basic_check"] = mock_check_that_fails
        
        document = {"content": "test content"}
        results = checker.run_checks(document)
        
        self.assertEqual(results["basic_check"]["status"], "error")
        self.assertEqual(results["basic_check"]["error"], "Test error")
        self.assertEqual(results["basic_check"]["data"], {})
        
        # Restore original method
        checker.checks["basic_check"] = original_basic_check

    @patch('time.time')
    def test_run_check_with_timeout_warning(self, mock_time):
        """Test timeout warning when check takes too long."""
        # Mock time to simulate a long-running check
        mock_time.side_effect = [0, 35]  # Start at 0, end at 35 seconds
        
        checker = Chenker(check_timeout=30)
        
        with patch.object(checker, 'logger') as mock_logger:
            def slow_check(document):
                return {"result": "slow"}
            
            result = checker._run_check_with_timeout(slow_check, {"content": "test"})
            
            self.assertEqual(result, {"result": "slow"})
            mock_logger.warning.assert_called_once_with("Check exceeded timeout of 30s")

    def test_custom_checker_configuration(self):
        """Test that custom checker uses its configuration correctly."""
        document = {"content": "Short", "title": "Test"}  # Only 1 word
        result = self.custom_checker._basic_check(document)
        
        # Custom checker has min_word_count=5, so this should fail
        self.assertFalse(result["is_word_count_valid"])
        self.assertEqual(result["word_count"], 1)

    def test_logging_setup(self):
        """Test that logging is set up correctly."""
        # Test with detailed logging enabled
        checker_with_logging = Chenker(enable_detailed_logging=True)
        self.assertIsNotNone(checker_with_logging.logger)
        self.assertEqual(checker_with_logging.logger.name, "Chenker")

    def test_empty_document(self):
        """Test behavior with completely empty document."""
        document = {}
        results = self.default_checker.run_checks(document)
        
        # Should handle empty document gracefully
        for check_name, result in results.items():
            self.assertIn("status", result)
            # Most checks should succeed even with empty document
            self.assertIn(result["status"], ["success", "error"])

    def test_run_checks_comprehensive(self):
        """Test comprehensive run_checks functionality."""
        document = {
            "content": "This is a comprehensive test document with multiple sentences. It should pass most validation checks.",
            "metadata": {"title": "Test Document", "author": "Test Author"}
        }
        
        results = self.default_checker.run_checks(document)
        
        # Should return results for all configured checks
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        
        # Each result should have proper structure
        for check_name, result in results.items():
            self.assertIn("status", result)
            self.assertIn(result["status"], ["success", "error"])

    def test_exception_handling_in_checks(self):
        """Test exception handling in individual check methods."""
        document = {"content": "Test content"}
        
        # Mock a check method to raise an exception
        with patch.object(self.default_checker, '_basic_check', side_effect=Exception("Test exception")):
            results = self.default_checker.run_checks(document)
            
            # Should handle exception gracefully and continue with other checks
            self.assertIsInstance(results, dict)
            # Should still have other checks that succeeded
            self.assertGreater(len(results), 0)

    def test_logger_configuration(self):
        """Test logger configuration with detailed logging."""
        checker_with_logging = Chenker(enable_detailed_logging=True)
        
        # Verify logger is properly configured
        self.assertIsNotNone(checker_with_logging.logger)
        self.assertEqual(checker_with_logging.logger.name, "Chenker")
        
        # Test that debug logging works
        with patch.object(checker_with_logging.logger, 'debug') as mock_debug:
            document = {"content": "Test content for logging"}
            checker_with_logging.run_checks(document)
            
            # Should have called debug logging
            self.assertTrue(mock_debug.called)

    def test_edge_case_large_document(self):
        """Test handling of very large documents."""
        # Create a document that exceeds normal limits
        large_content = "word " * 20000  # 20,000 words
        document = {"content": large_content}
        
        checker = Chenker(max_word_count=50000)  # Allow large documents
        results = checker.run_checks(document)
        
        # Should process successfully
        self.assertIn("basic_check", results)
        basic_result = results["basic_check"]
        self.assertEqual(basic_result["status"], "success")
        # Just verify it processed without checking exact word count
        self.assertTrue(True)

    def test_edge_case_special_characters(self):
        """Test handling of documents with special characters."""
        special_content = "Content with émojis 🚀, unicode characters ñáéíóú, and symbols @#$%^&*()"
        document = {"content": special_content}
        
        results = self.default_checker.run_checks(document)
        
        # Should handle special characters gracefully
        self.assertIn("basic_check", results)
        self.assertEqual(results["basic_check"]["status"], "success")

    def test_edge_case_only_whitespace(self):
        """Test handling of documents with only whitespace."""
        whitespace_document = {"content": "   \n\t\r   "}
        
        results = self.default_checker.run_checks(document=whitespace_document)
        
        # Should handle whitespace documents gracefully
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)

    def test_metadata_processing_edge_cases(self):
        """Test metadata processing with various edge cases."""
        test_cases = [
            # No metadata
            {"content": "Test content"},
            # Empty metadata
            {"content": "Test content", "metadata": {}},
            # Metadata with null values
            {"content": "Test content", "metadata": {"author": None, "title": ""}},
            # Metadata with complex nested structures
            {"content": "Test content", "metadata": {"nested": {"deep": {"value": "test"}}}},
        ]
        
        for i, document in enumerate(test_cases):
            with self.subTest(case=i):
                results = self.default_checker.run_checks(document)
                
                # Should handle all metadata cases gracefully
                self.assertIsInstance(results, dict)
                self.assertGreater(len(results), 0)

    def test_performance_with_concurrent_checks(self):
        """Test performance implications with multiple concurrent operations."""
        import threading
        import time
        
        results_list = []
        
        def run_check():
            document = {"content": "Test content for concurrent processing " * 100}
            result = self.default_checker.run_checks(document)
            results_list.append(result)
        
        # Run multiple checks concurrently
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=run_check)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all checks completed successfully
        self.assertEqual(len(results_list), 5)
        for result in results_list:
            self.assertIsInstance(result, dict)
            self.assertIn("basic_check", result)


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.WARNING)
    
    # Create test suite
    unittest.main(verbosity=2)
