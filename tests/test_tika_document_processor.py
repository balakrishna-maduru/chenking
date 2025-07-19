#!/usr/bin/env python3
"""
Tests for Tika Document Processor

This module tests the TikaDocumentProcessor class which processes documents
extracted by Apache Tika through the Chenking pipeline.
"""

import pytest
import json
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chenking.tika_document_processor import TikaDocumentProcessor, main


class TestTikaDocumentProcessor:
    """Test cases for TikaDocumentProcessor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock the Processor to avoid API calls during tests
        with patch('chenking.tika_document_processor.Processor') as mock_processor:
            mock_processor.return_value.process.return_value = {
                "id": "test",
                "status": "processed",
                "chenks": [{"content": "test content", "embedding": [0.1, 0.2]}]
            }
            self.processor = TikaDocumentProcessor()
            self.mock_processor = mock_processor
    
    def test_initialization(self):
        """Test TikaDocumentProcessor initialization."""
        assert self.processor is not None
        assert hasattr(self.processor, 'processor')
        assert hasattr(self.processor, 'logger')
    
    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        with patch('chenking.tika_document_processor.Processor') as mock_processor:
            processor = TikaDocumentProcessor(
                api_url="http://custom:8003/embedding",
                min_word_count=5,
                max_word_count=100
            )
            mock_processor.assert_called_with(
                api_url="http://custom:8003/embedding",
                min_word_count=5,
                max_word_count=100
            )
    
    def test_validate_tika_document_valid(self):
        """Test validation of valid Tika document."""
        valid_doc = {
            "id": "test_doc",
            "content": "This is test content",
            "metadata": {"file_name": "test.pdf"}
        }
        
        assert self.processor.validate_tika_document(valid_doc) is True
    
    def test_validate_tika_document_missing_id(self):
        """Test validation fails when id is missing."""
        invalid_doc = {
            "content": "This is test content",
            "metadata": {"file_name": "test.pdf"}
        }
        
        assert self.processor.validate_tika_document(invalid_doc) is False
    
    def test_validate_tika_document_missing_content(self):
        """Test validation fails when content is missing."""
        invalid_doc = {
            "id": "test_doc",
            "metadata": {"file_name": "test.pdf"}
        }
        
        assert self.processor.validate_tika_document(invalid_doc) is False
    
    def test_validate_tika_document_invalid_content_type(self):
        """Test validation fails when content is not string."""
        invalid_doc = {
            "id": "test_doc",
            "content": ["not", "a", "string"],
            "metadata": {"file_name": "test.pdf"}
        }
        
        assert self.processor.validate_tika_document(invalid_doc) is False
    
    def test_validate_tika_document_empty_content(self):
        """Test validation with empty content (should warn but pass)."""
        doc_with_empty_content = {
            "id": "test_doc",
            "content": "   ",  # Only whitespace
            "metadata": {"file_name": "test.pdf"}
        }
        
        # Should pass validation but log warning
        assert self.processor.validate_tika_document(doc_with_empty_content) is True
    
    def test_enhance_document_metadata_basic(self):
        """Test basic metadata enhancement."""
        doc = {
            "id": "test_doc",
            "content": "Test content",
            "metadata": {
                "file_name": "test.pdf",
                "title": "Test Document",
                "author": "Test Author"
            }
        }
        
        enhanced = self.processor.enhance_document_metadata(doc)
        
        assert enhanced["id"] == "test_doc"
        assert enhanced["content"] == "Test content"
        assert enhanced["format"] == "pdf"
        assert enhanced["title"] == "Test Document"
        assert enhanced["author"] == "Test Author"
        assert "tika_processed_at" in enhanced
        assert enhanced["tika_metadata"] == doc["metadata"]
    
    def test_enhance_document_metadata_missing_fields(self):
        """Test metadata enhancement with missing fields."""
        doc = {
            "id": "test_doc",
            "content": "Test content",
            "metadata": {}
        }
        
        enhanced = self.processor.enhance_document_metadata(doc)
        
        assert enhanced["format"] == "unknown"
        assert enhanced["title"] == "Document test_doc"
        assert enhanced["author"] == "Unknown"
        assert "tika_processed_at" in enhanced
    
    def test_enhance_document_metadata_alternative_fields(self):
        """Test metadata enhancement with alternative field names."""
        doc = {
            "id": "test_doc",
            "content": "Test content",
            "metadata": {
                "resourceName": "document.docx",
                "dc:title": "DC Title",
                "dc:creator": "DC Creator",
                "Content-Type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            }
        }
        
        enhanced = self.processor.enhance_document_metadata(doc)
        
        assert enhanced["format"] == "docx"
        assert enhanced["title"] == "DC Title"
        assert enhanced["author"] == "DC Creator"
    
    def test_enhance_document_metadata_content_type_fallback(self):
        """Test format extraction from Content-Type when no filename."""
        doc = {
            "id": "test_doc",
            "content": "Test content",
            "metadata": {
                "Content-Type": "application/pdf"
            }
        }
        
        enhanced = self.processor.enhance_document_metadata(doc)
        assert enhanced["format"] == "application/pdf"
    
    def test_process_document_success(self):
        """Test successful document processing."""
        doc = {
            "id": "test_doc",
            "content": "This is a test document with enough content",
            "metadata": {"file_name": "test.pdf"}
        }
        
        result = self.processor.process_document(doc)
        
        assert result["id"] == "test_doc"
        assert result["status"] == "processed"
        assert result["source"] == "tika"
        assert "original_metadata" in result
        assert "chenks" in result
    
    def test_process_document_validation_failure(self):
        """Test document processing with validation failure."""
        invalid_doc = {
            "content": "Test content"  # Missing ID
        }
        
        result = self.processor.process_document(invalid_doc)
        
        assert result["status"] == "error"
        assert "Document validation failed" in result["error"]
        assert "original_document" in result
    
    def test_process_document_processing_exception(self):
        """Test document processing with exception during processing."""
        doc = {
            "id": "test_doc",
            "content": "Test content",
            "metadata": {}
        }
        
        # Mock processor to raise an exception
        self.processor.processor.process.side_effect = Exception("Processing error")
        
        result = self.processor.process_document(doc)
        
        assert result["status"] == "error"
        assert "Processing error" in result["error"]
        assert result["id"] == "test_doc"
    
    def test_process_batch_success(self):
        """Test successful batch processing."""
        docs = [
            {
                "id": "doc1",
                "content": "First document content",
                "metadata": {"file_name": "doc1.pdf"}
            },
            {
                "id": "doc2", 
                "content": "Second document content",
                "metadata": {"file_name": "doc2.pdf"}
            }
        ]
        
        results = self.processor.process_batch(docs)
        
        assert len(results) == 2
        assert all(r["status"] == "processed" for r in results)
        assert results[0]["id"] == "doc1"
        assert results[1]["id"] == "doc2"
    
    def test_process_batch_mixed_results(self):
        """Test batch processing with mixed success/failure."""
        docs = [
            {
                "id": "doc1",
                "content": "Valid document",
                "metadata": {}
            },
            {
                "content": "Invalid document"  # Missing ID
            }
        ]
        
        results = self.processor.process_batch(docs)
        
        assert len(results) == 2
        assert results[0]["status"] == "processed"
        assert results[1]["status"] == "error"
    
    def test_load_json_file_valid(self):
        """Test loading valid JSON file."""
        test_data = {"id": "test", "content": "test content"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            loaded_data = self.processor.load_json_file(temp_path)
            assert loaded_data == test_data
        finally:
            os.unlink(temp_path)
    
    def test_load_json_file_invalid_json(self):
        """Test loading invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                self.processor.load_json_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_load_json_file_not_found(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.processor.load_json_file("/non/existent/file.json")
    
    def test_save_results_success(self):
        """Test saving results to file."""
        test_results = {"status": "completed", "documents": []}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.processor.save_results(test_results, temp_path)
            
            # Verify file was created and contains correct data
            with open(temp_path, 'r') as f:
                saved_data = json.load(f)
            assert saved_data == test_results
        finally:
            os.unlink(temp_path)
    
    def test_save_results_failure(self):
        """Test saving results to invalid path."""
        test_results = {"status": "completed"}
        
        with pytest.raises(Exception):
            self.processor.save_results(test_results, "/invalid/path/file.json")


class TestTikaDocumentProcessorMain:
    """Test cases for the main function and CLI interface."""
    
    def test_main_stdin_processing(self):
        """Test processing document from stdin."""
        test_input = {
            "id": "stdin_test",
            "content": "Test content from stdin",
            "metadata": {}
        }
        
        with patch('sys.stdin', StringIO(json.dumps(test_input))), \
             patch('chenking.tika_document_processor.Processor') as mock_processor, \
             patch('sys.argv', ['tika_document_processor.py', '--stdin']), \
             patch('builtins.print') as mock_print:
            
            mock_processor.return_value.process.return_value = {
                "id": "stdin_test",
                "status": "processed"
            }
            
            main()
            
            # Verify print was called with processed result
            mock_print.assert_called_once()
            printed_output = mock_print.call_args[0][0]
            result = json.loads(printed_output)
            assert result["id"] == "stdin_test"
            assert result["status"] == "processed"
    
    def test_main_file_processing(self):
        """Test processing document from file."""
        test_data = {
            "id": "file_test",
            "content": "Test content from file",
            "metadata": {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            input_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name
        
        try:
            with patch('chenking.tika_document_processor.Processor') as mock_processor, \
                 patch('sys.argv', ['tika_document_processor.py', '--input', input_path, '--output', output_path]):
                
                mock_processor.return_value.process.return_value = {
                    "id": "file_test",
                    "status": "processed"
                }
                
                main()
                
                # Verify output file was created
                with open(output_path, 'r') as f:
                    result = json.load(f)
                assert result["id"] == "file_test"
                assert result["status"] == "processed"
        
        finally:
            os.unlink(input_path)
            os.unlink(output_path)
    
    def test_main_batch_processing(self):
        """Test batch processing from file."""
        test_data = [
            {"id": "doc1", "content": "First document", "metadata": {}},
            {"id": "doc2", "content": "Second document", "metadata": {}}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            input_path = f.name
        
        try:
            with patch('chenking.tika_document_processor.Processor') as mock_processor, \
                 patch('sys.argv', ['tika_document_processor.py', '--input', input_path, '--batch']), \
                 patch('builtins.print') as mock_print:
                
                mock_processor.return_value.process.return_value = {
                    "status": "processed"
                }
                
                main()
                
                # Verify batch processing occurred
                mock_print.assert_called_once()
        finally:
            os.unlink(input_path)
    
    def test_main_directory_processing(self):
        """Test processing directory of JSON files."""
        test_docs = [
            {"id": "doc1", "content": "First document", "metadata": {}},
            {"id": "doc2", "content": "Second document", "metadata": {}}
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test JSON files
            for i, doc in enumerate(test_docs):
                file_path = os.path.join(temp_dir, f"doc{i+1}.json")
                with open(file_path, 'w') as f:
                    json.dump(doc, f)
            
            with patch('chenking.tika_document_processor.Processor') as mock_processor, \
                 patch('sys.argv', ['tika_document_processor.py', '--input', temp_dir, '--batch']), \
                 patch('builtins.print') as mock_print:
                
                mock_processor.return_value.process.return_value = {
                    "status": "processed"
                }
                
                main()
                
                # Verify processing occurred
                mock_print.assert_called_once()
    
    def test_main_keyboard_interrupt(self):
        """Test handling of keyboard interrupt."""
        with patch('chenking.tika_document_processor.Processor') as mock_processor, \
             patch('sys.argv', ['tika_document_processor.py', '--stdin']), \
             patch('sys.stdin', StringIO('{"id":"test","content":"test"}')), \
             patch('json.load', side_effect=KeyboardInterrupt()), \
             pytest.raises(SystemExit) as exc_info:
            
            main()
            
            assert exc_info.value.code == 1
    
    def test_main_processing_exception(self):
        """Test handling of processing exceptions."""
        with patch('chenking.tika_document_processor.Processor') as mock_processor, \
             patch('sys.argv', ['tika_document_processor.py', '--stdin']), \
             patch('sys.stdin', StringIO('{"id":"test","content":"test"}')), \
             patch('json.load', side_effect=Exception("Processing failed")), \
             pytest.raises(SystemExit) as exc_info:
            
            main()
            
            assert exc_info.value.code == 1
    
    def test_main_with_custom_arguments(self):
        """Test main function with custom configuration arguments."""
        test_input = {"id": "test", "content": "test content", "metadata": {}}
        
        args = [
            'tika_document_processor.py', '--stdin',
            '--min-word-count', '5',
            '--max-word-count', '100',
            '--disable-page-splitting',
            '--verbose',
            '--api-url', 'http://custom:8003/api'
        ]
        
        with patch('sys.stdin', StringIO(json.dumps(test_input))), \
             patch('chenking.tika_document_processor.Processor') as mock_processor, \
             patch('sys.argv', args), \
             patch('builtins.print'):
            
            mock_processor.return_value.process.return_value = {
                "id": "test",
                "status": "processed"
            }
            
            main()
            
            # Verify processor was initialized with custom args
            call_args = mock_processor.call_args
            assert call_args[1]['min_word_count'] == 5
            assert call_args[1]['max_word_count'] == 100
            assert call_args[1]['enable_page_splitting'] is False
            assert call_args[0][0] == 'http://custom:8003/api'  # api_url


if __name__ == "__main__":
    pytest.main([__file__])
