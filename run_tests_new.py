#!/usr/bin/env python3
"""
Test runner and demonstration script for Chenking package.
Focuses on DocumentProcessor as the main class.
"""

import sys
import os
import unittest

# Add the project directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def run_tests():
    """Run all tests for Chenking package."""
    print("=" * 60)
    print("Running Chenking Test Suite")
    print("=" * 60)
    
    # Discover and run tests from tests directory
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), 'tests')
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 60)
    
    return result.wasSuccessful()

def demonstrate_usage():
    """Demonstrate DocumentProcessor usage with examples."""
    print("\n" + "=" * 60)
    print("DocumentProcessor Usage Demonstration")
    print("=" * 60)
    
    from chenking.document_processor import DocumentProcessor
    from unittest.mock import patch, Mock
    
    # Mock the embedding API for demonstration
    with patch('chenking.embedding_client.requests.post') as mock_post:
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
            "vector": [0.6, 0.7, 0.8, 0.9, 1.0]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        # Example 1: Basic usage with default settings
        print("\n1. Basic DocumentProcessor Usage:")
        processor = DocumentProcessor("https://api.example.com/embeddings")
        
        sample_doc = {
            "id": "demo_001",
            "content": "This is a comprehensive demonstration document. It contains multiple sentences and demonstrates various features of the document processor. The content is well-structured and should pass most validation checks.",
            "title": "Demo Document",
            "author": "Demo Author",
            "format": "txt",
            "metadata": {"category": "demonstration", "version": "1.0"}
        }
        
        result = processor.process(sample_doc)
        print(f"Document processed successfully: {result['id']}")
        print(f"Checks completed: {len(result['chenkings'])}")
        print(f"Processing time: {result['processing_info']['total_processing_time']:.4f}s")
        
        successful_checks = result['processing_info']['successful_checks']
        total_checks = result['processing_info']['checks_completed']
        print(f"Successful checks: {successful_checks}/{total_checks}")
        
        # Example 2: Custom configuration
        print("\n2. Custom Configuration:")
        custom_processor = DocumentProcessor(
            "https://api.example.com/embeddings",
            min_word_count=5,
            max_word_count=100,
            enable_detailed_logging=True,
            supported_formats=["txt", "md"],
            required_fields=["content", "title"]
        )
        
        short_doc = {
            "id": "short_001",
            "content": "Brief content for testing custom limits.",
            "title": "Short Document"
        }
        
        result = custom_processor.process(short_doc)
        print(f"Custom processing completed for: {result['id']}")
        
        # Example 3: Batch processing
        print("\n3. Batch Processing:")
        documents = [
            {"id": "batch_001", "content": "First document in batch processing demo."},
            {"id": "batch_002", "content": "Second document with different content for testing."},
            {"id": "batch_003", "content": "Third document to complete the batch demonstration."}
        ]
        
        batch_results = processor.process_batch(documents)
        print(f"Batch processed: {len(batch_results)} documents")
        
        # Generate statistics
        stats = processor.get_processing_stats(batch_results)
        print(f"Success rate: {stats['success_rate']:.2%}")
        print(f"Average processing time: {stats['average_processing_time']:.4f}s")
        
        # Example 4: Document with validation issues
        print("\n4. Document with Validation Issues:")
        problematic_doc = {
            "id": "problem_001",
            "content": "Too short",  # Will fail word count validation
            "format": "pdf"  # Unsupported format
            # Missing other fields
        }
        
        try:
            result = processor.process(problematic_doc)
            print("Document processed (may have validation issues)")
            
            # Show specific issues
            for check_name, check_result in result['chenkings'].items():
                if check_result['status'] == 'success':
                    # Check for validation failures within successful checks
                    data = check_result['data']
                    if 'is_word_count_valid' in data and not data['is_word_count_valid']:
                        print(f"  - Word count issue in {check_name}: {data.get('word_count', 0)} words")
                    if 'is_format_supported' in data and not data['is_format_supported']:
                        print(f"  - Format issue in {check_name}: {data.get('format', 'unknown')} not supported")
        
        except ValueError as e:
            print(f"Document processing failed: {e}")
        
        # Example 5: Integration features
        print("\n5. Integration Features:")
        integration_doc = {
            "id": "integration_001",
            "content": "This document demonstrates the integration capabilities of the DocumentProcessor. It combines document validation with embedding generation to provide comprehensive analysis.",
            "title": "Integration Demo",
            "author": "System",
            "metadata": {
                "source": "integration_test",
                "priority": "high",
                "tags": ["demo", "integration", "analysis"]
            }
        }
        
        result = processor.process(integration_doc)
        
        # Show integration results
        print(f"Integration processing for: {result['id']}")
        print("Check results with embeddings:")
        for check_name, check_result in result['chenkings'].items():
            if check_result['status'] == 'success':
                has_embedding = check_result['embedding'] is not None
                has_vector = check_result['vector'] is not None
                print(f"  {check_name}: ✓ (embedding: {'✓' if has_embedding else '✗'}, vector: {'✓' if has_vector else '✗'})")
            else:
                print(f"  {check_name}: ✗ ({check_result.get('error', 'unknown error')})")
    
    print("\nDemonstration completed!")

def demonstrate_embedding_client():
    """Demonstrate EmbeddingClient functionality."""
    print("\n" + "=" * 60)
    print("EmbeddingClient Demonstration")
    print("=" * 60)
    
    from chenking.embedding_client import EmbeddingClient
    from unittest.mock import patch, Mock
    
    with patch('chenking.embedding_client.requests.post') as mock_post, \
         patch('chenking.embedding_client.requests.get') as mock_get:
        
        # Mock embedding response
        mock_response = Mock()
        mock_response.json.return_value = {
            "embedding": [0.1, 0.2, 0.3],
            "vector": [0.4, 0.5, 0.6]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        # Mock health check
        mock_health_response = Mock()
        mock_health_response.status_code = 200
        mock_get.return_value = mock_health_response
        
        print("\n1. EmbeddingClient Basic Usage:")
        client = EmbeddingClient("https://api.example.com/embeddings")
        
        # Health check
        is_healthy = client.health_check()
        print(f"API Health Check: {'✓ Healthy' if is_healthy else '✗ Unhealthy'}")
        
        # Generate embedding
        check_data = {"word_count": 25, "has_content": True, "char_count": 150}
        result = client.get_embedding(check_data)
        
        print(f"Embedding Status: {result['status']}")
        if result['status'] == 'success':
            print(f"Embedding Length: {len(result['embedding'])}")
            print(f"Vector Length: {len(result['vector'])}")
            print(f"Request Time: {result['request_time']:.4f}s")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        demonstrate_usage()
    elif len(sys.argv) > 1 and sys.argv[1] == '--demo-embedding':
        demonstrate_embedding_client()
    elif len(sys.argv) > 1 and sys.argv[1] == '--demo-all':
        demonstrate_usage()
        demonstrate_embedding_client()
    else:
        success = run_tests()
        if not success:
            sys.exit(1)
        
        # Optionally run demo after successful tests
        if len(sys.argv) > 1 and sys.argv[1] == '--with-demo':
            demonstrate_usage()
            demonstrate_embedding_client()
