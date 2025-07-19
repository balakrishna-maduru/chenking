#!/usr/bin/env python3
"""
Test the integrated Chenking system with the local embedding API
"""

import sys
import logging
from chenking import Processor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_integration():
    """Test the complete Chenking integration with local embedding API"""
    
    # Sample document to test
    test_document = {
        "id": "test_doc_1",
        "content": "This is a sample document for testing the Chenking integration with our new local embedding API. It contains enough content to pass validation checks.",
        "metadata": {
            "title": "Test Document",
            "author": "Test User",
            "type": "integration_test"
        }
    }
    
    print("🔧 Initializing Chenking Processor...")
    
    try:
        # Initialize processor with default local API URL
        processor = Processor()
        
        print("✅ Processor initialized successfully")
        print(f"📡 API URL: {processor.api_url}")
        
        # Test health check
        print("\n🏥 Testing embedding API health...")
        if processor.embedder.health_check():
            print("✅ Embedding API is healthy")
        else:
            print("❌ Embedding API health check failed")
            return False
        
        print("\n📄 Processing test document...")
        
        # Process the document
        result = processor.process(test_document)
        
        print("✅ Document processed successfully!")
        print(f"📊 Document ID: {result.get('id')}")
        print(f"⏱️  Total processing time: {result.get('processing_info', {}).get('total_processing_time', 0):.2f}s")
        print(f"✅ Checks completed: {result.get('processing_info', {}).get('checks_completed', 0)}")
        print(f"✅ Successful checks: {result.get('processing_info', {}).get('successful_checks', 0)}")
        
        # Show embedding info for each check
        print("\n🧠 Embedding Results:")
        for check_name, check_result in result.get('chenkings', {}).items():
            embedding = check_result.get('embedding')
            if embedding:
                print(f"  ✅ {check_name}: Generated {len(embedding)}-dimensional embedding")
            else:
                print(f"  ❌ {check_name}: No embedding generated")
                if check_result.get('embedding_error'):
                    print(f"      Error: {check_result['embedding_error']}")
        
        # Test batch processing
        print("\n📚 Testing batch processing...")
        batch_documents = [
            {**test_document, "id": "batch_doc_1", "content": "First batch document content"},
            {**test_document, "id": "batch_doc_2", "content": "Second batch document content"},
        ]
        
        batch_results = processor.process_batch(batch_documents)
        print(f"✅ Batch processing completed: {len(batch_results)} documents processed")
        
        # Get processing statistics
        stats = processor.get_processing_stats(batch_results)
        print(f"📈 Success rate: {stats.get('success_rate', 0):.1%}")
        print(f"⏱️  Average processing time: {stats.get('average_processing_time', 0):.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Chenking Integration Test")
    print("=" * 50)
    
    success = test_integration()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Integration test completed successfully!")
        sys.exit(0)
    else:
        print("💥 Integration test failed!")
        sys.exit(1)
