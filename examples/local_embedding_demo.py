#!/usr/bin/env python3
"""
Local Embedding Configuration for Chenking

This shows how to use Chenking with a local embedding service
instead of external APIs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chenking.processor import Processor
from chenking.embedding_client import EmbeddingClient
import requests
import time

class LocalEmbeddingDemo:
    """Demonstrate Chenking with local embedding service."""
    
    def __init__(self):
        # Local embedding service endpoints
        self.local_endpoints = {
            "custom": "http://localhost:8000/chenking/embedding",
            "openai_compatible": "http://localhost:8000/embeddings",
            "huggingface": "http://localhost:8080/embed",
            "ollama": "http://localhost:11434/api/embeddings"
        }
        
    def check_service_health(self, service_name: str, health_url: str) -> bool:
        """Check if a local embedding service is running."""
        try:
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def find_available_service(self) -> str:
        """Find the first available embedding service."""
        service_checks = {
            "custom": "http://localhost:8000/health",
            "huggingface": "http://localhost:8080/health",
            "ollama": "http://localhost:11434/api/tags"
        }
        
        print("🔍 Checking for available embedding services...")
        
        for service, health_url in service_checks.items():
            if self.check_service_health(service, health_url):
                endpoint = self.local_endpoints[service]
                print(f"✅ Found {service} service at {endpoint}")
                return endpoint
        
        print("❌ No local embedding services found!")
        print("💡 Run './setup_local_embeddings.sh' to start a local service")
        return None
    
    def demo_with_local_embeddings(self):
        """Demonstrate Chenking with local embedding service."""
        print("🏠 Chenking Local Embedding Demo")
        print("=" * 40)
        
        # Find available service
        endpoint = self.find_available_service()
        if not endpoint:
            return
        
        # Initialize Chenking with local endpoint
        print(f"\n🔧 Initializing Chenking with local embeddings...")
        processor = Processor(endpoint)
        
        # Test documents
        test_docs = [
            {
                "id": "local_test_1",
                "title": "Local Embedding Test",
                "content": "This document is being processed with a local embedding model. It demonstrates how Chenking can work entirely offline without external API dependencies."
            },
            {
                "id": "local_test_2", 
                "title": "Performance Test",
                "content": "Local embeddings provide faster response times and complete data privacy. This is ideal for sensitive documents or environments with strict data governance requirements."
            }
        ]
        
        print(f"\n📄 Processing {len(test_docs)} documents...")
        
        for doc in test_docs:
            start_time = time.time()
            result = processor.process(doc)
            processing_time = time.time() - start_time
            
            print(f"\n📊 Document: {doc['id']}")
            print(f"   Status: {'✅ Success' if result['processing_info']['status'] != 'failed' else '❌ Failed'}")
            print(f"   Processing Time: {processing_time:.3f}s")
            print(f"   Checks Completed: {result['processing_info']['checks_completed']}")
            print(f"   Successful Checks: {result['processing_info']['successful_checks']}")
            
            # Check if embeddings were generated
            embedding_count = 0
            for check_name, check_result in result['chenkings'].items():
                if check_result.get('embedding') is not None:
                    embedding_count += 1
            
            print(f"   Embeddings Generated: {embedding_count}/5")
        
        print(f"\n🎉 Local embedding demo completed!")
        print(f"\n💡 Benefits of local embeddings:")
        print(f"   • 🔒 Complete data privacy")
        print(f"   • ⚡ Faster response times") 
        print(f"   • 💰 No API costs")
        print(f"   • 📡 Works offline")
        print(f"   • 🎛️  Full control over model")
    
    def show_integration_examples(self):
        """Show how to integrate with different local services."""
        print(f"\n🔧 Integration Examples:")
        print(f"=" * 30)
        
        examples = {
            "Custom FastAPI": {
                "endpoint": "http://localhost:8000/chenking/embedding",
                "description": "Native Chenking format, optimized for document processing"
            },
            "OpenAI Compatible": {
                "endpoint": "http://localhost:8000/embeddings", 
                "description": "Drop-in replacement for OpenAI embeddings API"
            },
            "Ollama": {
                "endpoint": "http://localhost:11434/api/embeddings",
                "description": "Local LLM with embedding capabilities"
            }
        }
        
        for name, config in examples.items():
            print(f"\n📦 {name}:")
            print(f"   Endpoint: {config['endpoint']}")
            print(f"   Description: {config['description']}")
            print(f"   Usage: processor = Processor('{config['endpoint']}')")

def main():
    """Run the local embedding demo."""
    demo = LocalEmbeddingDemo()
    demo.demo_with_local_embeddings()
    demo.show_integration_examples()

if __name__ == "__main__":
    main()
