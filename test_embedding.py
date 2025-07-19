#!/usr/bin/env python3
import requests
import json

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:8002/health")
        print(f"Health status: {response.status_code}")
        print(f"Health response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_simple_embedding():
    """Test a simple embedding"""
    try:
        data = {
            "input": "Hello, world!",
            "model": "sentence-transformers/all-MiniLM-L6-v2"
        }
        
        response = requests.post(
            "http://localhost:8002/embeddings",
            headers={"Content-Type": "application/json"},
            json=data,
            timeout=30
        )
        
        print(f"Embedding status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Embedding dimensions: {len(result['data'][0]['embedding'])}")
            print("✅ Embedding test passed!")
            return True
        else:
            print(f"❌ Embedding test failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Embedding test error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing Chenking Embedding API...")
    
    # Test health first
    if test_health():
        print("✅ Health check passed")
        
        # Test embedding
        test_simple_embedding()
    else:
        print("❌ Health check failed")
