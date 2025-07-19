#!/usr/bin/env python3
"""
Test runner and demonstration script for Chenking package.
Runs unit tests and integration tests with the local embedding API.
"""

import sys
import os
import unittest
import subprocess

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

def run_integration_test():
    """Run the integration test with the local embedding API."""
    print("\n" + "=" * 60)
    print("Running Integration Test")
    print("=" * 60)
    
    try:
        # Run the integration test script
        result = subprocess.run([sys.executable, 'test_integration.py'], 
                              capture_output=False, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode == 0:
            print("✅ Integration test passed!")
            return True
        else:
            print("❌ Integration test failed!")
            return False
    except Exception as e:
        print(f"❌ Error running integration test: {e}")
        return False

def check_api_status():
    """Check if the local embedding API is running."""
    print("\n" + "=" * 60)
    print("Checking API Status")
    print("=" * 60)
    
    try:
        result = subprocess.run(['docker', 'ps', '--filter', 'name=chenking-custom-embeddings', '--format', 'table {{.Names}}\t{{.Status}}'], 
                              capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode == 0 and 'chenking-custom-embeddings' in result.stdout:
            print("✅ Embedding API container is running")
            
            # Also check health endpoint
            health_result = subprocess.run(['curl', '-s', 'http://localhost:8002/health'], 
                                         capture_output=True, text=True)
            if health_result.returncode == 0 and 'healthy' in health_result.stdout:
                print("✅ API health check passed")
                return True
            else:
                print("⚠️  API container running but health check failed")
                return False
        else:
            print("❌ Embedding API container is not running")
            print("Run: docker compose up custom-embedding-api -d")
            return False
    except Exception as e:
        print(f"❌ Error checking API status: {e}")
        return False

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--integration':
        # Check API status first
        if not check_api_status():
            print("\n❌ Cannot run integration test - API not available")
            sys.exit(1)
        
        success = run_integration_test()
        if not success:
            sys.exit(1)
    elif len(sys.argv) > 1 and sys.argv[1] == '--check-api':
        success = check_api_status()
        if not success:
            sys.exit(1)
    elif len(sys.argv) > 1 and sys.argv[1] == '--full':
        # Run unit tests first
        print("🧪 Running Unit Tests...")
        unit_success = run_tests()
        
        # Then check API and run integration test
        print("\n🔍 Checking API Status...")
        api_success = check_api_status()
        
        if api_success:
            print("\n🔗 Running Integration Test...")
            integration_success = run_integration_test()
        else:
            print("\n⚠️  Skipping integration test - API not available")
            integration_success = False
        
        # Summary
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Unit Tests: {'✅ PASSED' if unit_success else '❌ FAILED'}")
        print(f"API Status: {'✅ HEALTHY' if api_success else '❌ UNHEALTHY'}")
        print(f"Integration: {'✅ PASSED' if integration_success else '❌ FAILED/SKIPPED'}")
        
        if not (unit_success and api_success and integration_success):
            sys.exit(1)
    else:
        # Default: just run unit tests
        success = run_tests()
        if not success:
            sys.exit(1)
        
        print("\n💡 Tips:")
        print("  --integration    Run integration test with local API")
        print("  --check-api      Check if local embedding API is running")
        print("  --full           Run unit tests + integration test")
