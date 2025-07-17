# Chenking Test Suite

This directory contains comprehensive test cases for the **Chenking** package, with `Processor` as the main class that orchestrates document validation and embedding generation.

## Test Architecture

### 📁 Test Files

1. **`test_document_processor.py`** - Main test suite (16 tests)
   - Tests for the primary `Processor` class
   - Integration testing with mocked embedding API
   - Batch processing and statistics generation
   - Error handling and edge cases

2. **`test_document_checker.py`** - Document validation tests (27 tests)
   - Comprehensive validation logic testing
   - Configuration parameter validation
   - All check methods and their edge cases
   - Performance and timeout testing

3. **`test_embedding_client.py`** - API client tests (17 tests)
   - HTTP request handling and retry logic
   - Error scenarios and timeout handling
   - Health checks and API communication
   - Response processing and validation

### 🏗️ Package Structure
```
chenking/
├── __init__.py              # Package exports
├── document_processor.py    # Main orchestration class
├── document_checker.py      # Document validation engine  
└── embedding_client.py      # API communication client
```

## Running Tests

### 🚀 Quick Start
```bash
# Run all tests with Poetry
poetry run python run_tests.py

# Run with pytest (more detailed output)
poetry run pytest tests/ -v

# Run with coverage
poetry run pytest tests/ --cov=chenking --cov-report=html
```

### 🎯 Specific Test Categories
```bash
# Run only Processor tests
poetry run pytest tests/test_document_processor.py -v

# Run only Chenker tests  
poetry run pytest tests/test_document_checker.py -v

# Run only EmbeddingClient tests
poetry run pytest tests/test_embedding_client.py -v
```

### 🔍 Coverage and Quality
```bash
# Generate HTML coverage report
poetry run pytest tests/ --cov=chenking --cov-report=html

# Run with coverage and missing lines
poetry run pytest tests/ --cov=chenking --cov-report=term-missing

# Check code style
poetry run black chenking/ tests/
poetry run flake8 chenking/ tests/
```

## Test Coverage: 99% �

| Module | Statements | Missing | Coverage |
|--------|------------|---------|----------|
| `document_processor.py` | 67 | 1 | **99%** |
| `document_checker.py` | 86 | 0 | **100%** |
| `embedding_client.py` | 44 | 1 | **98%** |
| **TOTAL** | **202** | **2** | **99%** |

## Demonstrations

### 🎬 Usage Examples
```bash
# See Processor in action
poetry run python run_tests.py --demo

# Demonstrate EmbeddingClient features
poetry run python run_tests.py --demo-embedding

# Run all demonstrations
poetry run python run_tests.py --demo-all
```

### 📋 Example Output
```
============================================================
Processor Usage Demonstration
============================================================

1. Basic Processor Usage:
Document processed successfully: demo_001
Checks completed: 5
Processing time: 0.0007s
Successful checks: 5/5

2. Custom Configuration:
Custom processing completed for: short_001

3. Batch Processing:
Batch processed: 3 documents
Success rate: 100.00%
Average processing time: 0.0002s
```

## Test Features

### ✅ Processor Tests
- **Initialization**: Default and custom configurations
- **Document Processing**: Valid documents, validation, embedding generation
- **Error Handling**: Invalid documents, API failures, checker errors
- **Batch Processing**: Multiple documents, error scenarios
- **Statistics**: Processing metrics and success rates
- **Integration**: End-to-end workflow testing

### ✅ Chenker Tests  
- **Configuration**: Parameter validation and edge cases
- **Check Methods**: All 5 validation checks (basic, length, format, field, quality)
- **Performance**: Timeout protection and execution timing
- **Logging**: Debug and info message generation
- **Edge Cases**: Empty documents, malformed input

### ✅ EmbeddingClient Tests
- **API Communication**: HTTP requests with proper headers
- **Retry Logic**: Exponential backoff and error recovery
- **Error Handling**: Timeouts, network errors, HTTP errors
- **Health Checks**: API availability monitoring
- **Response Processing**: Partial responses and data validation

## Key Testing Strategies

### 🎭 Mocking Strategy
- **HTTP Requests**: All external API calls are mocked
- **Time Functions**: Timeout and retry scenarios use time mocking
- **Error Injection**: Systematic testing of failure modes
- **Response Simulation**: Realistic API response patterns

### 🧪 Test Categories
1. **Unit Tests**: Individual component functionality
2. **Integration Tests**: Component interaction testing  
3. **Error Testing**: Failure scenarios and recovery
4. **Performance Tests**: Timeout and execution time validation
5. **Edge Case Tests**: Boundary conditions and invalid input

### 📊 Validation Patterns
- **Input Validation**: Type checking and required field validation
- **Output Verification**: Response structure and data integrity
- **State Testing**: Object state consistency after operations
- **Side Effect Testing**: Logging, timing, and external interactions

## Development Workflow

### 🔄 Test-Driven Development
1. **Write Tests First**: Define expected behavior
2. **Implement Features**: Make tests pass
3. **Refactor Code**: Improve while maintaining test coverage
4. **Add Edge Cases**: Expand test coverage for robustness

### 🐛 Debugging Tests
```bash
# Run specific test with detailed output
poetry run pytest tests/test_document_processor.py::TestProcessor::test_process_valid_document -v -s

# Run tests with pdb debugging
poetry run pytest tests/ --pdb

# Run only failed tests from last run
poetry run pytest tests/ --lf
```

### 📈 Continuous Integration
- All tests must pass before merging
- Coverage must remain above 95%
- No lint errors allowed
- Documentation must be updated

## Sample Test Code

### Processor Example
```python
@patch('chenking.embedding_client.requests.post')
def test_process_valid_document(self, mock_post):
    """Test processing a valid document successfully."""
    # Mock embedding API response
    mock_response = Mock()
    mock_response.json.return_value = {
        "embedding": [0.1, 0.2, 0.3],
        "vector": [0.4, 0.5, 0.6]
    }
    mock_post.return_value = mock_response
    
    result = self.processor.process(self.valid_document)
    
    # Verify processing results
    self.assertIn("chenkings", result)
    self.assertIn("processing_info", result)
```

### EmbeddingClient Example  
```python
def test_get_embedding_retry_with_backoff(self):
    """Test retry mechanism with exponential backoff."""
    mock_post.side_effect = [
        requests.exceptions.RequestException("Error 1"),
        requests.exceptions.RequestException("Error 2"), 
        mock_success_response
    ]
    
    result = self.client.get_embedding(data)
    
    # Verify retry behavior
    self.assertEqual(mock_post.call_count, 3)
    self.assertEqual(result["status"], "success")
```

## Contributing

### 🚀 Adding New Tests
1. **Follow naming convention**: `test_<feature_name>`
2. **Include docstrings**: Explain test purpose clearly
3. **Test both success and failure**: Positive and negative cases
4. **Mock external dependencies**: Use appropriate mocking
5. **Verify assertions**: Include meaningful assertion messages

### 📝 Test Documentation
- **Document test purpose**: What functionality is being tested
- **Explain complex setups**: Mock configurations and data preparation
- **Note dependencies**: External services or specific configurations
- **Include examples**: Sample inputs and expected outputs

### 🎯 Quality Standards
- **Test Independence**: Each test should run in isolation
- **Deterministic Results**: Tests should have consistent outcomes
- **Meaningful Names**: Test names should describe behavior
- **Appropriate Scope**: One concept per test method
- **Clear Assertions**: Verify specific behaviors, not implementation details

---

**Total Test Count**: 60 tests  
**Execution Time**: ~90 seconds  
**Coverage**: 99%  
**Success Rate**: 100% ✅
