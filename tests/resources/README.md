# Test Resources

This directory contains input and output test files for the Chenking Tika document processing system.

## Directory Structure

```
tests/resources/
├── inputs/          # Input test documents
└── outputs/         # Expected/sample outputs
```

## Input Files (`inputs/`)

### Example Documents
- **`example_tika_document.json`** - Single Tika-extracted PDF document example
- **`example_tika_batch.json`** - Batch of multiple Tika documents (PDF, DOCX, PPTX, XLSX)

### Test Documents  
- **`test_page_splitting.json`** - Multi-page document with form feed characters for testing page splitting
- **`short_test_doc.json`** - Short document (21 words) for testing word count validation

## Output Files (`outputs/`)

### Example Results
- **`example_single_result.json`** - Result from processing single example document
- **`example_batch_result.json`** - Result from processing batch example
- **`example_stdin_result.json`** - Result from stdin processing

### Test Results
- **`test_page_result.json`** - Page splitting test results
- **`test_no_pages.json`** - Same document with page splitting disabled
- **`test_chenk_numbering.json`** - Results showing chenk numbering
- **`test_chenk_final.json`** - Final chenk numbering test
- **`test_with_content.json`** - Results with enhanced content tracking
- **`short_test_result.json`** - Short document processing results

### Legacy Results
- **`tika_result_single.json`** - Legacy single document result
- **`tika_result_batch.json`** - Legacy batch processing result

## Usage

### Testing with Input Files

```bash
# Process single document
python tika_document_processor.py --input tests/resources/inputs/example_tika_document.json --output tests/resources/outputs/new_result.json

# Process batch
python tika_document_processor.py --input tests/resources/inputs/example_tika_batch.json --batch --output tests/resources/outputs/new_batch_result.json

# Test page splitting
python tika_document_processor.py --input tests/resources/inputs/test_page_splitting.json --output tests/resources/outputs/new_page_result.json

# Test short document
python tika_document_processor.py --input tests/resources/inputs/short_test_doc.json --output tests/resources/outputs/new_short_result.json
```

### Configuration Testing

```bash
# Test with different word limits
python tika_document_processor.py --input tests/resources/inputs/short_test_doc.json --max-word-count 30 --output tests/resources/outputs/word_limit_test.json

# Test without page splitting
python tika_document_processor.py --input tests/resources/inputs/test_page_splitting.json --disable-page-splitting --output tests/resources/outputs/no_pages_test.json

# Test with chenk numbering
python tika_document_processor.py --input tests/resources/inputs/example_tika_document.json --enable-chenk-numbering --output tests/resources/outputs/chenk_test.json
```

## Input Document Formats

All input documents follow the standard Tika extraction format:

```json
{
  "id": "unique_document_identifier",
  "content": "extracted text content from Tika",
  "metadata": {
    "file_name": "original_document.pdf",
    "Content-Type": "application/pdf",
    "author": "Document Author",
    "title": "Document Title",
    "page_count": 5,
    "creation_date": "2025-01-15T10:30:00Z"
  }
}
```

## Output Structure

All output documents include:
- Original document data (id, content, metadata)
- Chenking validation results for 6 checks
- Vector embeddings for successful validations
- Processing metadata and statistics
- Chenk numbering and content tracking
