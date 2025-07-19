# Chenking + LlamaIndex Integration Guide

This guide shows you how to integrate **Chenking's document validation and processing** with **LlamaIndex's RAG capabilities** to create robust, high-quality document ingestion pipelines.

## 🎯 Integration Benefits

### **Chenking adds to LlamaIndex:**
- ✅ **Document Quality Control** - Validate documents before indexing
- ✅ **Content Filtering** - Only high-quality docs make it to your RAG system
- ✅ **Metadata Enhancement** - Rich validation metadata for better retrieval
- ✅ **Batch Processing** - Efficient handling of large document collections
- ✅ **Error Handling** - Graceful handling of malformed documents

### **LlamaIndex adds to Chenking:**
- ✅ **Vector Search** - Semantic retrieval capabilities
- ✅ **LLM Integration** - Query answering with GPT/Claude/etc.
- ✅ **Document Chunking** - Smart text splitting for embeddings
- ✅ **Multiple Data Sources** - PDFs, web pages, databases, etc.

## 🏗️ Integration Patterns

### **Pattern 1: Pre-Processing Pipeline**
```python
# Step 1: Validate with Chenking
from chenking import Processor

processor = Processor("https://your-embedding-api.com")
validated_docs = []

for doc in raw_documents:
    result = processor.process(doc)
    if is_valid(result):  # Your validation logic
        validated_docs.append(convert_to_llama_format(doc, result))

# Step 2: Index with LlamaIndex
from llama_index import VectorStoreIndex, Document

llama_docs = [Document(text=doc['content'], metadata=doc['metadata']) 
              for doc in validated_docs]
index = VectorStoreIndex.from_documents(llama_docs)
```

### **Pattern 2: Custom Document Loader**
```python
from llama_index.readers.base import BaseReader
from chenking import Processor

class ChenkingValidatedReader(BaseReader):
    def __init__(self, validation_config=None):
        self.processor = Processor("https://api.example.com", **(validation_config or {}))
    
    def load_data(self, documents):
        validated = []
        for doc in documents:
            result = self.processor.process(doc)
            if self._is_valid(result):
                validated.append(self._to_llama_doc(doc, result))
        return validated
```

### **Pattern 3: Embedding Integration**
```python
from llama_index.embeddings.base import BaseEmbedding
from chenking import EmbeddingClient

class ChenkingEmbedding(BaseEmbedding):
    def __init__(self, api_url):
        self.client = EmbeddingClient(api_url)
    
    def _get_text_embedding(self, text):
        check_data = {"content": text, "word_count": len(text.split())}
        result = self.client.get_embedding(check_data)
        return result.get("embedding", [])
```

## 💡 Use Cases

### **1. Enterprise Document Ingestion**
```python
# Validate corporate documents before RAG indexing
validation_config = {
    "min_word_count": 50,
    "required_fields": ["content", "title", "department"],
    "supported_formats": ["pdf", "docx", "txt"],
    "enable_detailed_logging": True
}

# Only well-structured, complete documents make it to the RAG system
```

### **2. Academic Paper Processing**
```python
# Ensure research papers meet quality standards
academic_config = {
    "min_word_count": 1000,
    "required_fields": ["content", "title", "authors", "abstract"],
    "check_timeout": 60  # Longer timeout for large papers
}
```

### **3. Customer Support Knowledge Base**
```python
# Validate support articles for completeness
support_config = {
    "min_word_count": 20,
    "required_fields": ["content", "title", "category"],
    "max_word_count": 2000  # Keep articles concise
}
```

## 🔧 Implementation Examples

### **Basic Integration**
```python
def create_validated_rag_system(documents, embedding_api_url):
    from chenking import Processor
    from llama_index import VectorStoreIndex, Document
    
    # Step 1: Validate documents
    processor = Processor(embedding_api_url, min_word_count=20)
    valid_docs = []
    
    for doc in documents:
        try:
            result = processor.process(doc)
            if result['processing_info']['successful_checks'] >= 4:  # 4 out of 5 checks
                # Add validation metadata
                metadata = {
                    'validation_score': result['processing_info']['successful_checks'],
                    'processing_time': result['processing_info']['total_processing_time'],
                    'quality_metrics': extract_quality_metrics(result)
                }
                valid_docs.append(Document(
                    text=doc['content'], 
                    metadata={**doc.get('metadata', {}), **metadata}
                ))
        except Exception as e:
            print(f"Validation failed for {doc.get('id', 'unknown')}: {e}")
    
    # Step 2: Create RAG index
    if valid_docs:
        index = VectorStoreIndex.from_documents(valid_docs)
        return index
    else:
        raise ValueError("No documents passed validation")

# Usage
documents = [
    {
        "id": "doc1",
        "content": "Your document content here...",
        "title": "Document Title",
        "metadata": {"source": "web", "category": "tech"}
    }
]

index = create_validated_rag_system(documents, "https://api.example.com/embed")
query_engine = index.as_query_engine()
response = query_engine.query("Your question here")
```

### **Advanced Quality Filtering**
```python
def extract_quality_metrics(chenking_result):
    """Extract quality metrics from Chenking validation results."""
    metrics = {}
    
    # Basic content metrics
    basic_check = chenking_result.get('chenkings', {}).get('basic_check', {})
    if basic_check.get('status') == 'success':
        data = basic_check['data']
        metrics['word_count'] = data.get('word_count', 0)
        metrics['has_sufficient_content'] = data.get('is_word_count_valid', False)
    
    # Content quality metrics
    quality_check = chenking_result.get('chenkings', {}).get('content_quality_check', {})
    if quality_check.get('status') == 'success':
        data = quality_check['data']
        metrics['structure_score'] = data.get('structure_metrics', {})
        metrics['completeness_score'] = data.get('metadata_completeness', {}).get('completeness_score', 0)
    
    return metrics

def is_high_quality_document(chenking_result, min_quality_score=0.7):
    """Determine if document meets quality standards."""
    metrics = extract_quality_metrics(chenking_result)
    
    # Check minimum requirements
    if not metrics.get('has_sufficient_content', False):
        return False
    
    # Check completeness score
    if metrics.get('completeness_score', 0) < min_quality_score:
        return False
    
    # Check successful validation rate
    successful = chenking_result.get('processing_info', {}).get('successful_checks', 0)
    total = chenking_result.get('processing_info', {}).get('checks_completed', 1)
    
    return (successful / total) >= min_quality_score
```

## 📊 Monitoring and Analytics

### **Validation Pipeline Metrics**
```python
class ValidationPipelineMonitor:
    def __init__(self):
        self.stats = {
            'total_processed': 0,
            'passed_validation': 0,
            'failed_validation': 0,
            'quality_scores': [],
            'processing_times': []
        }
    
    def track_document(self, chenking_result):
        self.stats['total_processed'] += 1
        
        if is_high_quality_document(chenking_result):
            self.stats['passed_validation'] += 1
        else:
            self.stats['failed_validation'] += 1
        
        # Track quality metrics
        metrics = extract_quality_metrics(chenking_result)
        self.stats['quality_scores'].append(metrics.get('completeness_score', 0))
        
        # Track processing time
        proc_time = chenking_result.get('processing_info', {}).get('total_processing_time', 0)
        self.stats['processing_times'].append(proc_time)
    
    def get_summary(self):
        total = self.stats['total_processed']
        if total == 0:
            return "No documents processed yet"
        
        return {
            'validation_rate': self.stats['passed_validation'] / total,
            'average_quality_score': sum(self.stats['quality_scores']) / len(self.stats['quality_scores']),
            'average_processing_time': sum(self.stats['processing_times']) / len(self.stats['processing_times']),
            'total_processed': total
        }
```

## 🚀 Production Deployment

### **Recommended Architecture**
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Documents │────│ Chenking Pipeline│────│ LlamaIndex RAG  │
│   (PDF, TXT,    │    │ - Validation     │    │ - Vector Store  │
│    Web, etc.)   │    │ - Quality Check  │    │ - Query Engine  │
└─────────────────┘    │ - Metadata       │    │ - LLM Response  │
                       └──────────────────┘    └─────────────────┘
```

### **Best Practices**
1. **Batch Processing**: Use `Processor.process_batch()` for large document collections
2. **Error Handling**: Always wrap validation in try-catch blocks
3. **Quality Thresholds**: Set appropriate quality thresholds for your use case
4. **Monitoring**: Track validation rates and quality metrics
5. **Caching**: Cache validation results for frequently processed documents

## 🔗 Next Steps

1. **Install Dependencies**:
   ```bash
   pip install llama-index chenking
   ```

2. **Try the Integration**: Start with the examples in `examples/llamaindex_integration.py`

3. **Customize Validation**: Adjust Chenking configuration for your specific needs

4. **Scale Up**: Use batch processing for production workloads

5. **Monitor Quality**: Implement quality monitoring for your document pipeline

---

**Ready to build better RAG systems with validated, high-quality documents!** 🎉
