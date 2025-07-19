#!/usr/bin/env python3
"""
Simple Chenking + LlamaIndex Integration Example

This example shows how to use Chenking for document validation
before indexing with LlamaIndex, without complex dependencies.
"""

import json
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch

# Import Chenking components
from chenking.processor import Processor
from chenking.chenker import Chenker


class SimpleChenkingLlamaIntegration:
    """
    Simple integration example showing how Chenking can enhance
    LlamaIndex document processing pipelines.
    """
    
    def __init__(self, embedding_api_url: str, validation_config: Optional[Dict] = None):
        """Initialize with Chenking processor and validation config."""
        self.processor = Processor(embedding_api_url, **(validation_config or {}))
        self.validation_stats = {"processed": 0, "valid": 0, "invalid": 0}
        self.validated_documents = []
    
    def validate_documents_for_llama(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate documents using Chenking before LlamaIndex processing.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of validated documents ready for LlamaIndex
        """
        validated = []
        
        print(f"🔍 Validating {len(documents)} documents with Chenking...")
        
        for doc in documents:
            try:
                # Process with Chenking
                result = self.processor.process(doc)
                self.validation_stats["processed"] += 1
                
                # Check if document is valid
                if self._is_document_valid(result):
                    # Convert to LlamaIndex-ready format
                    llama_doc = self._prepare_for_llama(doc, result)
                    validated.append(llama_doc)
                    self.validation_stats["valid"] += 1
                    
                    print(f"✅ {doc.get('id', 'unknown')}: PASSED validation")
                else:
                    self.validation_stats["invalid"] += 1
                    print(f"❌ {doc.get('id', 'unknown')}: FAILED validation")
                    self._print_validation_issues(result)
                    
            except Exception as e:
                self.validation_stats["invalid"] += 1
                print(f"💥 {doc.get('id', 'unknown')}: ERROR - {e}")
        
        self.validated_documents = validated
        self._print_validation_summary()
        return validated
    
    def _is_document_valid(self, chenking_result: Dict[str, Any]) -> bool:
        """Determine if document passed Chenking validation."""
        # Check for processing errors
        if chenking_result.get("processing_info", {}).get("status") == "failed":
            return False
        
        # Calculate success rate
        info = chenking_result.get("processing_info", {})
        successful = info.get("successful_checks", 0)
        total = info.get("checks_completed", 1)
        success_rate = successful / total
        
        # Require at least 80% of checks to pass
        return success_rate >= 0.8
    
    def _prepare_for_llama(self, original_doc: Dict[str, Any], 
                          chenking_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare validated document for LlamaIndex with enhanced metadata.
        
        This format is compatible with LlamaIndex Document objects:
        Document(text=doc['text'], metadata=doc['metadata'])
        """
        # Extract quality metrics
        quality_score = self._calculate_quality_score(chenking_result)
        
        # Prepare LlamaIndex-compatible document
        llama_doc = {
            "text": original_doc.get("content", ""),
            "metadata": {
                # Original metadata
                "doc_id": original_doc.get("id"),
                "title": original_doc.get("title"),
                "author": original_doc.get("author"),
                "format": original_doc.get("format"),
                
                # Chenking validation metadata
                "chenking_validation": {
                    "quality_score": quality_score,
                    "checks_passed": chenking_result.get("processing_info", {}).get("successful_checks", 0),
                    "total_checks": chenking_result.get("processing_info", {}).get("checks_completed", 0),
                    "processing_time": chenking_result.get("processing_info", {}).get("total_processing_time", 0),
                    "validation_timestamp": chenking_result.get("processing_info", {}).get("timestamp")
                },
                
                # Content metrics for better retrieval
                "content_metrics": self._extract_content_metrics(chenking_result)
            }
        }
        
        # Add original metadata if present
        if "metadata" in original_doc:
            llama_doc["metadata"].update(original_doc["metadata"])
        
        return llama_doc
    
    def _calculate_quality_score(self, chenking_result: Dict[str, Any]) -> float:
        """Calculate overall quality score from validation results."""
        chenkings = chenking_result.get("chenkings", {})
        scores = []
        
        # Basic content quality
        basic_check = chenkings.get("basic_check", {}).get("data", {})
        if basic_check.get("is_word_count_valid"):
            scores.append(1.0)
        else:
            scores.append(0.0)
        
        # Format support
        format_check = chenkings.get("format_check", {}).get("data", {})
        if format_check.get("is_format_supported"):
            scores.append(1.0)
        else:
            scores.append(0.5)
        
        # Content quality
        quality_check = chenkings.get("content_quality_check", {}).get("data", {})
        if quality_check:
            completeness = quality_check.get("metadata_completeness", {}).get("completeness_score", 0)
            scores.append(completeness)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _extract_content_metrics(self, chenking_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract useful content metrics for LlamaIndex."""
        metrics = {}
        chenkings = chenking_result.get("chenkings", {})
        
        # Basic metrics
        basic_data = chenkings.get("basic_check", {}).get("data", {})
        metrics["word_count"] = basic_data.get("word_count", 0)
        
        # Length metrics
        length_data = chenkings.get("length_check", {}).get("data", {})
        metrics["char_count"] = length_data.get("char_count", 0)
        metrics["line_count"] = length_data.get("line_count", 0)
        
        # Quality metrics
        quality_data = chenkings.get("content_quality_check", {}).get("data", {})
        if quality_data:
            structure = quality_data.get("structure_metrics", {})
            metrics["sentence_count"] = structure.get("sentence_count", 0)
            metrics["paragraph_count"] = structure.get("paragraph_count", 0)
        
        return metrics
    
    def _print_validation_issues(self, chenking_result: Dict[str, Any]):
        """Print specific validation issues for debugging."""
        chenkings = chenking_result.get("chenkings", {})
        
        for check_name, check_result in chenkings.items():
            if check_result.get("status") == "error":
                print(f"    💥 {check_name}: {check_result.get('error', 'Unknown error')}")
            elif check_result.get("status") == "success":
                # Check for specific validation failures
                data = check_result.get("data", {})
                if check_name == "basic_check" and not data.get("is_word_count_valid"):
                    print(f"    ⚠️  {check_name}: Word count issue ({data.get('word_count', 0)} words)")
                elif check_name == "format_check" and not data.get("is_format_supported"):
                    print(f"    ⚠️  {check_name}: Unsupported format ({data.get('format', 'unknown')})")
    
    def _print_validation_summary(self):
        """Print validation summary statistics."""
        stats = self.validation_stats
        total = stats["processed"]
        
        if total > 0:
            success_rate = stats["valid"] / total * 100
            print(f"\n📊 Validation Summary:")
            print(f"   Total processed: {total}")
            print(f"   Passed: {stats['valid']} ({success_rate:.1f}%)")
            print(f"   Failed: {stats['invalid']} ({100-success_rate:.1f}%)")
        
    def simulate_llama_index_usage(self):
        """
        Simulate how you would use the validated documents with LlamaIndex.
        This shows the integration pattern without requiring LlamaIndex installation.
        """
        if not self.validated_documents:
            print("❌ No validated documents available for LlamaIndex")
            return
        
        print(f"\n🔗 Simulating LlamaIndex integration with {len(self.validated_documents)} documents...")
        
        # This is what you would do with actual LlamaIndex:
        print("\n📝 LlamaIndex Integration Pattern:")
        print("```python")
        print("from llama_index import Document, VectorStoreIndex")
        print()
        print("# Convert Chenking-validated docs to LlamaIndex Documents")
        print("llama_docs = [")
        print("    Document(text=doc['text'], metadata=doc['metadata'])")
        print("    for doc in validated_documents")
        print("]")
        print()
        print("# Create vector index")
        print("index = VectorStoreIndex.from_documents(llama_docs)")
        print("query_engine = index.as_query_engine()")
        print()
        print("# Query with validated, high-quality documents")
        print("response = query_engine.query('Your question here')")
        print("```")
        
        # Show sample document structure
        print(f"\n📄 Sample validated document structure:")
        sample_doc = self.validated_documents[0]
        print(json.dumps({
            "text": sample_doc["text"][:100] + "...",
            "metadata": {
                k: v for k, v in sample_doc["metadata"].items() 
                if k in ["doc_id", "title", "chenking_validation", "content_metrics"]
            }
        }, indent=2))


def demo_chenking_llama_integration():
    """Demonstrate Chenking + LlamaIndex integration."""
    print("🚀 Chenking + LlamaIndex Integration Demo")
    print("=" * 50)
    
    # Sample documents with varying quality
    documents = [
        {
            "id": "high_quality_doc",
            "content": "This is a comprehensive document about machine learning and artificial intelligence. It covers various algorithms including supervised learning, unsupervised learning, and reinforcement learning. The document provides detailed explanations of neural networks, decision trees, and support vector machines. It also discusses practical applications in computer vision, natural language processing, and recommendation systems.",
            "title": "Machine Learning Comprehensive Guide",
            "author": "AI Researcher",
            "format": "txt",
            "metadata": {"category": "education", "source": "university"}
        },
        {
            "id": "medium_quality_doc", 
            "content": "Brief overview of Python programming. Covers basic syntax and common libraries.",
            "title": "Python Basics",
            "format": "md"
        },
        {
            "id": "low_quality_doc",
            "content": "Too short",  # Will fail validation
            "title": "Short Note"
        },
        {
            "id": "good_structured_doc",
            "content": "Deep learning represents a significant advancement in artificial intelligence, utilizing neural networks with multiple layers to learn complex patterns from data. This field has revolutionized computer vision, enabling systems to recognize objects, faces, and scenes with human-level accuracy. Natural language processing has also benefited tremendously, with models like transformers achieving remarkable performance in translation, summarization, and text generation.",
            "title": "Deep Learning Revolution",
            "author": "Neural Network Expert", 
            "format": "txt",
            "metadata": {"category": "research", "year": "2024"}
        }
    ]
    
    # Mock embedding API for demo
    with patch('chenking.embedding_client.requests.post') as mock_post:
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
            "vector": [0.6, 0.7, 0.8, 0.9, 1.0]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        # Configure validation for demonstration
        validation_config = {
            "min_word_count": 15,  # Require at least 15 words
            "max_word_count": 5000,
            "supported_formats": ["txt", "md", "html"],
            "required_fields": ["content", "title"],
            "enable_detailed_logging": False
        }
        
        # Create integration instance
        integration = SimpleChenkingLlamaIntegration(
            "https://api.example.com/embeddings",
            validation_config
        )
        
        # Validate documents for LlamaIndex
        validated_docs = integration.validate_documents_for_llama(documents)
        
        # Simulate LlamaIndex usage
        integration.simulate_llama_index_usage()
        
        return validated_docs


if __name__ == "__main__":
    demo_chenking_llama_integration()
