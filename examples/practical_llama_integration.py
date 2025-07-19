#!/usr/bin/env python3
"""
Practical Chenking + LlamaIndex Integration (Mock Mode)

This example shows how to integrate Chenking with LlamaIndex
without external dependencies, perfect for development and testing.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, Mock
from chenking.processor import Processor
from typing import List, Dict, Any


class MockedChenkingLlamaIntegration:
    """
    Production-ready integration example with mocked responses
    for development and testing.
    """
    
    def __init__(self) -> None:
        # Mock the embedding client to avoid network calls
        with patch('chenking.embedding_client.EmbeddingClient.get_embedding') as mock_embedding:
            mock_embedding.return_value = [0.1] * 512  # Mock 512-dim embedding
            self.processor = Processor("https://mock-api.com")
        print("🔧 Chenking processor initialized (mock mode)")
    
    def create_quality_filtered_llama_pipeline(self, documents: List[Dict[str, Any]], 
                                               min_quality: float = 0.8) -> List[Dict[str, Any]]:
        """
        Create a quality-filtered pipeline for LlamaIndex.
        
        Args:
            documents: Raw documents to process
            min_quality: Minimum quality score (0.0 to 1.0)
            
        Returns:
            List of high-quality documents ready for LlamaIndex
        """
        print(f"\n🔍 Processing {len(documents)} documents (min quality: {min_quality})")
        
        validated_docs = []
        quality_stats = {"high": 0, "medium": 0, "low": 0, "failed": 0}
        
        # Mock the embedding calls for consistent demo
        with patch('chenking.embedding_client.EmbeddingClient.get_embedding') as mock_embedding:
            mock_embedding.return_value = [0.1] * 512
            
            for doc in documents:
                try:
                    # Process with Chenking
                    result = self.processor.process(doc)
                    quality_score = self._calculate_quality_score(result)
                    
                    # Categorize quality
                    if quality_score >= 0.9:
                        quality_stats["high"] += 1
                        quality_label = "HIGH"
                    elif quality_score >= 0.7:
                        quality_stats["medium"] += 1  
                        quality_label = "MEDIUM"
                    elif quality_score >= 0.5:
                        quality_stats["low"] += 1
                        quality_label = "LOW"
                    else:
                        quality_stats["failed"] += 1
                        quality_label = "FAILED"
                        
                    # Filter by minimum quality
                    if quality_score >= min_quality:
                        llama_doc = self._create_llama_document(doc, result, quality_score)
                        validated_docs.append(llama_doc)
                        status = "✅ ACCEPTED"
                    else:
                        status = "❌ REJECTED"
                    
                    print(f"{status} {doc.get('id', 'unknown')}: {quality_label} ({quality_score:.2f})")
                    
                except Exception as e:
                    quality_stats["failed"] += 1
                    print(f"💥 {doc.get('id', 'unknown')}: ERROR - {e}")
        
        self._print_quality_summary(quality_stats, validated_docs, len(documents))
        return validated_docs
    
    def _calculate_quality_score(self, result: Dict[str, Any]) -> float:
        """Calculate quality score from Chenking processing results."""
        info = result.get("processing_info", {})
        successful = info.get("successful_checks", 0) or 0
        total = info.get("checks_completed", 1) or 1
        return float(successful / total)
    
    def _create_llama_document(self, original_doc: Dict[str, Any], 
                              chenking_result: Dict[str, Any], 
                              quality_score: float) -> Dict[str, Any]:
        """Create LlamaIndex-compatible document with enhanced metadata."""
        return {
            "text": original_doc["content"],
            "metadata": {
                # Original metadata
                "doc_id": original_doc.get("id"),
                "title": original_doc.get("title"),
                "source": original_doc.get("source", "unknown"),
                
                # Chenking quality metadata
                "quality_score": quality_score,
                "validation_passed": True,
                "chenking_checks": {
                    "passed": chenking_result.get("processing_info", {}).get("successful_checks", 0),
                    "total": chenking_result.get("processing_info", {}).get("checks_completed", 0),
                    "processing_time": chenking_result.get("processing_info", {}).get("total_processing_time", 0)
                },
                
                # Content metrics for better retrieval
                "content_length": len(original_doc["content"]),
                "word_count": len(original_doc["content"].split()),
                "estimated_read_time": len(original_doc["content"].split()) / 200  # words per minute
            }
        }
    
    def _print_quality_summary(self, stats: Dict[str, int], validated: List[Dict], total: int) -> None:
        """Print detailed quality analysis summary."""
        print(f"\n📊 Quality Analysis Summary:")
        print(f"   🔴 Failed: {stats['failed']} documents")
        print(f"   🟡 Low Quality: {stats['low']} documents (0.5-0.7)")
        print(f"   🟠 Medium Quality: {stats['medium']} documents (0.7-0.9)")
        print(f"   🟢 High Quality: {stats['high']} documents (0.9+)")
        print(f"   ✅ Accepted for LlamaIndex: {len(validated)}/{total} ({len(validated)/total*100:.1f}%)")
    
    def demonstrate_llamaindex_usage(self, validated_docs: List[Dict[str, Any]]) -> None:
        """Show practical LlamaIndex integration patterns."""
        if not validated_docs:
            print("❌ No validated documents available for LlamaIndex demo")
            return
            
        print(f"\n🚀 LlamaIndex Integration with {len(validated_docs)} validated documents")
        print("=" * 60)
        
        # Show the integration code
        print("📝 Production Integration Code:")
        print("```python")
        print("from llama_index import Document, VectorStoreIndex")
        print("from llama_index.query_engine import RetrieverQueryEngine")
        print("")
        print("# Convert Chenking-validated docs to LlamaIndex format")
        print("llama_documents = [")
        print("    Document(text=doc['text'], metadata=doc['metadata'])")
        print("    for doc in chenking_validated_documents")
        print("    if doc['metadata']['quality_score'] >= 0.8  # Quality filter")
        print("]")
        print("")
        print("# Create vector index with quality-assured documents")
        print("index = VectorStoreIndex.from_documents(llama_documents)")
        print("")
        print("# Create query engine with enhanced retrieval")
        print("query_engine = index.as_query_engine(")
        print("    similarity_top_k=5,")
        print("    response_mode='tree_summarize'")
        print(")")
        print("")
        print("# Query with confidence in document quality")
        print("response = query_engine.query(")
        print("    'What are the key technical insights?'")
        print(")")
        print("```")
        
        # Show sample document
        sample_doc = validated_docs[0]
        print(f"\n📄 Sample validated document structure:")
        print(f"   Document ID: {sample_doc['metadata']['doc_id']}")
        print(f"   Quality Score: {sample_doc['metadata']['quality_score']:.2f}")
        print(f"   Word Count: {sample_doc['metadata']['word_count']}")
        print(f"   Checks Passed: {sample_doc['metadata']['chenking_checks']['passed']}/{sample_doc['metadata']['chenking_checks']['total']}")
        print(f"   Content Preview: {sample_doc['text'][:100]}...")


def main() -> None:
    """Run the comprehensive integration demonstration."""
    print("🚀 Practical Chenking + LlamaIndex Integration Demo")
    print("=" * 55)
    
    # Sample documents with varying quality
    sample_documents = [
        {
            "id": "tech_whitepaper",
            "title": "Advanced Machine Learning in Production",
            "source": "engineering_blog",
            "content": "This comprehensive whitepaper examines the implementation of advanced machine learning models in production environments. We discuss deployment strategies, monitoring approaches, model versioning, and performance optimization techniques. The document covers real-world case studies from enterprise implementations, including challenges faced during scaling, data pipeline management, and maintaining model accuracy over time. Key topics include MLOps practices, automated testing, continuous integration for ML models, and infrastructure considerations for high-throughput prediction services."
        },
        {
            "id": "meeting_notes",
            "title": "Weekly Team Sync",
            "source": "internal_notes",
            "content": "Quick meeting notes from today's sync. Discussed project timeline and next steps."
        },
        {
            "id": "research_paper",
            "title": "Transformer Architectures for Natural Language Processing",
            "source": "academic_paper",
            "content": "This research paper presents a comprehensive analysis of transformer architectures and their applications in natural language processing tasks. We examine the attention mechanism, positional encoding, and multi-head attention structures that form the foundation of modern NLP models. Our experiments demonstrate the effectiveness of these architectures across various tasks including machine translation, text summarization, and question answering. The paper includes detailed ablation studies, performance comparisons with traditional approaches, and discussions on computational efficiency and scaling considerations for large language models."
        },
        {
            "id": "incomplete_doc",
            "title": "Draft Document",
            "source": "work_in_progress", 
            "content": "This is just a draft..."
        },
        {
            "id": "comprehensive_guide",
            "title": "Complete Guide to Distributed Systems",
            "source": "technical_documentation",
            "content": "This comprehensive guide covers all aspects of distributed systems design and implementation. Topics include consistency models, consensus algorithms, distributed storage systems, microservices architecture, load balancing strategies, fault tolerance mechanisms, and monitoring approaches. The guide provides practical examples, implementation patterns, and best practices derived from real-world deployments. Each section includes code examples, architectural diagrams, and performance considerations. Special attention is given to emerging technologies like service mesh, containerization, and cloud-native approaches to distributed system design."
        }
    ]
    
    # Initialize integration
    integration = MockedChenkingLlamaIntegration()
    
    # Test with different quality thresholds
    print("\n" + "="*60)
    print("🎯 Testing with HIGH quality threshold (0.9+)")
    high_quality_docs = integration.create_quality_filtered_llama_pipeline(
        sample_documents, min_quality=0.9
    )
    
    print("\n" + "="*60)  
    print("🎯 Testing with MEDIUM quality threshold (0.7+)")
    medium_quality_docs = integration.create_quality_filtered_llama_pipeline(
        sample_documents, min_quality=0.7
    )
    
    # Demonstrate LlamaIndex usage
    integration.demonstrate_llamaindex_usage(medium_quality_docs)
    
    print(f"\n✨ Demo complete! Ready to integrate Chenking quality control with LlamaIndex RAG.")


if __name__ == "__main__":
    main()
