#!/usr/bin/env python3
"""
Quick Chenking + LlamaIndex Integration Demo

This shows the simplest way to use Chenking with LlamaIndex.
Run this to see immediate results!
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chenking.processor import Processor
from typing import List, Dict, Any

class QuickChenkingLlamaDemo:
    """Simplest possible Chenking + LlamaIndex integration."""
    
    def __init__(self) -> None:
        # Initialize Chenking processor
        self.processor = Processor("https://api.mock-embedding.com")
        print("🔧 Chenking processor initialized")
    
    def validate_and_prepare_for_llama(self, documents: List[Dict]) -> List[Dict]:
        """
        Validate documents with Chenking and prepare for LlamaIndex.
        
        Returns LlamaIndex-ready documents with enhanced metadata.
        """
        validated = []
        
        print(f"\n🔍 Processing {len(documents)} documents...")
        
        for doc in documents:
            # Process with Chenking
            result = self.processor.process(doc)
            
            # Check if document passes validation
            if self._passes_validation(result):
                # Convert to LlamaIndex format
                llama_doc = {
                    "text": doc["content"],
                    "metadata": {
                        "doc_id": doc.get("id"),
                        "title": doc.get("title"),
                        "quality_score": self._get_quality_score(result),
                        "validation_passed": True,
                        "chenking_metrics": {
                            "checks_passed": result.get("processing_info", {}).get("successful_checks", 0),
                            "processing_time": result.get("processing_info", {}).get("total_processing_time", 0)
                        }
                    }
                }
                validated.append(llama_doc)
                print(f"✅ {doc.get('id', 'unknown')}: Quality score {self._get_quality_score(result):.2f}")
            else:
                print(f"❌ {doc.get('id', 'unknown')}: Failed validation")
        
        print(f"\n📊 Result: {len(validated)}/{len(documents)} documents validated for LlamaIndex")
        return validated
    
    def _passes_validation(self, result: Dict) -> bool:
        """Simple validation logic - require 80% check success rate."""
        info = result.get("processing_info", {})
        successful = info.get("successful_checks", 0) or 0
        total = info.get("checks_completed", 1) or 1
        return bool((successful / total) >= 0.8)
    
    def _get_quality_score(self, result: Dict) -> float:
        """Calculate quality score from Chenking results."""
        info = result.get("processing_info", {})
        successful = info.get("successful_checks", 0) or 0
        total = info.get("checks_completed", 1) or 1
        return float(successful / total)
    
    def show_llamaindex_integration_pattern(self, validated_docs: List[Dict]) -> None:
        """Show how to use validated docs with LlamaIndex."""
        print("\n🔗 LlamaIndex Integration Pattern:")
        print("```python")
        print("from llama_index import Document, VectorStoreIndex")
        print("")
        print("# Use Chenking-validated documents")
        print("llama_docs = [")
        print("    Document(text=doc['text'], metadata=doc['metadata'])")
        print("    for doc in validated_documents")
        print("]")
        print("")
        print("# Create index with high-quality documents")
        print("index = VectorStoreIndex.from_documents(llama_docs)")
        print("query_engine = index.as_query_engine()")
        print("")
        print("# Query with confidence in document quality")
        print("response = query_engine.query('What are the key insights?')")
        print("```")
        
        if validated_docs:
            print(f"\n📄 Sample validated document metadata:")
            sample = validated_docs[0]["metadata"]
            print(f"   Quality Score: {sample['quality_score']:.2f}")
            print(f"   Checks Passed: {sample['chenking_metrics']['checks_passed']}")
            print(f"   Processing Time: {sample['chenking_metrics']['processing_time']:.4f}s")


def main() -> None:
    """Run the quick integration demo."""
    print("🚀 Quick Chenking + LlamaIndex Integration Demo")
    print("=" * 50)
    
    # Sample documents
    sample_docs = [
        {
            "id": "tech_article",
            "title": "Advanced Machine Learning Techniques",
            "content": "This article explores cutting-edge machine learning algorithms including transformer architectures, attention mechanisms, and their applications in natural language processing. The content provides detailed explanations of how these technologies work and their practical implementations."
        },
        {
            "id": "short_note",
            "title": "Quick Note",
            "content": "Brief reminder about meeting."
        },
        {
            "id": "research_paper",
            "title": "Deep Learning in Computer Vision",
            "content": "This comprehensive research paper examines the evolution of deep learning techniques in computer vision applications. It covers convolutional neural networks, object detection algorithms, image segmentation methods, and their real-world applications in autonomous vehicles, medical imaging, and robotics. The paper provides both theoretical foundations and practical implementation guidelines for researchers and practitioners."
        }
    ]
    
    # Initialize demo
    demo = QuickChenkingLlamaDemo()
    
    # Validate documents
    validated = demo.validate_and_prepare_for_llama(sample_docs)
    
    # Show integration pattern
    demo.show_llamaindex_integration_pattern(validated)
    
    print(f"\n✨ Integration complete! Ready to use {len(validated)} validated documents with LlamaIndex.")


if __name__ == "__main__":
    main()
