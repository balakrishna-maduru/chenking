"""
Chenking + LlamaIndex Integration Examples

This module demonstrates how to integrate Chenking's document validation 
and processing capabilities with LlamaIndex for enhanced document ingestion
and RAG (Retrieval Augmented Generation) applications.
"""

from typing import List, Dict, Any, Optional
from llama_index import Document, VectorStoreIndex, ServiceContext
from llama_index.node_parser import SimpleNodeParser
from llama_index.embeddings.base import BaseEmbedding
import numpy as np

from chenking.processor import Processor
from chenking.chenker import Chenker
from chenking.embedding_client import EmbeddingClient


class ChenkingDocumentProcessor:
    """
    A custom document processor that integrates Chenking validation
    with LlamaIndex document ingestion pipeline.
    """
    
    def __init__(self, 
                 embedding_api_url: str,
                 validation_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the processor with Chenking validation capabilities.
        
        Args:
            embedding_api_url: URL for the embedding API
            validation_config: Optional configuration for document validation
        """
        self.processor = Processor(embedding_api_url, **(validation_config or {}))
        self.validation_stats = {"processed": 0, "valid": 0, "invalid": 0}
    
    def validate_and_process_documents(self, 
                                     documents: List[Dict[str, Any]]) -> List[Document]:
        """
        Validate documents using Chenking and convert valid ones to LlamaIndex Documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of validated LlamaIndex Document objects
        """
        valid_documents = []
        
        for doc in documents:
            try:
                # Use Chenking to validate and process the document
                result = self.processor.process(doc)
                
                self.validation_stats["processed"] += 1
                
                # Check if document passed validation
                if self._is_document_valid(result):
                    # Convert to LlamaIndex Document
                    llama_doc = self._convert_to_llama_document(doc, result)
                    valid_documents.append(llama_doc)
                    self.validation_stats["valid"] += 1
                else:
                    print(f"Document {doc.get('id', 'unknown')} failed validation")
                    self.validation_stats["invalid"] += 1
                    
            except Exception as e:
                print(f"Error processing document {doc.get('id', 'unknown')}: {e}")
                self.validation_stats["invalid"] += 1
        
        return valid_documents
    
    def _is_document_valid(self, chenking_result: Dict[str, Any]) -> bool:
        """
        Determine if a document is valid based on Chenking validation results.
        
        Args:
            chenking_result: Result from Chenking processor
            
        Returns:
            True if document is valid, False otherwise
        """
        if chenking_result.get("processing_info", {}).get("status") == "failed":
            return False
        
        # Check if most validation checks passed
        chenkings = chenking_result.get("chenkings", {})
        successful_checks = sum(1 for check in chenkings.values() 
                              if check.get("status") == "success")
        total_checks = len(chenkings)
        
        # Require at least 80% of checks to pass
        return successful_checks / total_checks >= 0.8 if total_checks > 0 else False
    
    def _convert_to_llama_document(self, 
                                 original_doc: Dict[str, Any],
                                 chenking_result: Dict[str, Any]) -> Document:
        """
        Convert a validated document to LlamaIndex Document format.
        
        Args:
            original_doc: Original document dictionary
            chenking_result: Chenking validation results
            
        Returns:
            LlamaIndex Document object
        """
        # Extract content
        content = original_doc.get("content", "")
        
        # Create metadata including Chenking validation results
        metadata = {
            "doc_id": original_doc.get("id"),
            "title": original_doc.get("title"),
            "author": original_doc.get("author"),
            "format": original_doc.get("format"),
            "chenking_validation": {
                "checks_passed": chenking_result.get("processing_info", {}).get("successful_checks", 0),
                "total_checks": chenking_result.get("processing_info", {}).get("checks_completed", 0),
                "processing_time": chenking_result.get("processing_info", {}).get("total_processing_time", 0),
                "quality_score": self._calculate_quality_score(chenking_result)
            }
        }
        
        # Add original metadata if present
        if "metadata" in original_doc:
            metadata.update(original_doc["metadata"])
        
        return Document(text=content, metadata=metadata)
    
    def _calculate_quality_score(self, chenking_result: Dict[str, Any]) -> float:
        """Calculate a quality score based on Chenking validation results."""
        chenkings = chenking_result.get("chenkings", {})
        
        if not chenkings:
            return 0.0
        
        quality_indicators = []
        
        # Check basic content quality
        basic_check = chenkings.get("basic_check", {}).get("data", {})
        if basic_check.get("is_word_count_valid"):
            quality_indicators.append(1.0)
        
        # Check content structure
        quality_check = chenkings.get("content_quality_check", {}).get("data", {})
        if quality_check:
            metadata_score = quality_check.get("metadata_completeness", {}).get("completeness_score", 0)
            quality_indicators.append(metadata_score)
        
        # Check format support
        format_check = chenkings.get("format_check", {}).get("data", {})
        if format_check.get("is_format_supported"):
            quality_indicators.append(1.0)
        
        return sum(quality_indicators) / len(quality_indicators) if quality_indicators else 0.0
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total = self.validation_stats["processed"]
        return {
            **self.validation_stats,
            "success_rate": self.validation_stats["valid"] / total if total > 0 else 0,
            "failure_rate": self.validation_stats["invalid"] / total if total > 0 else 0
        }


class ChenkingEmbedding(BaseEmbedding):
    """
    Custom embedding class that uses Chenking's EmbeddingClient
    for LlamaIndex integration.
    """
    
    def __init__(self, embedding_api_url: str, **kwargs):
        """
        Initialize with Chenking's EmbeddingClient.
        
        Args:
            embedding_api_url: URL for the embedding API
        """
        super().__init__(**kwargs)
        self.embedding_client = EmbeddingClient(embedding_api_url)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text using Chenking's embedding client.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        # Prepare data for Chenking embedding client
        check_data = {
            "content": text,
            "word_count": len(text.split()),
            "char_count": len(text)
        }
        
        result = self.embedding_client.get_embedding(check_data)
        
        if result.get("status") == "success":
            return result.get("embedding", [])
        else:
            # Fallback to random embedding if API fails
            return np.random.rand(384).tolist()  # Adjust size as needed
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding lists
        """
        return [self._get_text_embedding(text) for text in texts]


def create_chenking_rag_system(documents: List[Dict[str, Any]], 
                              embedding_api_url: str,
                              validation_config: Optional[Dict[str, Any]] = None) -> VectorStoreIndex:
    """
    Create a complete RAG system using Chenking for document validation
    and LlamaIndex for retrieval and generation.
    
    Args:
        documents: List of documents to process
        embedding_api_url: URL for embedding API
        validation_config: Optional validation configuration
        
    Returns:
        Configured VectorStoreIndex ready for querying
    """
    # Initialize Chenking document processor
    doc_processor = ChenkingDocumentProcessor(embedding_api_url, validation_config)
    
    # Validate and convert documents
    print("Validating documents with Chenking...")
    llama_documents = doc_processor.validate_and_process_documents(documents)
    
    # Print validation statistics
    stats = doc_processor.get_validation_stats()
    print(f"Validation complete: {stats['valid']}/{stats['processed']} documents passed")
    print(f"Success rate: {stats['success_rate']:.2%}")
    
    if not llama_documents:
        raise ValueError("No valid documents found after Chenking validation")
    
    # Create custom embedding using Chenking
    chenking_embedding = ChenkingEmbedding(embedding_api_url)
    
    # Configure service context with Chenking embedding
    service_context = ServiceContext.from_defaults(
        embed_model=chenking_embedding,
        node_parser=SimpleNodeParser.from_defaults()
    )
    
    # Create and return the index
    print("Building vector index...")
    index = VectorStoreIndex.from_documents(
        llama_documents, 
        service_context=service_context
    )
    
    print("RAG system ready!")
    return index


# Example usage functions
def example_basic_integration():
    """Basic example of using Chenking with LlamaIndex."""
    
    # Sample documents
    documents = [
        {
            "id": "doc1",
            "content": "This is a well-structured document about artificial intelligence. It contains comprehensive information about machine learning algorithms and their applications in various domains.",
            "title": "AI Overview",
            "author": "AI Researcher",
            "format": "txt"
        },
        {
            "id": "doc2", 
            "content": "Short doc",  # This will likely fail validation
            "title": "Brief Note"
        },
        {
            "id": "doc3",
            "content": "This document discusses the latest developments in natural language processing. It covers transformer architectures, attention mechanisms, and their impact on language understanding tasks.",
            "title": "NLP Advances",
            "author": "NLP Expert",
            "format": "md"
        }
    ]
    
    # Mock embedding API URL (replace with real URL)
    embedding_api_url = "https://api.example.com/embeddings"
    
    # Custom validation configuration
    validation_config = {
        "min_word_count": 15,
        "max_word_count": 10000,
        "supported_formats": ["txt", "md", "html"],
        "required_fields": ["content", "title"]
    }
    
    try:
        # Create RAG system with Chenking validation
        index = create_chenking_rag_system(
            documents, 
            embedding_api_url, 
            validation_config
        )
        
        # Create query engine
        query_engine = index.as_query_engine()
        
        # Example queries
        response = query_engine.query("What is artificial intelligence?")
        print(f"Response: {response}")
        
        return index
        
    except Exception as e:
        print(f"Error creating RAG system: {e}")
        return None


if __name__ == "__main__":
    print("Chenking + LlamaIndex Integration Example")
    print("=" * 50)
    
    # Run basic integration example
    example_basic_integration()
