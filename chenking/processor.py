import logging
import time
from typing import Dict, Any, Optional
from chenking.chenker import Chenker
from chenking.embedding_client import EmbeddingClient


class Processor:
    """
    Main class that combines Chenker and EmbeddingClient to process documents.
    
    This class orchestrates the document validation and embedding generation process,
    providing a unified interface for comprehensive document analysis.
    """

    def __init__(self, api_url: str, **kwargs):
        """
        Initialize Processor with embedding API URL and optional checker config.
        
        Args:
            api_url: URL for the embedding API service
            **kwargs: Configuration parameters passed to Chenker
        """
        self.api_url = api_url
        self.checker = Chenker(**kwargs)
        self.embedder = EmbeddingClient(api_url)
        self.logger = logging.getLogger(self.__class__.__name__)

    def process(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a document through validation checks and embedding generation.
        
        Args:
            document: Document data with 'content' and optional metadata
            
        Returns:
            Processed document with validation results and embeddings
            
        Raises:
            ValueError: If document is invalid or missing required fields
        """
        # Validate input document first
        if not isinstance(document, dict):
            raise ValueError("Document must be a dictionary")
        
        if not document.get("content"):
            raise ValueError("Document must contain 'content' field")
            
        self.logger.info(f"Processing document: {document.get('id', 'unknown')}")

        start_time = time.time()
        
        # Prepare output structure
        output = {
            "id": document.get("id"),
            "content": document.get("content"),
            "metadata": document.get("metadata", {}),
            "chenkings": {},
            "processing_info": {
                "timestamp": time.time(),
                "processor_version": "1.0.0"
            }
        }

        try:
            # Run document validation checks
            self.logger.debug("Running document validation checks")
            check_results = self.checker.run_checks(document)

            # Process each check result with embeddings
            for check_name, result in check_results.items():
                self.logger.debug(f"Processing embeddings for check: {check_name}")
                
                if result["status"] == "success":
                    embedding_result = self.embedder.get_embedding(result["data"])
                    output["chenkings"][check_name] = {
                        "data": result["data"],
                        "status": result["status"],
                        "execution_time": result["execution_time"],
                        "embedding": embedding_result.get("embedding"),
                        "vector": embedding_result.get("vector"),
                        "embedding_error": embedding_result.get("error")
                    }
                else:
                    # Handle failed checks
                    output["chenkings"][check_name] = {
                        "data": result["data"],
                        "status": result["status"],
                        "error": result.get("error"),
                        "execution_time": result["execution_time"],
                        "embedding": None,
                        "vector": None,
                        "embedding_error": "Check failed, no embedding generated"
                    }

            # Add processing summary
            processing_time = time.time() - start_time
            output["processing_info"]["total_processing_time"] = processing_time
            output["processing_info"]["checks_completed"] = len(check_results)
            output["processing_info"]["successful_checks"] = sum(
                1 for result in check_results.values() 
                if result["status"] == "success"
            )
            
            self.logger.info(f"Document processing completed in {processing_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            output["processing_info"]["error"] = str(e)
            output["processing_info"]["status"] = "failed"
            
        return output

    def process_batch(self, documents: list) -> list:
        """
        Process multiple documents in batch.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of processed documents
        """
        self.logger.info(f"Processing batch of {len(documents)} documents")
        
        results = []
        for i, document in enumerate(documents):
            try:
                self.logger.debug(f"Processing document {i+1}/{len(documents)}")
                result = self.process(document)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process document {i+1}: {str(e)}")
                # Add error document to results
                error_result = {
                    "id": document.get("id", f"doc_{i}"),
                    "content": document.get("content", ""),
                    "metadata": document.get("metadata", {}),
                    "chenkings": {},
                    "processing_info": {
                        "timestamp": time.time(),
                        "error": str(e),
                        "status": "failed"
                    }
                }
                results.append(error_result)
        
        return results

    def get_processing_stats(self, processed_documents: list) -> Dict[str, Any]:
        """
        Generate statistics from processed documents.
        
        Args:
            processed_documents: List of processed document results
            
        Returns:
            Statistics dictionary
        """
        if not processed_documents:
            return {"total_documents": 0}
        
        total_docs = len(processed_documents)
        successful_docs = sum(
            1 for doc in processed_documents 
            if doc.get("processing_info", {}).get("status") != "failed"
        )
        
        total_processing_time = sum(
            doc.get("processing_info", {}).get("total_processing_time", 0)
            for doc in processed_documents
        )
        
        avg_processing_time = total_processing_time / total_docs if total_docs > 0 else 0
        
        # Collect check statistics
        check_stats = {}
        for doc in processed_documents:
            for check_name, check_result in doc.get("chenkings", {}).items():
                if check_name not in check_stats:
                    check_stats[check_name] = {"success": 0, "failed": 0}
                
                if check_result.get("status") == "success":
                    check_stats[check_name]["success"] += 1
                else:
                    check_stats[check_name]["failed"] += 1
        
        return {
            "total_documents": total_docs,
            "successful_documents": successful_docs,
            "failed_documents": total_docs - successful_docs,
            "success_rate": successful_docs / total_docs if total_docs > 0 else 0,
            "total_processing_time": total_processing_time,
            "average_processing_time": avg_processing_time,
            "check_statistics": check_stats
        }
