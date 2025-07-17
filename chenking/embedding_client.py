import requests
import logging
import time
from typing import Dict, Any, Optional


class EmbeddingClient:
    """
    Client for calling embedding APIs with document check data.
    
    Handles API communication, error handling, and response processing
    for generating embeddings from document validation results.
    """

    def __init__(self, api_url: str, timeout: int = 30, max_retries: int = 3):
        """
        Initialize EmbeddingClient.
        
        Args:
            api_url: URL for the embedding API service
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.api_url = api_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_embedding(self, check_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get embedding for given check data with retry logic.
        
        Args:
            check_data: Document check result data
            
        Returns:
            Dictionary containing embedding results or error information
        """
        self.logger.debug(f"Requesting embedding for data: {type(check_data)}")
        
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                
                response = requests.post(
                    self.api_url, 
                    json=check_data,
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                result = response.json()
                request_time = time.time() - start_time
                
                self.logger.debug(f"Embedding request completed in {request_time:.2f}s")
                
                return {
                    "embedding": result.get("embedding"),
                    "vector": result.get("vector"),
                    "request_time": request_time,
                    "status": "success"
                }
                
            except requests.exceptions.Timeout as e:
                self.logger.warning(f"Timeout on attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries:
                    return {
                        "embedding": None, 
                        "vector": None, 
                        "error": f"Timeout after {self.max_retries + 1} attempts: {str(e)}",
                        "status": "timeout"
                    }
                    
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed on attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries:
                    return {
                        "embedding": None, 
                        "vector": None, 
                        "error": f"Request failed after {self.max_retries + 1} attempts: {str(e)}",
                        "status": "error"
                    }
                    
            except Exception as e:
                self.logger.error(f"Unexpected error: {str(e)}")
                return {
                    "embedding": None, 
                    "vector": None, 
                    "error": f"Unexpected error: {str(e)}",
                    "status": "error"
                }
            
            # Wait before retry (exponential backoff)
            if attempt < self.max_retries:
                wait_time = 2 ** attempt
                self.logger.debug(f"Waiting {wait_time}s before retry")
                time.sleep(wait_time)
        
        # This should never be reached
        return {
            "embedding": None, 
            "vector": None, 
            "error": "Max retries exceeded",
            "status": "error"
        }

    def health_check(self) -> bool:
        """
        Check if the embedding API is healthy and responding.
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            # Send a simple health check request
            response = requests.get(
                f"{self.api_url.rstrip('/')}/health",
                timeout=self.timeout
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"Health check failed: {str(e)}")
            return False
