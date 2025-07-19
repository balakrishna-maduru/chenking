import requests
import logging
import time
from typing import Dict, Any, Optional


class EmbeddingClient:
    """
    Client for calling embedding APIs with document check data.
    
    Handles API communication, error handling, and response processing
    for generating embeddings from document validation results.
    Updated to work with the local Chenking embedding API.
    """

    def __init__(self, api_url: str = "http://localhost:8002/chenking/embedding", 
                 timeout: int = 30, max_retries: int = 3):
        """
        Initialize EmbeddingClient.
        
        Args:
            api_url: URL for the embedding API service (defaults to local API)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.api_url = api_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Extract base URL for health checks
        if "/chenking/embedding" in api_url:
            self.base_url = api_url.replace("/chenking/embedding", "")
        else:
            self.base_url = api_url.rstrip('/')
            
        self.logger.info(f"EmbeddingClient initialized with API URL: {self.api_url}")

    def get_embedding(self, check_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get embedding for given check data with retry logic.
        
        Args:
            check_data: Document check result data
            
        Returns:
            Dictionary containing embedding results or error information
        """
        self.logger.debug(f"Requesting embedding for data: {type(check_data)}")
        
        # Transform check_data to match Chenking API format
        embedding_request = self._prepare_embedding_request(check_data)
        
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                
                response = requests.post(
                    self.api_url, 
                    json=embedding_request,
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
                    "status": "success",
                    "model": result.get("model"),
                    "dimensions": result.get("dimensions")
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
            # Send a simple health check request to the base URL
            health_url = f"{self.base_url}/health"
            self.logger.debug(f"Checking health at: {health_url}")
            
            response = requests.get(
                health_url,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                health_data = response.json()
                model_loaded = health_data.get("model_loaded", False)
                self.logger.debug(f"Health check successful, model loaded: {model_loaded}")
                return model_loaded
            else:
                self.logger.warning(f"Health check failed with status: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.warning(f"Health check failed: {str(e)}")
            return False

    def _prepare_embedding_request(self, check_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform check data into format expected by Chenking embedding API.
        
        Args:
            check_data: Raw check result data
            
        Returns:
            Dictionary formatted for Chenking API request
        """
        # Extract relevant information from check data
        word_count = check_data.get("word_count", 0)
        char_count = check_data.get("char_count", 0)
        has_content = check_data.get("has_content", False)
        
        # Create content summary from available data
        content_summary = ""
        if check_data.get("content"):
            # Use first 200 characters as summary
            content_summary = str(check_data["content"])[:200]
        elif check_data.get("summary"):
            content_summary = str(check_data["summary"])
        else:
            # Create summary from check data itself
            content_parts = []
            for key, value in check_data.items():
                if isinstance(value, (str, int, float, bool)) and key != "content":
                    content_parts.append(f"{key}:{value}")
            content_summary = " ".join(content_parts[:5])  # Limit to first 5 fields
        
        return {
            "word_count": word_count,
            "char_count": char_count,
            "has_content": has_content,
            "content_summary": content_summary
        }
