#!/usr/bin/env python3
"""
Local Embedding API for Chenking
Provides OpenAI-compatible embeddings API using sentence-transformers
"""

import os
# Set threading environment variables early
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
import time
import logging
from typing import List, Dict, Any, Optional, Union
import redis
import json
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chenking Embedding API",
    description="Local embedding service for Chenking document processing",
    version="1.0.0"
)

# Global model instance
model = None
redis_client = None

class EmbeddingRequest(BaseModel):
    input: Union[List[str], str]
    model: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2"

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: Dict[str, int]

class ChenkingEmbeddingRequest(BaseModel):
    """Chenking-specific embedding request format"""
    word_count: Optional[int] = None
    has_content: Optional[bool] = None
    char_count: Optional[int] = None
    content_summary: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the embedding model and Redis connection"""
    global model, redis_client
    
    model_name = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    device = os.getenv("DEVICE", "cpu")
    
    logger.info(f"Loading model: {model_name} on device: {device}")
    
    try:
        model = SentenceTransformer(model_name, device=device)
        # Set model to evaluation mode and configure threading
        model.eval()
        import torch
        torch.set_num_threads(1)
        logger.info("✅ Embedding model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise
    
    # Initialize Redis if available
    try:
        redis_host = os.getenv("REDIS_HOST", "redis-cache")
        redis_client = redis.Redis(host=redis_host, port=6379, decode_responses=True)
        redis_client.ping()
        logger.info("✅ Redis cache connected")
    except Exception as e:
        logger.warning(f"⚠️ Redis not available: {e}")
        redis_client = None

def get_cache_key(text: str, model_name: str) -> str:
    """Generate cache key for embedding"""
    content = f"{model_name}:{text}"
    return hashlib.md5(content.encode()).hexdigest()

def get_cached_embedding(text: str, model_name: str) -> Optional[List[float]]:
    """Get embedding from cache if available"""
    if not redis_client:
        return None
    
    try:
        cache_key = get_cache_key(text, model_name)
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        logger.warning(f"Cache read error: {e}")
    
    return None

def cache_embedding(text: str, model_name: str, embedding: List[float], ttl: int = 3600):
    """Cache embedding for future use"""
    if not redis_client:
        return
    
    try:
        cache_key = get_cache_key(text, model_name)
        redis_client.setex(cache_key, ttl, json.dumps(embedding))
    except Exception as e:
        logger.warning(f"Cache write error: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "redis_available": redis_client is not None,
        "timestamp": time.time()
    }

@app.get("/models")
async def list_models():
    """List available models (OpenAI-compatible)"""
    model_name = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "chenking"
            }
        ]
    }

@app.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """Create embeddings (OpenAI-compatible endpoint)"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Handle both string and list inputs
    inputs = request.input if isinstance(request.input, list) else [request.input]
    
    embeddings_data = []
    total_tokens = 0
    
    for i, text in enumerate(inputs):
        # Check cache first
        cached_embedding = get_cached_embedding(text, request.model)
        
        if cached_embedding:
            embedding = cached_embedding
            logger.info(f"Cache hit for text length {len(text)}")
        else:
            # Generate embedding
            try:
                start_time = time.time()
                # Add some limits to prevent memory issues
                if len(text) > 512:  # More conservative limit
                    text = text[:512]
                    logger.warning(f"Text truncated to 512 characters")
                
                # Import torch and ensure single-threaded execution
                import torch
                torch.set_num_threads(1)
                
                # Use very conservative encoding parameters
                with torch.no_grad():
                    embedding = model.encode(
                        text, 
                        convert_to_tensor=False,  # Don't use tensors
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=1,
                        normalize_embeddings=False,
                        device='cpu'  # Force CPU
                    ).tolist()
                
                processing_time = time.time() - start_time
                
                # Cache the result
                cache_embedding(text, request.model, embedding)
                
                logger.info(f"Generated embedding in {processing_time:.3f}s for text length {len(text)}")
            except Exception as e:
                logger.error(f"Failed to generate embedding: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {str(e)}")
        
        embeddings_data.append({
            "object": "embedding",
            "embedding": embedding,
            "index": i
        })
        
        total_tokens += len(text.split())
    
    return EmbeddingResponse(
        data=embeddings_data,
        model=request.model,
        usage={
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens
        }
    )

@app.post("/chenking/embedding")
async def chenking_embedding(request: ChenkingEmbeddingRequest):
    """Chenking-specific embedding endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    # Create text representation from Chenking data
    text_parts = []
    if request.word_count is not None:
        text_parts.append(f"word_count:{request.word_count}")
    if request.has_content is not None:
        text_parts.append(f"has_content:{request.has_content}")
    if request.char_count is not None:
        text_parts.append(f"char_count:{request.char_count}")
    if request.content_summary:
        text_parts.append(f"summary:{request.content_summary}")
    
    if not text_parts:
        raise HTTPException(status_code=400, detail="No content provided for embedding")
    
    text = " ".join(text_parts)
    
    # Generate embedding
    try:
        # Import torch and ensure single-threaded execution
        import torch
        torch.set_num_threads(1)
        
        with torch.no_grad():
            embedding = model.encode(
                text, 
                convert_to_tensor=False,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=1,
                normalize_embeddings=False,
                device='cpu'
            )
        
        vector = embedding.tolist()
        
        processing_time = time.time() - start_time
        
        return {
            "embedding": vector,
            "vector": vector,  # Chenking expects both fields
            "status": "success",
            "request_time": processing_time,
            "model": os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
            "dimensions": len(vector)
        }
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Chenking Local Embedding API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "embeddings": "/embeddings",
            "chenking_embedding": "/chenking/embedding"
        },
        "model": os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
        "status": "ready" if model else "loading"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
