#!/usr/bin/env python3
"""
Minimal test version of the embedding API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import time
from typing import List, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Simple Embedding API Test", version="1.0.0")

# Global model instance
model = None

class EmbeddingRequest(BaseModel):
    input: Union[List[str], str]
    model: str = "test"

@app.on_event("startup")
async def startup_event():
    """Initialize without actually loading the model"""
    global model
    logger.info("Starting up - skipping model loading for now")
    model = "dummy_model"  # Just set a dummy value
    logger.info("✅ Startup complete")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": time.time()
    }

@app.post("/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    """Test embeddings endpoint without actual model"""
    logger.info(f"Received embedding request: {request}")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Handle both string and list inputs
    inputs = request.input if isinstance(request.input, list) else [request.input]
    logger.info(f"Processing {len(inputs)} inputs")
    
    embeddings_data = []
    
    for i, text in enumerate(inputs):
        logger.info(f"Processing text {i}: {text[:50]}...")
        
        # Return dummy embedding instead of using real model
        dummy_embedding = [0.1] * 384  # Standard size for all-MiniLM-L6-v2
        
        embeddings_data.append({
            "object": "embedding",
            "embedding": dummy_embedding,
            "index": i
        })
    
    logger.info(f"Returning {len(embeddings_data)} embeddings")
    
    return {
        "object": "list",
        "data": embeddings_data,
        "model": request.model,
        "usage": {
            "prompt_tokens": sum(len(text.split()) for text in inputs),
            "total_tokens": sum(len(text.split()) for text in inputs)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
