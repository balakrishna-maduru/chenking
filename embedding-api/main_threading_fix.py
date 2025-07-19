#!/usr/bin/env python3
"""
Test version with PyTorch threading fixes
"""

import os
# Set environment variables before importing torch/transformers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import time
from typing import List, Union
import gc
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Threading Fix Test", version="1.0.0")

# Global model instance
model = None

class EmbeddingRequest(BaseModel):
    input: Union[List[str], str]
    model: str = "sentence-transformers/all-MiniLM-L6-v2"

@app.on_event("startup")
async def startup_event():
    """Initialize the embedding model"""
    global model
    
    model_name = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    device = os.getenv("DEVICE", "cpu")
    
    logger.info(f"Loading model: {model_name} on device: {device}")
    
    try:
        import torch
        torch.set_num_threads(1)  # Force single threading
        
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device=device)
        logger.info("✅ Embedding model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

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
    """Test embeddings endpoint with threading fixes"""
    logger.info(f"Received embedding request: {request}")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Handle both string and list inputs
    inputs = request.input if isinstance(request.input, list) else [request.input]
    logger.info(f"Processing {len(inputs)} inputs")
    
    embeddings_data = []
    
    for i, text in enumerate(inputs):
        logger.info(f"Processing text {i}: {text[:50]}...")
        
        try:
            # Limit text length aggressively
            if len(text) > 256:  # Even smaller limit
                text = text[:256]
                logger.warning(f"Text truncated to 256 characters")
            
            logger.info("About to call model.encode()...")
            
            # Import torch here and ensure single-threaded execution
            import torch
            with torch.no_grad():  # Disable gradient computation
                torch.set_num_threads(1)
                
                # Try to encode with very conservative options
                embedding = model.encode(
                    text, 
                    convert_to_numpy=True, 
                    show_progress_bar=False,
                    batch_size=1,
                    normalize_embeddings=False,
                    device=None  # Use model's device
                )
            
            logger.info(f"model.encode() successful, embedding shape: {embedding.shape}")
            
            # Convert to list
            embedding_list = embedding.tolist()
            
            logger.info(f"Converted to list, length: {len(embedding_list)}")
            
            embeddings_data.append({
                "object": "embedding",
                "embedding": embedding_list,
                "index": i
            })
            
            # Force garbage collection after each embedding
            gc.collect()
            
        except Exception as e:
            logger.error(f"Failed to encode text {i}: {e}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            # Return dummy embedding on error instead of crashing
            dummy_embedding = [0.0] * 384
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
