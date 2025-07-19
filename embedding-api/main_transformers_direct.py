#!/usr/bin/env python3
"""
Test version using transformers directly instead of sentence-transformers
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
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Transformers Direct Test", version="1.0.0")

# Global model and tokenizer instances
model = None
tokenizer = None

class EmbeddingRequest(BaseModel):
    input: Union[List[str], str]
    model: str = "sentence-transformers/all-MiniLM-L6-v2"

def mean_pooling(model_output, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging"""
    import torch  # Import torch inside the function
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

@app.on_event("startup")
async def startup_event():
    """Initialize the embedding model using transformers directly"""
    global model, tokenizer
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    device = "cpu"
    
    logger.info(f"Loading model with transformers: {model_name} on device: {device}")
    
    try:
        import torch
        torch.set_num_threads(1)
        
        from transformers import AutoTokenizer, AutoModel
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()  # Set to evaluation mode
        
        logger.info("✅ Embedding model loaded successfully with transformers")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.error(f"Exception traceback: {traceback.format_exc()}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "timestamp": time.time()
    }

@app.post("/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    """Test embeddings endpoint using transformers directly"""
    logger.info(f"Received embedding request: {request}")
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Handle both string and list inputs
    inputs = request.input if isinstance(request.input, list) else [request.input]
    logger.info(f"Processing {len(inputs)} inputs")
    
    embeddings_data = []
    
    for i, text in enumerate(inputs):
        logger.info(f"Processing text {i}: {text[:50]}...")
        
        try:
            # Limit text length
            if len(text) > 256:
                text = text[:256]
                logger.warning(f"Text truncated to 256 characters")
            
            logger.info("About to tokenize...")
            
            import torch
            torch.set_num_threads(1)
            
            # Tokenize text
            encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
            logger.info("Tokenization successful")
            
            logger.info("About to run model forward pass...")
            
            # Compute token embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)
            
            logger.info("Model forward pass successful")
            
            # Perform pooling
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Convert to numpy
            embedding = sentence_embeddings.cpu().numpy()[0]
            embedding_list = embedding.tolist()
            
            logger.info(f"Embedding generated successfully, length: {len(embedding_list)}")
            
            embeddings_data.append({
                "object": "embedding",
                "embedding": embedding_list,
                "index": i
            })
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logger.error(f"Failed to encode text {i}: {e}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            # Return dummy embedding on error
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

# Make torch available globally for mean_pooling function
import torch

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
