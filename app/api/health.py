"""
Health check endpoint.
"""
from fastapi import APIRouter
import time
from datetime import datetime

from app.config import settings
from app.models.schemas import HealthResponse
from app.models.embedder import get_embedding_model
from app.processors.tokenizer import get_text_processor
from app.storage.memory import get_storage

router = APIRouter(tags=["Health"])

# Track startup time
_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns status of all components:
    - Embedding model loaded status
    - spaCy model loaded status
    - Storage backend
    - Uptime
    """
    embedder = get_embedding_model()
    processor = get_text_processor()
    storage = get_storage()
    
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        models_loaded={
            "embedding_model": embedder.is_loaded(),
            "spacy_model": processor.is_loaded()
        },
        storage_backend=settings.storage_backend,
        uptime_seconds=round(time.time() - _start_time, 2)
    )


@router.get("/ready")
async def readiness_check():
    """
    Readiness check - ensures models are loaded.
    """
    embedder = get_embedding_model()
    
    if not embedder.is_loaded():
        # Try to load
        embedder.load()
        
    return {"ready": True, "models_loaded": embedder.is_loaded()}


@router.get("/stats")
async def get_stats():
    """
    Get storage and model statistics.
    """
    storage = get_storage()
    embedder = get_embedding_model()
    
    stats = storage.stats()
    stats["embedding_model"] = embedder.model_name
    stats["embedding_dimension"] = embedder.dimension if embedder.is_loaded() else 0
    
    return stats
