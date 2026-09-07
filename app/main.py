"""
Job Matcher ML Service - Main FastAPI Application

A local ML service for matching job postings against user profiles using:
- NLP tokenization and lemmatization (spaCy)
- Semantic embeddings (sentence-transformers)
- Cosine similarity scoring
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api import embed, match, health
from app.models.embedder import get_embedding_model
from app.processors.tokenizer import get_text_processor

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler for startup and shutdown events.
    """
    # Startup: Pre-load models
    logger.info("="*50)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info("="*50)
    
    logger.info("Pre-loading embedding model...")
    embedder = get_embedding_model(settings.embedding_model)
    embedder.load()
    logger.info(f"Embedding model loaded: {settings.embedding_model} ({embedder.dimension} dims)")
    
    logger.info("Pre-loading text processor...")
    processor = get_text_processor(settings.spacy_model)
    processor.load()
    logger.info(f"Text processor loaded: {settings.spacy_model}")
    
    logger.info("="*50)
    logger.info(f"Server ready at http://{settings.host}:{settings.port}")
    logger.info("="*50)
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    ## Job Matcher ML Service
    
    A local ML service for intelligent job-profile matching using NLP and embeddings.
    
    ### Features
    - **Tokenization & Lemmatization**: Uses spaCy for text processing
    - **Semantic Embeddings**: sentence-transformers for dense vector representations
    - **Multi-component Scoring**: Skills, experience, semantic, and requirements matching
    - **Recommendations**: Strong Apply, Good Fit, Maybe, or Weak Match
    
    ### Scoring Weights
    - Skill Match: 40%
    - Experience Match: 30%
    - Semantic Match: 20%
    - Requirements Fit: 10%
    
    ### Workflow
    1. POST `/api/v1/embed/profile` - Embed user profile
    2. POST `/api/v1/embed/job` - Embed job posting
    3. POST `/api/v1/match` - Match profile against job
    """,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(embed.router)
app.include_router(embed.similarity_router)
app.include_router(match.router)


@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "embed_profile": "POST /api/v1/embed/profile",
            "embed_job": "POST /api/v1/embed/job",
            "match": "POST /api/v1/match",
            "match_inline": "POST /api/v1/match/inline",
            "find_similar": "GET /api/v1/match/similar-jobs/{profile_id}",
            "similarity": "POST /api/v1/similarity"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
