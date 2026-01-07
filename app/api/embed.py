"""
API endpoints for embedding profiles and jobs.
"""
from fastapi import APIRouter, HTTPException
import time
import logging

from app.models.schemas import (
    ProfileEmbedRequest, JobEmbedRequest, EmbedResponse, ErrorResponse
)
from app.processors.matcher import get_matching_engine
from app.storage.memory import get_storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/embed", tags=["Embedding"])


@router.post("/profile", response_model=EmbedResponse)
async def embed_profile(request: ProfileEmbedRequest):
    """
    Embed a user profile and store in vector store.
    
    This tokenizes, chunks, and creates embeddings for:
    - Skills (with years and proficiency)
    - Work experience
    - Education and certifications
    - Professional summary
    - Full profile text
    """
    start_time = time.time()
    
    try:
        engine = get_matching_engine()
        storage = get_storage()
        
        # Convert request to dict for processing
        profile_data = {
            "profile_id": request.profile_id,
            "profile_name": request.profile_name,
            "summary": request.summary,
            "skills": [s.model_dump() for s in request.skills],
            "experiences": [e.model_dump() for e in request.experiences],
            "education": [e.model_dump() for e in request.education],
            "certifications": request.certifications,
            "resume_text": request.resume_text,
            "years_of_experience": request.years_of_experience
        }
        
        # Create embeddings
        embeddings = engine.embed_profile(profile_data)
        
        # Store
        storage.store_profile(embeddings)
        
        elapsed = int((time.time() - start_time) * 1000)
        
        return EmbedResponse(
            id=request.profile_id,
            type="profile",
            chunks_created=len(embeddings.embeddings),
            skills_extracted=embeddings.chunks.skills_list,
            processing_time_ms=elapsed,
            success=True,
            message=f"Profile embedded successfully with {len(embeddings.chunks.skills_list)} skills"
        )
        
    except Exception as e:
        logger.error(f"Error embedding profile: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/job", response_model=EmbedResponse)
async def embed_job(request: JobEmbedRequest):
    """
    Embed a job posting and store in vector store.
    
    This extracts, chunks, and creates embeddings for:
    - Required skills (extracted from description and requirements)
    - Responsibilities
    - Qualifications
    - Full job text
    """
    start_time = time.time()
    
    try:
        engine = get_matching_engine()
        storage = get_storage()
        
        # Convert request to dict for processing
        job_data = {
            "job_id": request.job_id,
            "title": request.title,
            "company_name": request.company_name,
            "description": request.description,
            "requirements": request.requirements,
            "skills_required": request.skills_required,
            "location": request.location,
            "remote_type": request.remote_type,
            "experience_level": request.experience_level
        }
        
        # Create embeddings
        embeddings = engine.embed_job(job_data)
        
        # Store
        storage.store_job(embeddings)
        
        elapsed = int((time.time() - start_time) * 1000)
        
        return EmbedResponse(
            id=request.job_id,
            type="job",
            chunks_created=len(embeddings.embeddings),
            skills_extracted=embeddings.chunks.required_skills_list,
            processing_time_ms=elapsed,
            success=True,
            message=f"Job embedded successfully with {len(embeddings.chunks.required_skills_list)} skills extracted"
        )
        
    except Exception as e:
        logger.error(f"Error embedding job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/profile/{profile_id}")
async def delete_profile(profile_id: int):
    """Delete profile embeddings"""
    storage = get_storage()
    if storage.delete_profile(profile_id):
        return {"success": True, "message": f"Profile {profile_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")


@router.delete("/job/{job_id}")
async def delete_job(job_id: int):
    """Delete job embeddings"""
    storage = get_storage()
    if storage.delete_job(job_id):
        return {"success": True, "message": f"Job {job_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
