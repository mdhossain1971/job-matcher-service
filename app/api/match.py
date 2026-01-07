"""
API endpoints for matching profiles against jobs.
"""
from fastapi import APIRouter, HTTPException
from typing import List
import time
import logging

from app.models.schemas import (
    MatchRequest, BatchMatchRequest, MatchResult, BatchMatchResult,
    ProfileEmbedRequest, JobEmbedRequest, InlineMatchRequest
)
from app.processors.matcher import get_matching_engine
from app.storage.memory import get_storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/match", tags=["Matching"])


@router.post("", response_model=MatchResult)
async def match_profile_job(request: MatchRequest):
    """
    Match a profile against a job.
    
    Both profile and job must be embedded first using the /embed endpoints.
    
    Returns:
    - Final weighted score (0-100)
    - Recommendation (strong_apply, good_fit, maybe, weak_match)
    - Score breakdown by component
    - Matched and missing skills
    - Strengths and gaps analysis
    """
    storage = get_storage()
    engine = get_matching_engine()
    
    # Get stored embeddings
    profile = storage.get_profile(request.profile_id)
    if not profile:
        raise HTTPException(
            status_code=404, 
            detail=f"Profile {request.profile_id} not found. Embed it first using POST /api/v1/embed/profile"
        )
        
    job = storage.get_job(request.job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {request.job_id} not found. Embed it first using POST /api/v1/embed/job"
        )
    
    # Compute match
    result = engine.match(profile, job)
    
    return result


@router.post("/batch", response_model=BatchMatchResult)
async def match_batch(request: BatchMatchRequest):
    """
    Match a profile against multiple jobs.
    
    Useful for scoring many jobs at once.
    """
    start_time = time.time()
    
    storage = get_storage()
    engine = get_matching_engine()
    
    # Get profile
    profile = storage.get_profile(request.profile_id)
    if not profile:
        raise HTTPException(
            status_code=404,
            detail=f"Profile {request.profile_id} not found"
        )
    
    results = []
    for job_id in request.job_ids:
        job = storage.get_job(job_id)
        if job:
            result = engine.match(profile, job)
            results.append(result)
        else:
            logger.warning(f"Job {job_id} not found, skipping")
            
    elapsed = int((time.time() - start_time) * 1000)
    
    return BatchMatchResult(
        profile_id=request.profile_id,
        total_jobs=len(request.job_ids),
        processed=len(results),
        results=results,
        processing_time_ms=elapsed
    )


@router.post("/inline", response_model=MatchResult)
async def match_inline(request: InlineMatchRequest):
    """
    Match profile against job without storing.
    
    This embeds both on-the-fly and returns the match result.
    Useful for one-off matching without persisting embeddings.
    """
    engine = get_matching_engine()
    
    profile = request.profile
    job = request.job
    
    # Convert to dicts
    profile_data = {
        "profile_id": profile.profile_id,
        "profile_name": profile.profile_name,
        "summary": profile.summary,
        "skills": [s.model_dump() for s in profile.skills],
        "experiences": [e.model_dump() for e in profile.experiences],
        "education": [e.model_dump() for e in profile.education],
        "certifications": profile.certifications,
        "resume_text": profile.resume_text,
        "years_of_experience": profile.years_of_experience
    }
    
    job_data = {
        "job_id": job.job_id,
        "title": job.title,
        "company_name": job.company_name,
        "description": job.description,
        "requirements": job.requirements,
        "skills_required": job.skills_required,
        "location": job.location,
        "remote_type": job.remote_type,
        "experience_level": job.experience_level
    }
    
    # Embed on-the-fly
    profile_emb = engine.embed_profile(profile_data)
    job_emb = engine.embed_job(job_data)
    
    # Match
    result = engine.match(profile_emb, job_emb)
    
    return result


@router.get("/similar-jobs/{profile_id}")
async def find_similar_jobs(profile_id: int, limit: int = 10):
    """
    Find top N jobs that match a profile.
    
    Scores all stored jobs and returns the best matches.
    """
    storage = get_storage()
    engine = get_matching_engine()
    
    # Get profile
    profile = storage.get_profile(profile_id)
    if not profile:
        raise HTTPException(
            status_code=404,
            detail=f"Profile {profile_id} not found"
        )
    
    # Get all jobs and score them
    jobs = storage.get_all_jobs()
    if not jobs:
        return {"profile_id": profile_id, "jobs": [], "message": "No jobs stored"}
    
    results = []
    for job in jobs:
        result = engine.match(profile, job)
        results.append(result)
    
    # Sort by score descending
    results.sort(key=lambda r: r.score, reverse=True)
    
    # Return top N
    return {
        "profile_id": profile_id,
        "total_jobs": len(jobs),
        "top_matches": results[:limit]
    }
