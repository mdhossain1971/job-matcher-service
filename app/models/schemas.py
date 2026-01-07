"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# ============================================
# Enums
# ============================================

class Recommendation(str, Enum):
    STRONG_APPLY = "strong_apply"
    GOOD_FIT = "good_fit"
    MAYBE = "maybe"
    WEAK_MATCH = "weak_match"


class Proficiency(str, Enum):
    EXPERT = "expert"
    ADVANCED = "advanced"
    INTERMEDIATE = "intermediate"
    BEGINNER = "beginner"


# ============================================
# Input Models
# ============================================

class SkillInput(BaseModel):
    """Skill with optional metadata"""
    name: str
    category: Optional[str] = None
    proficiency: Optional[str] = None
    years: Optional[int] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Java",
                "category": "LANGUAGE",
                "proficiency": "EXPERT",
                "years": 15
            }
        }


class ExperienceInput(BaseModel):
    """Work experience entry"""
    company_name: str
    title: str
    description: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    is_current: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "company_name": "AppDynamics",
                "title": "Staff Software Engineer",
                "description": "Led design and development of microservices...",
                "start_date": "2020-03",
                "end_date": "2025-04",
                "is_current": False
            }
        }


class EducationInput(BaseModel):
    """Education entry"""
    institution: str
    degree: str
    field_of_study: Optional[str] = None
    end_date: Optional[str] = None


class ProfileEmbedRequest(BaseModel):
    """Request to embed a user profile"""
    profile_id: int
    profile_name: Optional[str] = None
    summary: Optional[str] = None
    skills: List[SkillInput] = []
    experiences: List[ExperienceInput] = []
    education: List[EducationInput] = []
    certifications: List[str] = []
    resume_text: Optional[str] = None
    years_of_experience: Optional[int] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "profile_id": 1,
                "profile_name": "Software Engineer",
                "summary": "15+ years of experience in Application Architecture...",
                "skills": [
                    {"name": "Java", "years": 15, "proficiency": "EXPERT"},
                    {"name": "Spring Boot", "years": 10, "proficiency": "EXPERT"}
                ],
                "experiences": [
                    {
                        "company_name": "AppDynamics",
                        "title": "Staff Software Engineer",
                        "description": "Led team of 5 engineers..."
                    }
                ],
                "years_of_experience": 15
            }
        }


class JobEmbedRequest(BaseModel):
    """Request to embed a job posting"""
    job_id: int
    title: str
    company_name: Optional[str] = None
    description: Optional[str] = None
    requirements: Optional[str] = None
    skills_required: Optional[str] = None
    location: Optional[str] = None
    remote_type: Optional[str] = None
    experience_level: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": 123,
                "title": "Senior Software Engineer",
                "company_name": "Walmart",
                "description": "We are looking for a Senior Software Engineer...",
                "requirements": "5+ years of experience in Java, Spring Boot...",
                "skills_required": "Java, Spring Boot, Kafka, Kubernetes",
                "location": "Sunnyvale, CA",
                "remote_type": "HYBRID"
            }
        }


class MatchRequest(BaseModel):
    """Request to match profile against job"""
    profile_id: int
    job_id: int


class BatchMatchRequest(BaseModel):
    """Request to match profile against multiple jobs"""
    profile_id: int
    job_ids: List[int]


class InlineMatchRequest(BaseModel):
    """Request to match profile against job inline (without storing)"""
    profile: ProfileEmbedRequest
    job: JobEmbedRequest


# ============================================
# Output Models
# ============================================

class ScoreBreakdown(BaseModel):
    """Breakdown of individual score components"""
    skill_match: float = Field(..., description="Skill matching score (0-100)")
    experience_match: float = Field(..., description="Experience relevance score (0-100)")
    semantic_match: float = Field(..., description="Overall semantic similarity (0-100)")
    requirements_fit: float = Field(..., description="Requirements alignment score (0-100)")


class MatchResult(BaseModel):
    """Result of matching profile against job"""
    profile_id: int
    job_id: int
    score: int = Field(..., ge=0, le=100, description="Final weighted score")
    recommendation: Recommendation
    breakdown: ScoreBreakdown
    matched_skills: List[str] = Field(default_factory=list, description="Skills found in both profile and job")
    missing_skills: List[str] = Field(default_factory=list, description="Required skills not in profile")
    strengths: List[str] = Field(default_factory=list, description="Key matching points")
    gaps: List[str] = Field(default_factory=list, description="Areas to address")
    processing_time_ms: Optional[int] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "profile_id": 1,
                "job_id": 123,
                "score": 87,
                "recommendation": "strong_apply",
                "breakdown": {
                    "skill_match": 92,
                    "experience_match": 85,
                    "semantic_match": 82,
                    "requirements_fit": 78
                },
                "matched_skills": ["Java", "Spring Boot", "Kafka"],
                "missing_skills": ["Kubernetes", "MongoDB"],
                "strengths": ["15+ years experience", "Team leadership"],
                "gaps": ["NoSQL database experience"]
            }
        }


class EmbedResponse(BaseModel):
    """Response after embedding"""
    id: int
    type: str  # "profile" or "job"
    chunks_created: int
    skills_extracted: Optional[List[str]] = None
    processing_time_ms: int
    success: bool = True
    message: Optional[str] = None


class BatchMatchResult(BaseModel):
    """Result of batch matching"""
    profile_id: int
    total_jobs: int
    processed: int
    results: List[MatchResult]
    processing_time_ms: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    models_loaded: Dict[str, bool]
    storage_backend: str
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
