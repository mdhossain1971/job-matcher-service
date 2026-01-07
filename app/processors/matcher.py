"""
Matching and scoring engine for profile-job matching.

Scoring Strategy:
- Skill Match (50%): Most important - must have matching skills
- Experience Match (25%): Relevant work history
- Semantic Match (15%): Overall text similarity
- Requirements Fit (10%): Education, certifications

Penalties:
- Zero skill matches: Cap final score at 45
- Low skill match (<30%): Reduce experience score by 50%
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import logging
import time

from app.config import settings
from app.models.embedder import get_embedding_model, EmbeddingModel
from app.processors.chunker import ProfileChunks, JobChunks, get_chunker
from app.models.schemas import MatchResult, ScoreBreakdown, Recommendation

# Create dedicated logger for scoring
logger = logging.getLogger("scoring")


@dataclass
class ProfileEmbeddings:
    """Stored embeddings for a profile"""
    profile_id: int
    chunks: ProfileChunks
    embeddings: Dict[str, np.ndarray]  # chunk_name -> embedding


@dataclass
class JobEmbeddings:
    """Stored embeddings for a job"""
    job_id: int
    chunks: JobChunks
    embeddings: Dict[str, np.ndarray]  # chunk_name -> embedding


class MatchingEngine:
    """
    Core matching engine that computes similarity scores between profiles and jobs.
    """
    
    def __init__(self):
        self.embedder = get_embedding_model(settings.embedding_model)
        self.chunker = get_chunker()
        
        # Scoring weights from config
        self.weights = {
            "skill_match": settings.weight_skill_match,
            "experience_match": settings.weight_experience_match,
            "semantic_match": settings.weight_semantic_match,
            "requirements_fit": settings.weight_requirements_fit
        }
        
        logger.info(f"MatchingEngine initialized with weights: {self.weights}")
        
    def embed_profile(self, profile_data: dict) -> ProfileEmbeddings:
        """Create embeddings for a profile."""
        start_time = time.time()
        
        # Chunk the profile
        chunks = self.chunker.chunk_profile(profile_data)
        
        # Ensure model is loaded
        if not self.embedder.is_loaded():
            self.embedder.load()
            
        # Create embeddings for each chunk
        embeddings = self.embedder.encode_chunks(chunks.to_dict())
        
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Embedded profile {chunks.profile_id} in {elapsed:.0f}ms | Skills: {len(chunks.skills_list)}")
        
        if settings.log_embeddings:
            logger.debug(f"Profile embeddings shapes: {[(k, v.shape) for k, v in embeddings.items()]}")
        
        return ProfileEmbeddings(
            profile_id=chunks.profile_id,
            chunks=chunks,
            embeddings=embeddings
        )
    
    def embed_job(self, job_data: dict) -> JobEmbeddings:
        """Create embeddings for a job."""
        start_time = time.time()
        
        # Chunk the job
        chunks = self.chunker.chunk_job(job_data)
        
        # Ensure model is loaded
        if not self.embedder.is_loaded():
            self.embedder.load()
            
        # Create embeddings for each chunk
        embeddings = self.embedder.encode_chunks(chunks.to_dict())
        
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Embedded job {chunks.job_id} in {elapsed:.0f}ms | Required skills: {chunks.required_skills_list}")
        
        if settings.log_embeddings:
            logger.debug(f"Job embeddings shapes: {[(k, v.shape) for k, v in embeddings.items()]}")
        
        return JobEmbeddings(
            job_id=chunks.job_id,
            chunks=chunks,
            embeddings=embeddings
        )
    
    def match(
        self, 
        profile: ProfileEmbeddings, 
        job: JobEmbeddings
    ) -> MatchResult:
        """
        Match a profile against a job and compute scores.
        """
        start_time = time.time()
        job_title = job.chunks.full_text[:50] if job.chunks.full_text else f"Job {job.job_id}"
        
        if settings.log_scoring_details:
            logger.info(f"=" * 60)
            logger.info(f"MATCHING: Profile {profile.profile_id} vs Job {job.job_id}")
            logger.info(f"=" * 60)
        
        # 1. Skill Match (50%)
        skill_score, matched_skills, missing_skills, skill_details = self._compute_skill_match(
            profile.chunks, job.chunks, 
            profile.embeddings.get("skills"), 
            job.embeddings.get("required_skills")
        )
        
        # 2. Experience Match (25%)
        experience_score, exp_analysis = self._compute_experience_match(
            profile.chunks, job.chunks,
            profile.embeddings.get("experience"),
            job.embeddings.get("responsibilities"),
            skill_score  # Pass skill score for penalty calculation
        )
        
        # 3. Semantic Match (15%)
        semantic_score = self._compute_semantic_match(
            profile.embeddings.get("full"),
            job.embeddings.get("full")
        )
        
        # 4. Requirements Fit (10%)
        requirements_score, req_analysis = self._compute_requirements_fit(
            profile.chunks, job.chunks,
            profile.embeddings.get("education"),
            job.embeddings.get("qualifications")
        )
        
        # Log component scores
        if settings.log_scoring_details:
            logger.info(f"COMPONENT SCORES:")
            logger.info(f"  Skill Match:       {skill_score:5.1f} x {self.weights['skill_match']:.0%} = {skill_score * self.weights['skill_match']:5.1f}")
            logger.info(f"  Experience Match:  {experience_score:5.1f} x {self.weights['experience_match']:.0%} = {experience_score * self.weights['experience_match']:5.1f}")
            logger.info(f"  Semantic Match:    {semantic_score:5.1f} x {self.weights['semantic_match']:.0%} = {semantic_score * self.weights['semantic_match']:5.1f}")
            logger.info(f"  Requirements Fit:  {requirements_score:5.1f} x {self.weights['requirements_fit']:.0%} = {requirements_score * self.weights['requirements_fit']:5.1f}")
        
        # Compute weighted final score
        final_score = (
            skill_score * self.weights["skill_match"] +
            experience_score * self.weights["experience_match"] +
            semantic_score * self.weights["semantic_match"] +
            requirements_score * self.weights["requirements_fit"]
        )
        
        if settings.log_scoring_details:
            logger.info(f"  " + "-" * 40)
            logger.info(f"  Weighted Sum:      {final_score:5.1f}")
        
        # PENALTY: Cap score if NO skills match
        if len(matched_skills) == 0 and len(job.chunks.required_skills_list) > 0:
            cap = settings.zero_skill_match_cap
            if final_score > cap:
                logger.warning(f"  PENALTY: Zero skill match -> Capping {final_score:.1f} to {cap}")
                final_score = cap
        
        # Clamp to 0-100
        final_score = int(round(max(0, min(100, final_score))))
        
        # Determine recommendation
        recommendation = self._get_recommendation(final_score)
        
        if settings.log_scoring_details:
            logger.info(f"  FINAL SCORE:       {final_score} ({recommendation.value})")
            logger.info(f"=" * 60)
        
        # Generate strengths and gaps
        strengths, gaps = self._generate_insights(
            skill_score, experience_score, semantic_score, requirements_score,
            matched_skills, missing_skills, exp_analysis, req_analysis,
            profile.chunks, job.chunks
        )
        
        elapsed = int((time.time() - start_time) * 1000)
        
        return MatchResult(
            profile_id=profile.profile_id,
            job_id=job.job_id,
            score=final_score,
            recommendation=recommendation,
            breakdown=ScoreBreakdown(
                skill_match=round(skill_score, 1),
                experience_match=round(experience_score, 1),
                semantic_match=round(semantic_score, 1),
                requirements_fit=round(requirements_score, 1)
            ),
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            strengths=strengths,
            gaps=gaps,
            processing_time_ms=elapsed
        )
    
    def _compute_skill_match(
        self,
        profile_chunks: ProfileChunks,
        job_chunks: JobChunks,
        profile_skill_emb: Optional[np.ndarray],
        job_skill_emb: Optional[np.ndarray]
    ) -> Tuple[float, List[str], List[str], Dict]:
        """
        Compute skill match score.
        
        Strategy:
        - 70% weight on exact matches
        - 30% weight on semantic similarity
        - If NO exact matches, cap at 50% (semantic alone can't save you)
        """
        details = {}
        
        profile_skills = set(s.lower() for s in profile_chunks.skills_list)
        job_skills = set(s.lower() for s in job_chunks.required_skills_list)
        
        logger.debug(f"Profile skills ({len(profile_skills)}): {sorted(profile_skills)[:10]}...")
        logger.debug(f"Job requires ({len(job_skills)}): {sorted(job_skills)}")
        
        if not job_skills:
            # No skills specified in job, give neutral score
            logger.debug("No skills required by job, returning neutral 70")
            return 70.0, [], [], {"note": "No skills required"}
            
        # Exact matches
        exact_matches = profile_skills & job_skills
        missing = job_skills - profile_skills
        
        # Compute exact match ratio (0 to 1)
        exact_ratio = len(exact_matches) / len(job_skills)
        details["exact_matches"] = len(exact_matches)
        details["total_required"] = len(job_skills)
        details["exact_ratio"] = exact_ratio
        
        # Semantic similarity for skills text (for partial/related matches)
        semantic_sim = 0.3  # Default low
        if profile_skill_emb is not None and job_skill_emb is not None:
            if np.any(profile_skill_emb) and np.any(job_skill_emb):
                sim = self.embedder.similarity(profile_skill_emb, job_skill_emb)
                # Convert from [-1, 1] to [0, 1]
                semantic_sim = (sim + 1) / 2
        details["semantic_sim"] = semantic_sim
        
        # Scoring strategy
        if len(exact_matches) == 0:
            # NO exact skill matches - poor fit regardless of semantic similarity
            # Cap at 50 max (can't be a good fit without matching skills)
            score = semantic_sim * 50
            details["scoring_method"] = "zero_match_penalty"
            logger.debug(f"Zero skill matches -> score capped: {semantic_sim:.2f} x 50 = {score:.1f}")
        else:
            # Have some matches - blend exact (70%) and semantic bonus (30%)
            exact_score = exact_ratio * 100  # 0-100 based on match %
            semantic_bonus = semantic_sim * 30  # Up to 30 bonus from semantic
            score = exact_score * 0.7 + semantic_bonus
            details["scoring_method"] = "blended"
            details["exact_score"] = exact_score
            details["semantic_bonus"] = semantic_bonus
            logger.debug(f"Skill score: ({exact_ratio:.2f} x 100) x 0.7 + ({semantic_sim:.2f} x 30) = {score:.1f}")
        
        if settings.log_scoring_details:
            logger.info(f"  SKILL MATCH: {len(exact_matches)}/{len(job_skills)} exact = {score:.1f}")
            logger.info(f"    Matched: {sorted(exact_matches)}")
            logger.info(f"    Missing: {sorted(missing)}")
        
        return (
            min(100, max(0, score)),
            sorted(list(exact_matches)),
            sorted(list(missing)),
            details
        )
    
    def _compute_experience_match(
        self,
        profile_chunks: ProfileChunks,
        job_chunks: JobChunks,
        profile_exp_emb: Optional[np.ndarray],
        job_resp_emb: Optional[np.ndarray],
        skill_score: float = 50.0
    ) -> Tuple[float, Dict[str, str]]:
        """
        Compute experience match score.
        
        Strategy:
        - Semantic similarity of experience text
        - Years comparison bonus/penalty
        - PENALTY: If skill match is low (<30%), experience is discounted
        """
        analysis = {}
        score = 50.0  # Default neutral
        
        # 1. Semantic similarity of experience
        semantic_sim = 0.5
        if profile_exp_emb is not None and job_resp_emb is not None:
            if np.any(profile_exp_emb) and np.any(job_resp_emb):
                sim = self.embedder.similarity(profile_exp_emb, job_resp_emb)
                semantic_sim = (sim + 1) / 2  # Convert to 0-1
                score = semantic_sim * 100
                analysis["semantic_sim"] = f"{sim:.3f}"
        
        logger.debug(f"Experience semantic similarity: {semantic_sim:.3f} -> base score: {score:.1f}")
        
        # 2. Years of experience comparison
        profile_years = profile_chunks.years_experience
        job_years = job_chunks.years_required
        
        if profile_years and job_years:
            if profile_years >= job_years:
                # Meet or exceed requirement - bonus
                years_boost = min(20, (profile_years - job_years) * 2)
                score = min(100, score + years_boost)
                analysis["years_status"] = f"Have {profile_years}y, need {job_years}y (+{years_boost} bonus)"
                logger.debug(f"Years OK: {profile_years} >= {job_years} -> +{years_boost} bonus")
            else:
                # Under-qualified - penalty
                years_penalty = min(30, (job_years - profile_years) * 5)
                score = max(0, score - years_penalty)
                analysis["years_status"] = f"Have {profile_years}y, need {job_years}y (-{years_penalty} penalty)"
                logger.debug(f"Years SHORT: {profile_years} < {job_years} -> -{years_penalty} penalty")
        
        # 3. PENALTY: If skills don't match, experience can't save you
        low_skill_threshold = settings.low_skill_threshold * 100  # Convert to 0-100 scale
        if skill_score < low_skill_threshold:
            original_score = score
            score = score * settings.low_skill_experience_penalty
            analysis["skill_penalty"] = f"Low skill ({skill_score:.0f}<{low_skill_threshold:.0f}) -> exp reduced {original_score:.0f} to {score:.0f}"
            logger.debug(f"Low skill penalty: {original_score:.1f} x {settings.low_skill_experience_penalty} = {score:.1f}")
        
        if settings.log_scoring_details:
            logger.info(f"  EXPERIENCE: semantic={semantic_sim:.2f}, score={score:.1f}")
            for k, v in analysis.items():
                logger.info(f"    {k}: {v}")
        
        return score, analysis
    
    def _compute_semantic_match(
        self,
        profile_full_emb: Optional[np.ndarray],
        job_full_emb: Optional[np.ndarray]
    ) -> float:
        """
        Compute overall semantic similarity between full profile and job text.
        """
        if profile_full_emb is None or job_full_emb is None:
            return 50.0
            
        if not np.any(profile_full_emb) or not np.any(job_full_emb):
            return 50.0
            
        sim = self.embedder.similarity(profile_full_emb, job_full_emb)
        
        # Convert from [-1, 1] to [0, 100]
        score = ((sim + 1) / 2) * 100
        
        if settings.log_scoring_details:
            logger.info(f"  SEMANTIC: cosine={sim:.3f} -> score={score:.1f}")
        
        return score
    
    def _compute_requirements_fit(
        self,
        profile_chunks: ProfileChunks,
        job_chunks: JobChunks,
        profile_edu_emb: Optional[np.ndarray],
        job_qual_emb: Optional[np.ndarray]
    ) -> Tuple[float, Dict[str, str]]:
        """
        Compute requirements fit (education, certifications, etc.).
        """
        analysis = {}
        score = 60.0  # Default neutral
        
        # Semantic similarity of education to qualifications
        if profile_edu_emb is not None and job_qual_emb is not None:
            if np.any(profile_edu_emb) and np.any(job_qual_emb):
                sim = self.embedder.similarity(profile_edu_emb, job_qual_emb)
                score = ((sim + 1) / 2) * 100
                analysis["education_sim"] = f"{sim:.3f}"
        
        if settings.log_scoring_details:
            logger.info(f"  REQUIREMENTS: score={score:.1f}")
                
        return score, analysis
    
    def _get_recommendation(self, score: int) -> Recommendation:
        """Determine recommendation based on score"""
        if score >= settings.threshold_strong_apply:
            return Recommendation.STRONG_APPLY
        elif score >= settings.threshold_good_fit:
            return Recommendation.GOOD_FIT
        elif score >= settings.threshold_maybe:
            return Recommendation.MAYBE
        else:
            return Recommendation.WEAK_MATCH
    
    def _generate_insights(
        self,
        skill_score: float,
        experience_score: float,
        semantic_score: float,
        requirements_score: float,
        matched_skills: List[str],
        missing_skills: List[str],
        exp_analysis: Dict[str, str],
        req_analysis: Dict[str, str],
        profile_chunks: ProfileChunks,
        job_chunks: JobChunks
    ) -> Tuple[List[str], List[str]]:
        """
        Generate human-readable strengths and gaps.
        """
        strengths = []
        gaps = []
        
        # Skill-based insights
        if matched_skills:
            top_skills = matched_skills[:5]
            strengths.append(f"Strong skill match: {', '.join(top_skills)}")
            
        if len(matched_skills) >= len(job_chunks.required_skills_list) * 0.7:
            strengths.append(f"Matches {len(matched_skills)} of {len(job_chunks.required_skills_list)} required skills")
        elif len(matched_skills) == 0 and len(job_chunks.required_skills_list) > 0:
            gaps.append(f"No matching skills from {len(job_chunks.required_skills_list)} required")
            
        if missing_skills:
            top_missing = missing_skills[:3]
            gaps.append(f"Missing skills: {', '.join(top_missing)}")
            
        # Experience-based insights
        if experience_score >= 70:
            strengths.append("Relevant experience background")
        elif experience_score < 40:
            gaps.append("Experience may not be directly relevant")
            
        if profile_chunks.years_experience:
            if profile_chunks.years_experience >= 10:
                strengths.append(f"{profile_chunks.years_experience}+ years of experience")
            elif profile_chunks.years_experience >= 5:
                strengths.append(f"{profile_chunks.years_experience} years of experience")
                
        if "years_status" in exp_analysis and "need" in exp_analysis["years_status"] and "-" in exp_analysis["years_status"]:
            gaps.append(exp_analysis["years_status"])
            
        # Semantic insights
        if semantic_score >= 75:
            strengths.append("Strong overall profile-job alignment")
        elif semantic_score < 45:
            gaps.append("Profile focus differs from job requirements")
            
        # Limit to top insights
        return strengths[:5], gaps[:4]


# Global engine instance
_engine: Optional[MatchingEngine] = None


def get_matching_engine() -> MatchingEngine:
    """Get or create the global matching engine instance"""
    global _engine
    
    if _engine is None:
        _engine = MatchingEngine()
        
    return _engine
