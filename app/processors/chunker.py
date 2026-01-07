"""
Document chunker for extracting semantic sections from profiles and job postings.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import re
import logging

from app.processors.tokenizer import get_text_processor

logger = logging.getLogger(__name__)


@dataclass
class ProfileChunks:
    """Chunked representation of a user profile"""
    profile_id: int
    skills_text: str = ""
    experience_text: str = ""
    education_text: str = ""
    summary_text: str = ""
    full_text: str = ""  # Combined for overall semantic matching
    
    # Extracted data
    skills_list: List[str] = field(default_factory=list)
    years_experience: Optional[int] = None
    
    def to_dict(self) -> Dict[str, str]:
        """Return chunks as dictionary for embedding"""
        return {
            "skills": self.skills_text,
            "experience": self.experience_text,
            "education": self.education_text,
            "summary": self.summary_text,
            "full": self.full_text
        }


@dataclass
class JobChunks:
    """Chunked representation of a job posting"""
    job_id: int
    required_skills_text: str = ""
    responsibilities_text: str = ""
    qualifications_text: str = ""
    full_text: str = ""  # Combined for overall semantic matching
    
    # Extracted data
    required_skills_list: List[str] = field(default_factory=list)
    preferred_skills_list: List[str] = field(default_factory=list)
    years_required: Optional[int] = None
    
    def to_dict(self) -> Dict[str, str]:
        """Return chunks as dictionary for embedding"""
        return {
            "required_skills": self.required_skills_text,
            "responsibilities": self.responsibilities_text,
            "qualifications": self.qualifications_text,
            "full": self.full_text
        }


class DocumentChunker:
    """
    Chunks documents (profiles and jobs) into semantic sections for embedding.
    """
    
    def __init__(self):
        self.processor = get_text_processor()
        
    def chunk_profile(self, profile_data: Dict[str, Any]) -> ProfileChunks:
        """
        Chunk a user profile into semantic sections.
        
        Args:
            profile_data: Dictionary containing profile fields:
                - profile_id: int
                - skills: List[dict] with name, years, proficiency
                - experiences: List[dict] with company_name, title, description
                - education: List[dict] with institution, degree, field_of_study
                - certifications: List[str]
                - summary: str
                - resume_text: str (optional, for additional context)
                - years_of_experience: int (optional)
                
        Returns:
            ProfileChunks object with chunked text
        """
        profile_id = profile_data.get("profile_id", 0)
        chunks = ProfileChunks(profile_id=profile_id)
        
        # 1. Skills chunk
        skills = profile_data.get("skills", [])
        skills_parts = []
        skills_list = []
        
        for skill in skills:
            if isinstance(skill, dict):
                name = skill.get("name", "")
                years = skill.get("years")
                proficiency = skill.get("proficiency", "")
                
                if name:
                    skills_list.append(name.lower())
                    skill_str = name
                    if years:
                        skill_str += f" ({years} years)"
                    if proficiency:
                        skill_str += f" [{proficiency}]"
                    skills_parts.append(skill_str)
            elif isinstance(skill, str):
                skills_list.append(skill.lower())
                skills_parts.append(skill)
                
        chunks.skills_text = "Skills: " + ", ".join(skills_parts) if skills_parts else ""
        chunks.skills_list = skills_list
        
        # 2. Experience chunk
        experiences = profile_data.get("experiences", [])
        exp_parts = []
        
        for exp in experiences:
            if isinstance(exp, dict):
                title = exp.get("title", "")
                company = exp.get("company_name", "")
                description = exp.get("description", "")
                
                exp_str = f"{title} at {company}"
                if description:
                    # Truncate long descriptions
                    desc_clean = self.processor.clean_text(description)
                    if len(desc_clean) > 300:
                        desc_clean = desc_clean[:300] + "..."
                    exp_str += f": {desc_clean}"
                exp_parts.append(exp_str)
                
        chunks.experience_text = "Experience: " + " | ".join(exp_parts) if exp_parts else ""
        
        # 3. Education chunk
        education = profile_data.get("education", [])
        certifications = profile_data.get("certifications", [])
        edu_parts = []
        
        for edu in education:
            if isinstance(edu, dict):
                degree = edu.get("degree", "")
                field = edu.get("field_of_study", "")
                institution = edu.get("institution", "")
                
                edu_str = degree
                if field:
                    edu_str += f" in {field}"
                if institution:
                    edu_str += f" from {institution}"
                edu_parts.append(edu_str)
                
        for cert in certifications:
            if isinstance(cert, str):
                edu_parts.append(f"Certified: {cert}")
            elif isinstance(cert, dict):
                edu_parts.append(f"Certified: {cert.get('name', '')}")
                
        chunks.education_text = "Education: " + " | ".join(edu_parts) if edu_parts else ""
        
        # 4. Summary chunk
        summary = profile_data.get("summary", "")
        if summary:
            chunks.summary_text = f"Professional Summary: {self.processor.clean_text(summary)}"
            
        # 5. Full text (for overall semantic matching)
        full_parts = [
            chunks.summary_text,
            chunks.skills_text,
            chunks.experience_text,
            chunks.education_text
        ]
        
        # Add resume text if available
        resume_text = profile_data.get("resume_text", "")
        if resume_text:
            # Extract additional skills from resume
            resume_skills = self.processor.extract_skills(resume_text)
            for skill in resume_skills:
                if skill.lower() not in chunks.skills_list:
                    chunks.skills_list.append(skill.lower())
            
            # Add to full text
            full_parts.append(f"Resume: {self.processor.clean_text(resume_text)[:1000]}")
            
        chunks.full_text = " ".join(p for p in full_parts if p)
        
        # Extract years of experience
        chunks.years_experience = profile_data.get("years_of_experience")
        if not chunks.years_experience:
            # Try to extract from summary or resume
            chunks.years_experience = self.processor.extract_years_experience(
                summary or resume_text or ""
            )
            
        return chunks
    
    def chunk_job(self, job_data: Dict[str, Any]) -> JobChunks:
        """
        Chunk a job posting into semantic sections.
        
        Args:
            job_data: Dictionary containing job fields:
                - job_id: int
                - title: str
                - company_name: str
                - description: str
                - requirements: str
                - skills_required: str
                - location: str
                - remote_type: str
                - experience_level: str
                
        Returns:
            JobChunks object with chunked text
        """
        job_id = job_data.get("job_id", 0)
        chunks = JobChunks(job_id=job_id)
        
        title = job_data.get("title", "")
        company = job_data.get("company_name", "")
        description = job_data.get("description", "")
        requirements = job_data.get("requirements", "")
        skills_required = job_data.get("skills_required", "")
        location = job_data.get("location", "")
        remote_type = job_data.get("remote_type", "")
        
        # 1. Required skills chunk
        # Extract from skills_required field and from description/requirements
        all_skills = set()
        
        if skills_required:
            # Parse comma-separated skills
            for skill in skills_required.split(","):
                skill = skill.strip().lower()
                if skill:
                    all_skills.add(skill)
                    
        # Extract skills from description and requirements
        for text in [description, requirements]:
            if text:
                extracted = self.processor.extract_skills(text)
                all_skills.update(extracted)
                
        chunks.required_skills_list = sorted(list(all_skills))
        chunks.required_skills_text = f"Required Skills: {', '.join(chunks.required_skills_list)}" if all_skills else ""
        
        # 2. Responsibilities chunk (from description)
        if description:
            # Try to extract responsibilities section
            resp_text = self._extract_section(description, 
                ["responsibilities", "duties", "what you'll do", "you will"])
            if not resp_text:
                resp_text = description[:500]  # Use first part of description
            chunks.responsibilities_text = f"Responsibilities: {self.processor.clean_text(resp_text)}"
            
        # 3. Qualifications chunk (from requirements)
        qual_parts = []
        
        if requirements:
            qual_parts.append(self.processor.clean_text(requirements))
            
        # Add job metadata
        meta_parts = []
        if title:
            meta_parts.append(f"Position: {title}")
        if company:
            meta_parts.append(f"Company: {company}")
        if location:
            meta_parts.append(f"Location: {location}")
        if remote_type:
            meta_parts.append(f"Remote: {remote_type}")
            
        if meta_parts:
            qual_parts.insert(0, " | ".join(meta_parts))
            
        chunks.qualifications_text = "Qualifications: " + " ".join(qual_parts) if qual_parts else ""
        
        # 4. Full text (for overall semantic matching)
        full_parts = [
            f"Job Title: {title}",
            f"Company: {company}",
            chunks.required_skills_text,
            chunks.responsibilities_text,
            chunks.qualifications_text
        ]
        chunks.full_text = " ".join(p for p in full_parts if p)
        
        # Extract years required
        combined_text = f"{requirements} {description}"
        chunks.years_required = self.processor.extract_years_experience(combined_text)
        
        return chunks
    
    def _extract_section(self, text: str, keywords: List[str]) -> str:
        """
        Extract a section from text based on keywords.
        
        Args:
            text: Full text
            keywords: List of section header keywords
            
        Returns:
            Extracted section text or empty string
        """
        text_lower = text.lower()
        
        for keyword in keywords:
            # Look for section headers
            patterns = [
                rf'{keyword}[:\s]+(.{{50,500}})',  # Keyword: content
                rf'\n{keyword}\n(.{{50,500}})',    # Keyword on own line
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL)
                if match:
                    return match.group(1).strip()
                    
        return ""


# Global chunker instance
_chunker: Optional[DocumentChunker] = None


def get_chunker() -> DocumentChunker:
    """Get or create the global chunker instance"""
    global _chunker
    
    if _chunker is None:
        _chunker = DocumentChunker()
        
    return _chunker
