"""
Document chunker for extracting semantic sections from profiles and job postings.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import re
import logging

from app.processors.tokenizer import canonicalize_skill, get_text_processor
from app.processors.skill_groups import (
    flatten_groups,
    normalize_jd_text,
    parse_skill_groups,
    split_required_preferred,
)

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
    qualification_evidence_text: str = ""
    
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
            "full": self.full_text,
            "qualification_evidence": self.qualification_evidence_text,
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
    required_skill_groups: List[List[str]] = field(default_factory=list)
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
                    skills_list.append(canonicalize_skill(name))
                    skill_str = name
                    if years:
                        skill_str += f" ({years} years)"
                    if proficiency:
                        skill_str += f" [{proficiency}]"
                    skills_parts.append(skill_str)
            elif isinstance(skill, str):
                skills_list.append(canonicalize_skill(skill))
                skills_parts.append(skill)
                
        chunks.skills_text = "Skills: " + ", ".join(skills_parts) if skills_parts else ""
        chunks.skills_list = _dedupe_keep_order(skills_list)
        
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
                canon = canonicalize_skill(skill)
                if canon and canon not in chunks.skills_list:
                    chunks.skills_list.append(canon)
            
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

        # Qualification evidence: structured profile signal vs job required quals.
        # Reuses the existing compact experience_text (title/company + truncated
        # cleaned bullets) so demonstrated responsibilities are present without
        # concatenating resume_text / full_text (those stay on semantic_match).
        evidence_parts = []
        if chunks.years_experience:
            evidence_parts.append(f"{chunks.years_experience} years of experience")
        if chunks.summary_text:
            evidence_parts.append(chunks.summary_text)
        if chunks.skills_text:
            evidence_parts.append(chunks.skills_text)
        if chunks.experience_text:
            evidence_parts.append(chunks.experience_text)
        if chunks.education_text:
            evidence_parts.append(chunks.education_text)
        chunks.qualification_evidence_text = " ".join(evidence_parts)
            
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
        
        combined_prose = "\n".join(p for p in (description or "", requirements or "") if p)
        normalized = normalize_jd_text(combined_prose)
        sections = split_required_preferred(normalized)

        groups = parse_skill_groups(sections.required_text) if sections.required_text else []
        grouped_skills = {skill for group in groups for skill in group}

        preferred_from_text = set()
        if sections.preferred_text:
            preferred_from_text.update(self.processor.extract_skills(sections.preferred_text))

        # skills_required is a lossy flatten of the whole JD (often including
        # OR alternatives and preferred-only skills). Absorb; do not AND-promote.
        if skills_required:
            for raw in skills_required.split(","):
                token = canonicalize_skill(raw)
                if not token:
                    continue
                if token in grouped_skills:
                    continue
                if token in preferred_from_text or (
                    sections.preferred_text and token in sections.preferred_text.lower()
                ):
                    preferred_from_text.add(token)
                    continue
                if sections.found_required_header and token not in sections.required_text.lower():
                    continue
                groups.append([token])
                grouped_skills.add(token)

        chunks.required_skill_groups = groups
        chunks.required_skills_list = flatten_groups(groups)
        chunks.preferred_skills_list = sorted(preferred_from_text)
        chunks.required_skills_text = (
            f"Required Skills: {', '.join(chunks.required_skills_list)}"
            if chunks.required_skills_list else ""
        )
        
        # 2. Responsibilities chunk (from description)
        if description:
            # Try to extract responsibilities section
            resp_text = self._extract_section(description, 
                ["responsibilities", "duties", "what you'll do", "you will"])
            if not resp_text:
                resp_text = description[:500]  # Use first part of description
            chunks.responsibilities_text = f"Responsibilities: {self.processor.clean_text(resp_text)}"
            
        # 3. Qualifications chunk: required-quals text only. No title/company
        # metadata (those are not qualifications) and no preferred section.
        qual_source = sections.required_text
        if not qual_source and requirements:
            qual_source = normalize_jd_text(requirements)
        if not qual_source and normalized:
            resp_slice = self._extract_section(normalized,
                ["responsibilities", "duties", "what you'll do", "you will", "key responsibilities"])
            qual_source = normalized
            if resp_slice:
                qual_source = normalized.replace(resp_slice, " ", 1)
        chunks.qualifications_text = (
            f"Qualifications: {self.processor.clean_text(qual_source)}"
            if qual_source else ""
        )
        
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


def _dedupe_keep_order(values: List[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


# Global chunker instance
_chunker: Optional[DocumentChunker] = None


def get_chunker() -> DocumentChunker:
    """Get or create the global chunker instance"""
    global _chunker
    
    if _chunker is None:
        _chunker = DocumentChunker()
        
    return _chunker
