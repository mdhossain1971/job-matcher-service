"""
Configuration settings for Job Matcher Service
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os
import logging


class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    app_name: str = "Job Matcher ML Service"
    app_version: str = "1.1.0"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 5000
    
    # Logging Settings
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_scoring_details: bool = True  # Log detailed scoring breakdown
    log_embeddings: bool = False  # Log embedding vectors (verbose!)
    log_format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    
    # Model Settings
    embedding_model: str = "all-MiniLM-L6-v2"  # Fast and good quality
    spacy_model: str = "en_core_web_sm"  # Use 'en_core_web_lg' for better accuracy
    
    # Scoring Weights (must sum to 1.0)
    weight_skill_match: float = 0.50      # Increased from 0.40
    weight_experience_match: float = 0.25  # Decreased from 0.30
    weight_semantic_match: float = 0.15    # Decreased from 0.20
    weight_requirements_fit: float = 0.10  # Same
    
    # Scoring Penalties & Caps
    zero_skill_match_cap: int = 45         # Max score if NO skills match
    low_skill_threshold: float = 0.30      # Below this, penalize experience
    low_skill_experience_penalty: float = 0.5  # Multiply experience by this if skills low
    
    # Thresholds for recommendations
    threshold_strong_apply: int = 80       # Lowered from 85
    threshold_good_fit: int = 65           # Lowered from 70
    threshold_maybe: int = 50              # Lowered from 55
    
    # Storage Settings
    storage_backend: str = "memory"  # "memory" or "opensearch"
    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def setup_logging():
    """Configure logging based on settings"""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format=settings.log_format
    )
    
    # Create logger for scoring module
    scoring_logger = logging.getLogger("scoring")
    scoring_logger.setLevel(log_level)
    
    return scoring_logger


# Initialize logging
scoring_logger = setup_logging()
