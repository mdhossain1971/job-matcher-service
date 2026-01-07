"""
In-memory storage for embeddings (Phase 1).
Can be replaced with OpenSearch for production.
"""
from typing import Dict, Optional, List
import numpy as np
from dataclasses import dataclass
import logging
import threading

from app.processors.matcher import ProfileEmbeddings, JobEmbeddings

logger = logging.getLogger(__name__)


class InMemoryStorage:
    """
    Simple in-memory storage for profile and job embeddings.
    Thread-safe using locks.
    """
    
    def __init__(self):
        self._profiles: Dict[int, ProfileEmbeddings] = {}
        self._jobs: Dict[int, JobEmbeddings] = {}
        self._lock = threading.RLock()
        
    def store_profile(self, embeddings: ProfileEmbeddings) -> bool:
        """
        Store profile embeddings.
        
        Args:
            embeddings: ProfileEmbeddings to store
            
        Returns:
            True if successful
        """
        with self._lock:
            self._profiles[embeddings.profile_id] = embeddings
            logger.debug(f"Stored profile {embeddings.profile_id}")
            return True
            
    def get_profile(self, profile_id: int) -> Optional[ProfileEmbeddings]:
        """
        Retrieve profile embeddings.
        
        Args:
            profile_id: Profile ID
            
        Returns:
            ProfileEmbeddings or None
        """
        with self._lock:
            return self._profiles.get(profile_id)
            
    def delete_profile(self, profile_id: int) -> bool:
        """Delete profile embeddings"""
        with self._lock:
            if profile_id in self._profiles:
                del self._profiles[profile_id]
                return True
            return False
            
    def list_profiles(self) -> List[int]:
        """List all stored profile IDs"""
        with self._lock:
            return list(self._profiles.keys())
            
    def store_job(self, embeddings: JobEmbeddings) -> bool:
        """
        Store job embeddings.
        
        Args:
            embeddings: JobEmbeddings to store
            
        Returns:
            True if successful
        """
        with self._lock:
            self._jobs[embeddings.job_id] = embeddings
            logger.debug(f"Stored job {embeddings.job_id}")
            return True
            
    def get_job(self, job_id: int) -> Optional[JobEmbeddings]:
        """
        Retrieve job embeddings.
        
        Args:
            job_id: Job ID
            
        Returns:
            JobEmbeddings or None
        """
        with self._lock:
            return self._jobs.get(job_id)
            
    def delete_job(self, job_id: int) -> bool:
        """Delete job embeddings"""
        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                return True
            return False
            
    def list_jobs(self) -> List[int]:
        """List all stored job IDs"""
        with self._lock:
            return list(self._jobs.keys())
            
    def get_all_jobs(self) -> List[JobEmbeddings]:
        """Get all stored job embeddings"""
        with self._lock:
            return list(self._jobs.values())
            
    def clear(self) -> None:
        """Clear all stored data"""
        with self._lock:
            self._profiles.clear()
            self._jobs.clear()
            logger.info("Cleared all stored embeddings")
            
    def stats(self) -> Dict[str, int]:
        """Get storage statistics"""
        with self._lock:
            return {
                "profiles_count": len(self._profiles),
                "jobs_count": len(self._jobs)
            }


# Global storage instance
_storage: Optional[InMemoryStorage] = None


def get_storage() -> InMemoryStorage:
    """Get or create the global storage instance"""
    global _storage
    
    if _storage is None:
        _storage = InMemoryStorage()
        
    return _storage
