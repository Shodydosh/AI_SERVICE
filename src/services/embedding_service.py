"""Optimized embedding service with caching and batch processing."""
import hashlib
import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_
import torch
import numpy as np

from src.services.embedding_cache_manager import get_cache_manager
from src.database.multi_field_repository import MultiFieldEmbeddingRepository
from src.database.models import (
    CandidateMultiEmbedding,
    JobDescriptionMultiEmbedding
)
from src.embeddings.candidate_tower_encoder import CandidateTowerEncoder
from src.embeddings.job_tower_encoder import JobTowerEncoder

logger = logging.getLogger(__name__)


class OptimizedEmbeddingService:
    """
    Optimized embedding service with:
    - Smart caching (12-hour cycle)
    - Batch processing
    - Efficient PostgreSQL storage
    - Non-blocking realtime queries
    """
    
    def __init__(
        self,
        db: Session,
        cache_ttl_hours: float = 12.0,
        batch_size: int = 100
    ):
        """
        Initialize embedding service.
        
        Args:
            db: Database session
            cache_ttl_hours: Cache TTL in hours (default: 12)
            batch_size: Batch size for processing (default: 100)
        """
        self.db = db
        self.repository = MultiFieldEmbeddingRepository(db)
        self.cache = get_cache_manager(ttl_hours=cache_ttl_hours)
        self.batch_size = batch_size
        
        # Lazy load encoders (only when needed)
        self._candidate_encoder: Optional[CandidateTowerEncoder] = None
        self._job_encoder: Optional[JobTowerEncoder] = None
    
    @property
    def candidate_encoder(self) -> CandidateTowerEncoder:
        """Lazy load candidate encoder."""
        if self._candidate_encoder is None:
            self._candidate_encoder = CandidateTowerEncoder()
        return self._candidate_encoder
    
    @property
    def job_encoder(self) -> JobTowerEncoder:
        """Lazy load job encoder."""
        if self._job_encoder is None:
            self._job_encoder = JobTowerEncoder()
        return self._job_encoder
    
    def _compute_content_hash(self, title: str, skills: str, experience: str) -> str:
        """Compute content hash for change detection."""
        content = f"{title}|{skills}|{experience}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _compute_job_content_hash(self, title: str, skills: str, requirement: str) -> str:
        """Compute content hash for job."""
        content = f"{title}|{skills}|{requirement}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get_candidate_embedding(
        self,
        candidate_id: str,
        title: str,
        skills: Optional[str],
        experience: Optional[str],
        force_refresh: bool = False
    ) -> Dict[str, List[float]]:
        """
        Get candidate embedding with caching.
        
        Args:
            candidate_id: Candidate ID
            title: Candidate title
            skills: Candidate skills
            experience: Candidate experience
            force_refresh: Force refresh even if cache is valid
            
        Returns:
            Dict with title_embedding, skills_embedding, experience_embedding
        """
        content_hash = self._compute_content_hash(title or "", skills or "", experience or "")
        
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached = self.cache.get(candidate_id, 'candidate', content_hash)
            if cached is not None:
                logger.debug(f"Cache hit for candidate {candidate_id}")
                return cached
        
        # Check database
        db_record = self.repository.get_candidate_multi_embedding(candidate_id)
        
        if db_record and not force_refresh:
            # Check if database embedding is still fresh
            db_timestamp = db_record.embedding_timestamp or db_record.created_at
            if db_timestamp:
                age = datetime.now() - db_timestamp
                if age < timedelta(hours=self.cache.cache_ttl_hours):
                    # Check content hash
                    if db_record.content_hash == content_hash:
                        # Database embedding is fresh and content unchanged
                        embeddings = {
                            'title_embedding': db_record.title_embedding,
                            'skills_embedding': db_record.skills_embedding,
                            'experience_embedding': db_record.experience_embedding
                        }
                        # Cache it
                        self.cache.set(candidate_id, 'candidate', embeddings, content_hash)
                        logger.debug(f"Using fresh database embedding for candidate {candidate_id}")
                        return embeddings
        
        # Need to compute new embedding
        logger.info(f"Computing new embedding for candidate {candidate_id}")
        embeddings = self.candidate_encoder.encode_candidate(
            title=title,
            skills=skills,
            experience=experience
        )
        
        # Save to database (async/batch in production)
        self._save_candidate_embedding(
            candidate_id=candidate_id,
            title=title,
            skills=skills,
            experience=experience,
            embeddings=embeddings,
            content_hash=content_hash
        )
        
        # Cache it
        self.cache.set(candidate_id, 'candidate', embeddings, content_hash)
        
        return embeddings
    
    def get_job_embedding(
        self,
        job_id: str,
        title: str,
        skills: Optional[str],
        requirement: Optional[str],
        force_refresh: bool = False
    ) -> Dict[str, List[float]]:
        """
        Get job embedding with caching.
        
        Args:
            job_id: Job ID
            title: Job title
            skills: Job skills
            requirement: Job requirement
            force_refresh: Force refresh even if cache is valid
            
        Returns:
            Dict with title_embedding, skills_embedding, requirement_embedding
        """
        content_hash = self._compute_job_content_hash(title or "", skills or "", requirement or "")
        
        # Check cache first
        if not force_refresh:
            cached = self.cache.get(job_id, 'job', content_hash)
            if cached is not None:
                logger.debug(f"Cache hit for job {job_id}")
                return cached
        
        # Check database
        db_record = self.repository.get_job_multi_embedding(job_id)
        
        if db_record and not force_refresh:
            # Check if database embedding is still fresh
            db_timestamp = db_record.embedding_timestamp or db_record.created_at
            if db_timestamp:
                age = datetime.now() - db_timestamp
                if age < timedelta(hours=self.cache.cache_ttl_hours):
                    # Check content hash
                    if db_record.content_hash == content_hash:
                        # Database embedding is fresh and content unchanged
                        embeddings = {
                            'title_embedding': db_record.title_embedding,
                            'skills_embedding': db_record.skills_embedding,
                            'requirement_embedding': db_record.requirement_embedding
                        }
                        # Cache it
                        self.cache.set(job_id, 'job', embeddings, content_hash)
                        logger.debug(f"Using fresh database embedding for job {job_id}")
                        return embeddings
        
        # Need to compute new embedding
        logger.info(f"Computing new embedding for job {job_id}")
        embeddings = self.job_encoder.encode_job(
            title=title,
            skills=skills,
            requirements=requirement
        )
        
        # Save to database
        self._save_job_embedding(
            job_id=job_id,
            title=title,
            skills=skills,
            requirement=requirement,
            embeddings=embeddings,
            content_hash=content_hash
        )
        
        # Cache it
        self.cache.set(job_id, 'job', embeddings, content_hash)
        
        return embeddings
    
    def _save_candidate_embedding(
        self,
        candidate_id: str,
        title: str,
        skills: Optional[str],
        experience: Optional[str],
        embeddings: Dict[str, List[float]],
        content_hash: str
    ):
        """Save candidate embedding to database efficiently."""
        try:
            existing = self.repository.get_candidate_multi_embedding(candidate_id)
            
            if existing:
                # Update existing
                existing.title = title
                existing.skills = skills
                existing.experience = experience
                existing.title_embedding = embeddings['title_embedding']
                existing.skills_embedding = embeddings['skills_embedding']
                existing.experience_embedding = embeddings['experience_embedding']
                existing.embedding_timestamp = datetime.now()
                existing.content_hash = content_hash
                existing.updated_at = datetime.now()
            else:
                # Create new
                from src.database.models import CandidateMultiEmbedding
                candidate = CandidateMultiEmbedding(
                    candidate_id=candidate_id,
                    title=title,
                    skills=skills,
                    experience=experience,
                    title_embedding=embeddings['title_embedding'],
                    skills_embedding=embeddings['skills_embedding'],
                    experience_embedding=embeddings['experience_embedding'],
                    embedding_timestamp=datetime.now(),
                    content_hash=content_hash
                )
                self.db.add(candidate)
            
            self.db.commit()
            logger.debug(f"Saved embedding for candidate {candidate_id}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error saving candidate embedding {candidate_id}: {e}")
            raise
    
    def _save_job_embedding(
        self,
        job_id: str,
        title: str,
        skills: Optional[str],
        requirement: Optional[str],
        embeddings: Dict[str, List[float]],
        content_hash: str
    ):
        """Save job embedding to database efficiently."""
        try:
            existing = self.repository.get_job_multi_embedding(job_id)
            
            if existing:
                # Update existing
                existing.title = title
                existing.skills = skills
                existing.requirement = requirement
                existing.title_embedding = embeddings['title_embedding']
                existing.skills_embedding = embeddings['skills_embedding']
                existing.requirement_embedding = embeddings['requirement_embedding']
                existing.embedding_timestamp = datetime.now()
                existing.content_hash = content_hash
                existing.updated_at = datetime.now()
            else:
                # Create new
                from src.database.models import JobDescriptionMultiEmbedding
                job = JobDescriptionMultiEmbedding(
                    job_id=job_id,
                    title=title,
                    skills=skills,
                    requirement=requirement,
                    title_embedding=embeddings['title_embedding'],
                    skills_embedding=embeddings['skills_embedding'],
                    requirement_embedding=embeddings['requirement_embedding'],
                    embedding_timestamp=datetime.now(),
                    content_hash=content_hash
                )
                self.db.add(job)
            
            self.db.commit()
            logger.debug(f"Saved embedding for job {job_id}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error saving job embedding {job_id}: {e}")
            raise
    
    def batch_get_candidates_needing_refresh(
        self,
        limit: int = 1000
    ) -> List[CandidateMultiEmbedding]:
        """
        Get candidates that need embedding refresh (older than 12 hours).
        
        Args:
            limit: Maximum number of candidates to return
            
        Returns:
            List of candidates needing refresh
        """
        cutoff_time = datetime.now() - timedelta(hours=self.cache.cache_ttl_hours)
        
        return self.db.query(CandidateMultiEmbedding).filter(
            (CandidateMultiEmbedding.embedding_timestamp < cutoff_time) |
            (CandidateMultiEmbedding.embedding_timestamp.is_(None))
        ).limit(limit).all()
    
    def batch_get_jobs_needing_refresh(
        self,
        limit: int = 1000
    ) -> List[JobDescriptionMultiEmbedding]:
        """
        Get jobs that need embedding refresh (older than 12 hours).
        
        Args:
            limit: Maximum number of jobs to return
            
        Returns:
            List of jobs needing refresh
        """
        cutoff_time = datetime.now() - timedelta(hours=self.cache.cache_ttl_hours)
        
        return self.db.query(JobDescriptionMultiEmbedding).filter(
            (JobDescriptionMultiEmbedding.embedding_timestamp < cutoff_time) |
            (JobDescriptionMultiEmbedding.embedding_timestamp.is_(None))
        ).limit(limit).all()
    
    def batch_process_candidates(
        self,
        candidates: List[CandidateMultiEmbedding],
        batch_size: Optional[int] = None
    ) -> int:
        """
        Batch process candidates to refresh embeddings.
        
        Args:
            candidates: List of candidates to process
            batch_size: Batch size (default: self.batch_size)
            
        Returns:
            Number of candidates processed
        """
        batch_size = batch_size or self.batch_size
        processed = 0
        
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            logger.info(f"Processing candidate batch {i//batch_size + 1} ({len(batch)} candidates)")
            
            for candidate in batch:
                try:
                    # Compute new embedding
                    embeddings = self.candidate_encoder.encode_candidate(
                        title=candidate.title or "",
                        skills=candidate.skills,
                        experience=candidate.experience
                    )
                    
                    content_hash = self._compute_content_hash(
                        candidate.title or "",
                        candidate.skills or "",
                        candidate.experience or ""
                    )
                    
                    # Update database
                    candidate.title_embedding = embeddings['title_embedding']
                    candidate.skills_embedding = embeddings['skills_embedding']
                    candidate.experience_embedding = embeddings['experience_embedding']
                    candidate.embedding_timestamp = datetime.now()
                    candidate.content_hash = content_hash
                    candidate.updated_at = datetime.now()
                    
                    # Update cache
                    self.cache.set(
                        candidate.candidate_id,
                        'candidate',
                        embeddings,
                        content_hash
                    )
                    
                    processed += 1
                except Exception as e:
                    logger.error(f"Error processing candidate {candidate.candidate_id}: {e}")
                    continue
            
            # Commit batch
            try:
                self.db.commit()
                logger.info(f"Committed batch of {len(batch)} candidates")
            except Exception as e:
                self.db.rollback()
                logger.error(f"Error committing batch: {e}")
        
        return processed
    
    def batch_process_jobs(
        self,
        jobs: List[JobDescriptionMultiEmbedding],
        batch_size: Optional[int] = None
    ) -> int:
        """
        Batch process jobs to refresh embeddings.
        
        Args:
            jobs: List of jobs to process
            batch_size: Batch size (default: self.batch_size)
            
        Returns:
            Number of jobs processed
        """
        batch_size = batch_size or self.batch_size
        processed = 0
        
        for i in range(0, len(jobs), batch_size):
            batch = jobs[i:i + batch_size]
            logger.info(f"Processing job batch {i//batch_size + 1} ({len(batch)} jobs)")
            
            for job in batch:
                try:
                    # Compute new embedding
                    embeddings = self.job_encoder.encode_job(
                        title=job.title or "",
                        skills=job.skills,
                        requirements=job.requirement
                    )
                    
                    content_hash = self._compute_job_content_hash(
                        job.title or "",
                        job.skills or "",
                        job.requirement or ""
                    )
                    
                    # Update database
                    job.title_embedding = embeddings['title_embedding']
                    job.skills_embedding = embeddings['skills_embedding']
                    job.requirement_embedding = embeddings['requirement_embedding']
                    job.embedding_timestamp = datetime.now()
                    job.content_hash = content_hash
                    job.updated_at = datetime.now()
                    
                    # Update cache
                    self.cache.set(
                        job.job_id,
                        'job',
                        embeddings,
                        content_hash
                    )
                    
                    processed += 1
                except Exception as e:
                    logger.error(f"Error processing job {job.job_id}: {e}")
                    continue
            
            # Commit batch
            try:
                self.db.commit()
                logger.info(f"Committed batch of {len(batch)} jobs")
            except Exception as e:
                self.db.rollback()
                logger.error(f"Error committing batch: {e}")
        
        return processed

