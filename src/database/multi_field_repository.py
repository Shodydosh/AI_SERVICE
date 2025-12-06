"""Repository for Multi-Field Embeddings."""
from typing import List, Optional
from sqlalchemy.orm import Session
from src.database.models import (
    JobDescriptionMultiEmbedding,
    CandidateMultiEmbedding
)
import logging

logger = logging.getLogger(__name__)


class MultiFieldEmbeddingRepository:
    """Repository for Multi-Field Embedding operations."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_candidate_multi_embedding(self, candidate_id: str) -> Optional[CandidateMultiEmbedding]:
        """Get candidate by candidate_id."""
        return self.db.query(CandidateMultiEmbedding).filter(
            CandidateMultiEmbedding.candidate_id == candidate_id
        ).first()
    
    def get_job_multi_embedding(self, job_id: str) -> Optional[JobDescriptionMultiEmbedding]:
        """Get job by job_id."""
        return self.db.query(JobDescriptionMultiEmbedding).filter(
            JobDescriptionMultiEmbedding.job_id == job_id
        ).first()
    
    def get_all_candidate_multi_embeddings(self) -> List[CandidateMultiEmbedding]:
        """Get all candidates."""
        return self.db.query(CandidateMultiEmbedding).all()
    
    def get_all_job_multi_embeddings(self) -> List[JobDescriptionMultiEmbedding]:
        """Get all jobs."""
        return self.db.query(JobDescriptionMultiEmbedding).all()
    
    def count_candidate_multi_embeddings(self) -> int:
        """Count total candidates."""
        return self.db.query(CandidateMultiEmbedding).count()
    
    def count_job_multi_embeddings(self) -> int:
        """Count total jobs."""
        return self.db.query(JobDescriptionMultiEmbedding).count()
    
    def upsert_candidate_multi_embedding(
        self,
        candidate_id: str,
        title: str,
        skills: Optional[str],
        experience: Optional[str],
        title_embedding: List[float],
        skills_embedding: List[float],
        experience_embedding: List[float]
    ) -> CandidateMultiEmbedding:
        """Create or update candidate."""
        existing = self.db.query(CandidateMultiEmbedding).filter(
            CandidateMultiEmbedding.candidate_id == candidate_id
        ).first()
        
        if existing:
            existing.title = title
            existing.skills = skills
            existing.experience = experience
            existing.title_embedding = title_embedding
            existing.skills_embedding = skills_embedding
            existing.experience_embedding = experience_embedding
            self.db.commit()
            self.db.refresh(existing)
            return existing
        else:
            candidate = CandidateMultiEmbedding(
                candidate_id=candidate_id,
                title=title,
                skills=skills,
                experience=experience,
                title_embedding=title_embedding,
                skills_embedding=skills_embedding,
                experience_embedding=experience_embedding
            )
            self.db.add(candidate)
            self.db.commit()
            self.db.refresh(candidate)
            return candidate
    
    def upsert_job_multi_embedding(
        self,
        job_id: str,
        title: str,
        skills: Optional[str],
        requirement: Optional[str],
        title_embedding: List[float],
        skills_embedding: List[float],
        requirement_embedding: List[float]
    ) -> JobDescriptionMultiEmbedding:
        """Create or update job."""
        existing = self.db.query(JobDescriptionMultiEmbedding).filter(
            JobDescriptionMultiEmbedding.job_id == job_id
        ).first()
        
        if existing:
            existing.title = title
            existing.skills = skills
            existing.requirement = requirement
            existing.title_embedding = title_embedding
            existing.skills_embedding = skills_embedding
            existing.requirement_embedding = requirement_embedding
            self.db.commit()
            self.db.refresh(existing)
            return existing
        else:
            job = JobDescriptionMultiEmbedding(
                job_id=job_id,
                title=title,
                skills=skills,
                requirement=requirement,
                title_embedding=title_embedding,
                skills_embedding=skills_embedding,
                requirement_embedding=requirement_embedding
            )
            self.db.add(job)
            self.db.commit()
            self.db.refresh(job)
            return job


