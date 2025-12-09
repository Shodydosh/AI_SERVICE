"""Repository for Two-Tower database operations."""
from typing import List, Optional, Dict
from sqlalchemy.orm import Session
from datetime import datetime
from src.database.models import (
    JobDescriptionTwoTower,
    CandidateTwoTower,
    ReindexTracking
)
import logging

logger = logging.getLogger(__name__)


class TwoTowerRepository:
    """Repository for Two-Tower operations."""
    
    def __init__(self, db: Session):
        self.db = db
    
    # ========================================================================
    # Job Operations
    # ========================================================================
    
    def create_job(
        self,
        job_id: str,
        title: str,
        title_embedding: List[float],
        skills_embedding: List[float],
        requirement_embedding: List[float],
        skills: Optional[str] = None,
        requirement: Optional[str] = None,
        company: Optional[str] = None,
        location: Optional[str] = None,
        replace_existing: bool = True
    ) -> JobDescriptionTwoTower:
        """Create or update a job."""
        if replace_existing:
            existing = self.db.query(JobDescriptionTwoTower).filter(
                JobDescriptionTwoTower.job_id == job_id
            ).first()
            if existing:
                self.db.delete(existing)
                self.db.flush()
        
        job = JobDescriptionTwoTower(
            job_id=job_id,
            title=title,
            skills=skills,
            requirement=requirement,
            company=company,
            location=location,
            title_embedding=title_embedding,
            skills_embedding=skills_embedding,
            requirement_embedding=requirement_embedding
        )
        
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job
    
    def get_job(self, job_id: str) -> Optional[JobDescriptionTwoTower]:
        """Get job by job_id."""
        return self.db.query(JobDescriptionTwoTower).filter(
            JobDescriptionTwoTower.job_id == job_id
        ).first()
    
    def get_all_jobs(self) -> List[JobDescriptionTwoTower]:
        """Get all jobs."""
        return self.db.query(JobDescriptionTwoTower).all()
    
    def get_jobs_updated_since(self, since_time: datetime) -> List[JobDescriptionTwoTower]:
        """Get jobs updated since given time."""
        return self.db.query(JobDescriptionTwoTower).filter(
            JobDescriptionTwoTower.updated_at >= since_time
        ).all()
    
    # ========================================================================
    # Candidate Operations
    # ========================================================================
    
    def create_candidate(
        self,
        candidate_id: str,
        title_embedding: List[float],
        skills_embedding: List[float],
        experience_embedding: List[float],
        title: Optional[str] = None,
        skills: Optional[str] = None,
        experience: Optional[str] = None,
        name: Optional[str] = None,
        email: Optional[str] = None,
        replace_existing: bool = True
    ) -> CandidateTwoTower:
        """Create or update a candidate."""
        if replace_existing:
            existing = self.db.query(CandidateTwoTower).filter(
                CandidateTwoTower.candidate_id == candidate_id
            ).first()
            if existing:
                self.db.delete(existing)
                self.db.flush()
        
        candidate = CandidateTwoTower(
            candidate_id=candidate_id,
            title=title,
            skills=skills,
            experience=experience,
            name=name,
            email=email,
            title_embedding=title_embedding,
            skills_embedding=skills_embedding,
            experience_embedding=experience_embedding
        )
        
        self.db.add(candidate)
        self.db.commit()
        self.db.refresh(candidate)
        return candidate
    
    def get_candidate(self, candidate_id: str) -> Optional[CandidateTwoTower]:
        """Get candidate by candidate_id."""
        return self.db.query(CandidateTwoTower).filter(
            CandidateTwoTower.candidate_id == candidate_id
        ).first()
    
    def get_all_candidates(self) -> List[CandidateTwoTower]:
        """Get all candidates."""
        return self.db.query(CandidateTwoTower).all()
    
    def get_candidates_updated_since(self, since_time: datetime) -> List[CandidateTwoTower]:
        """Get candidates updated since given time."""
        return self.db.query(CandidateTwoTower).filter(
            CandidateTwoTower.updated_at >= since_time
        ).all()
    
    # ========================================================================
    # Reindex Tracking Operations
    # ========================================================================
    
    def create_reindex_tracking(
        self,
        reindex_type: str,
        status: str = "pending",
        total_records: Optional[int] = None
    ) -> ReindexTracking:
        """Create a reindex tracking record."""
        tracking = ReindexTracking(
            reindex_type=reindex_type,
            status=status,
            total_records=total_records,
            started_at=datetime.utcnow() if status == "running" else None
        )
        self.db.add(tracking)
        self.db.commit()
        self.db.refresh(tracking)
        return tracking
    
    def update_reindex_tracking(
        self,
        tracking_id: int,
        status: Optional[str] = None,
        processed_records: Optional[int] = None,
        error_message: Optional[str] = None
    ):
        """Update reindex tracking record."""
        tracking = self.db.query(ReindexTracking).filter(
            ReindexTracking.id == tracking_id
        ).first()
        
        if not tracking:
            return
        
        if status:
            tracking.status = status
            if status == "running" and not tracking.started_at:
                tracking.started_at = datetime.utcnow()
            elif status in ["completed", "failed"]:
                tracking.completed_at = datetime.utcnow()
        
        if processed_records is not None:
            tracking.processed_records = processed_records
        
        if error_message:
            tracking.error_message = error_message
        
        self.db.commit()


