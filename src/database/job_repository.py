"""Job Repository for database operations."""
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import update
import logging

logger = logging.getLogger(__name__)


class JobRepository:
    """Repository for Job database operations."""
    
    def __init__(self, db: Session):
        """
        Initialize Job repository.
        
        Args:
            db: Database session
        """
        self.db = db
    
    def get_job(self, job_id: str):
        """
        Get Job by ID.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job object or None
        """
        try:
            from src.database.new_models import Job
            return self.db.query(Job).filter(Job.id == job_id).first()
        except Exception as e:
            logger.error(f"Error getting Job {job_id}: {e}")
            return None
    
    def get_all_jobs(self, with_embeddings: bool = False):
        """
        Get all jobs from database.
        
        Args:
            with_embeddings: If True, only return jobs that have all embeddings
            
        Returns:
            List of Job objects
        """
        try:
            from src.database.new_models import Job
            query = self.db.query(Job)
            if with_embeddings:
                query = query.filter(
                    Job.title_embedding.isnot(None),
                    Job.skills_embedding.isnot(None),
                    Job.requirement_embedding.isnot(None)
                )
            return query.all()
        except Exception as e:
            logger.error(f"Error getting all jobs: {e}")
            return []
    
    def update_job_embeddings(
        self,
        job_id: str,
        title_embedding: List[float],
        skills_embedding: List[float],
        requirement_embedding: List[float],
        content_hash: Optional[str] = None,
        **extra_fields
    ):
        """
        Update Job embeddings and other fields in the main Job table.
        
        Args:
            job_id: Job ID
            title_embedding: Title embedding vector (list of floats)
            skills_embedding: Skills embedding vector (list of floats)
            requirement_embedding: Requirement embedding vector (list of floats)
            content_hash: Optional content hash for change detection
            **extra_fields: Additional fields to update (title, description, location, industry, etc.)
        """
        try:
            from src.database.new_models import Job
            
            # Prepare update dictionary
            update_dict = {
                'title_embedding': title_embedding,
                'skills_embedding': skills_embedding,
                'requirement_embedding': requirement_embedding
            }
            
            if content_hash:
                update_dict['contentHash'] = content_hash
            
            # Add extra fields if provided
            allowed_fields = ['title', 'description', 'location', 'industry', 'salary']
            for field, value in extra_fields.items():
                if field in allowed_fields and value is not None:
                    update_dict[field] = value
            
            # Update using SQLAlchemy
            self.db.query(Job).filter(Job.id == job_id).update(update_dict)
            
            logger.debug(f"Updated embeddings and fields for Job {job_id}")
        except Exception as e:
            logger.error(f"Error updating Job embeddings {job_id}: {e}")
            raise
    
    def create_or_update_job(
        self,
        job_id: str,
        company_id: str,
        title: str,
        description: Optional[str] = None,
        location: Optional[str] = None,
        industry: Optional[str] = None,
        **extra_fields
    ):
        """
        Create or update Job record.
        
        Args:
            job_id: Job ID
            company_id: Company ID
            title: Job title (required)
            description: Job description
            location: Job location
            industry: Industry
            **extra_fields: Additional fields
        """
        try:
            from src.database.new_models import Job
            from datetime import datetime
            import uuid
            
            job = self.get_job(job_id)
            if not job:
                # Create new Job
                job = Job(
                    id=job_id,
                    companyId=company_id,
                    title=title,
                    description=description,
                    location=location,
                    industry=industry,
                    urgent=False,
                    applicationCount=0,
                    createdAt=datetime.now(),
                    updatedAt=datetime.now(),
                    status=1  # Assuming 1 is active status
                )
                self.db.add(job)
                logger.debug(f"Created new Job {job_id}")
            else:
                # Update existing Job
                update_dict = {}
                if title is not None:
                    update_dict['title'] = title
                if description is not None:
                    update_dict['description'] = description
                if location is not None:
                    update_dict['location'] = location
                if industry is not None:
                    update_dict['industry'] = industry
                
                if update_dict:
                    update_dict['updatedAt'] = datetime.now()
                    self.db.query(Job).filter(Job.id == job_id).update(update_dict)
                    logger.debug(f"Updated Job {job_id}")
        except Exception as e:
            logger.error(f"Error creating/updating Job {job_id}: {e}")
            raise
