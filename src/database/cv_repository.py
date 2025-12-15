"""CV Repository for database operations."""
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import update
import logging

logger = logging.getLogger(__name__)


class CVRepository:
    """Repository for CV database operations."""
    
    def __init__(self, db: Session):
        """
        Initialize CV repository.
        
        Args:
            db: Database session
        """
        self.db = db
    
    def get_cv(self, cv_id: str):
        """
        Get CV by ID.
        
        Args:
            cv_id: CV ID
            
        Returns:
            CV object or None
        """
        try:
            from src.database.new_models import CV
            return self.db.query(CV).filter(CV.id == cv_id).first()
        except Exception as e:
            logger.error(f"Error getting CV {cv_id}: {e}")
            return None
    
    def update_cv_embeddings(
        self,
        cv_id: str,
        title_embedding: List[float],
        skills_embedding: List[float],
        experience_embedding: List[float],
        content_hash: Optional[str] = None,
        **extra_fields
    ):
        """
        Update CV embeddings and other fields in the main CV table.
        
        Args:
            cv_id: CV ID
            title_embedding: Title embedding vector (list of floats)
            skills_embedding: Skills embedding vector (list of floats)
            experience_embedding: Experience embedding vector (list of floats)
            content_hash: Optional content hash for change detection
            **extra_fields: Additional fields to update (title, fullName, summary, objective, etc.)
        """
        try:
            from src.database.new_models import CV
            
            # Prepare update dictionary
            update_dict = {
                'title_embedding': title_embedding,
                'skills_embedding': skills_embedding,
                'experience_embedding': experience_embedding
            }
            
            if content_hash:
                update_dict['contentHash'] = content_hash
            
            # Add extra fields if provided
            allowed_fields = ['title', 'fullName', 'summary', 'objective', 'address', 
                            'currentPosition', 'gender', 'nationality']
            for field, value in extra_fields.items():
                if field in allowed_fields and value is not None:
                    update_dict[field] = value
            
            # Update using SQLAlchemy
            self.db.query(CV).filter(CV.id == cv_id).update(update_dict)
            
            logger.debug(f"Updated embeddings and fields for CV {cv_id}")
        except Exception as e:
            logger.error(f"Error updating CV embeddings {cv_id}: {e}")
            raise
    
    def create_or_update_cv(
        self,
        cv_id: str,
        user_id: str,
        title: Optional[str] = None,
        full_name: Optional[str] = None,
        summary: Optional[str] = None,
        objective: Optional[str] = None,
        **extra_fields
    ):
        """
        Create or update CV record.
        
        Args:
            cv_id: CV ID
            user_id: User ID
            title: CV title
            full_name: Full name
            summary: Summary
            objective: Objective
            **extra_fields: Additional fields
        """
        try:
            from src.database.new_models import CV
            from datetime import datetime
            import uuid
            
            cv = self.get_cv(cv_id)
            if not cv:
                # Create new CV
                cv = CV(
                    id=cv_id,
                    userId=user_id,
                    title=title,
                    fullName=full_name,
                    summary=summary,
                    objective=objective,
                    isMain=False,
                    isOpenForJob=True,
                    createdAt=datetime.now(),
                    updatedAt=datetime.now()
                )
                self.db.add(cv)
                logger.debug(f"Created new CV {cv_id}")
            else:
                # Update existing CV
                update_dict = {}
                if title is not None:
                    update_dict['title'] = title
                if full_name is not None:
                    update_dict['fullName'] = full_name
                if summary is not None:
                    update_dict['summary'] = summary
                if objective is not None:
                    update_dict['objective'] = objective
                
                if update_dict:
                    update_dict['updatedAt'] = datetime.now()
                    self.db.query(CV).filter(CV.id == cv_id).update(update_dict)
                    logger.debug(f"Updated CV {cv_id}")
        except Exception as e:
            logger.error(f"Error creating/updating CV {cv_id}: {e}")
            raise
