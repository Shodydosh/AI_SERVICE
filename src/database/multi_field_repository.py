"""Repository for multi-field embeddings operations."""
from typing import List, Dict, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func
from src.database.models import (
    JobDescriptionMultiEmbedding,
    CandidateMultiEmbedding
)
import logging

logger = logging.getLogger(__name__)


class MultiFieldEmbeddingRepository:
    """Repository for multi-field embedding operations."""
    
    def __init__(self, db: Session):
        """
        Initialize repository.
        
        Args:
            db: Database session
        """
        self.db = db
    
    # ========================================================================
    # Job Description Multi-Embedding Operations
    # ========================================================================
    
    def create_job_multi_embedding(
        self,
        job_id: str,
        title: str,
        skills: Optional[str],
        requirement: Optional[str],
        title_embedding: List[float],
        skills_embedding: List[float],
        requirement_embedding: List[float],
        company: Optional[str] = None,
        location: Optional[str] = None,
        replace_existing: bool = True
    ) -> JobDescriptionMultiEmbedding:
        """
        Create or update a job multi-embedding.
        
        Args:
            job_id: Job ID
            title: Job title
            skills: Required skills
            requirement: Job requirements
            title_embedding: Title embedding vector
            skills_embedding: Skills embedding vector
            requirement_embedding: Requirement embedding vector
            company: Company name
            location: Job location
            replace_existing: If True, replace existing record
        
        Returns:
            Created or updated JobDescriptionMultiEmbedding
        """
        if replace_existing:
            # Delete existing record
            existing = self.db.query(JobDescriptionMultiEmbedding).filter(
                JobDescriptionMultiEmbedding.job_id == job_id
            ).first()
            if existing:
                self.db.delete(existing)
                self.db.flush()
        
        job_embedding = JobDescriptionMultiEmbedding(
            job_id=job_id,
            title=title,
            skills=skills,
            requirement=requirement,
            title_embedding=title_embedding,
            skills_embedding=skills_embedding,
            requirement_embedding=requirement_embedding,
            company=company,
            location=location
        )
        
        self.db.add(job_embedding)
        self.db.commit()
        self.db.refresh(job_embedding)
        
        return job_embedding
    
    def create_job_multi_embeddings_batch(
        self,
        jobs_data: List[Dict],
        replace_existing: bool = True
    ) -> int:
        """
        Create multiple job multi-embeddings in batch.
        
        Args:
            jobs_data: List of dicts with keys:
                job_id, title, skills, requirement, title_embedding,
                skills_embedding, requirement_embedding, company, location
            replace_existing: If True, replace existing records
        
        Returns:
            Number of records saved
        """
        if not jobs_data:
            return 0
        
        if replace_existing:
            # Delete existing records
            job_ids = [job['job_id'] for job in jobs_data]
            self.db.query(JobDescriptionMultiEmbedding).filter(
                JobDescriptionMultiEmbedding.job_id.in_(job_ids)
            ).delete(synchronize_session=False)
            self.db.flush()
        
        # Validate and clean data
        validated_data = []
        for i, job_data in enumerate(jobs_data):
            try:
                job_id = str(job_data.get('job_id', '')).strip()
                if not job_id:
                    logger.warning(f"Skipping job at index {i}: missing job_id")
                    continue
                
                # Validate embeddings
                if not job_data.get('title_embedding') or not isinstance(job_data.get('title_embedding'), list):
                    logger.warning(f"Skipping job {job_id}: missing or invalid title_embedding")
                    continue
                if not job_data.get('skills_embedding') or not isinstance(job_data.get('skills_embedding'), list):
                    logger.warning(f"Skipping job {job_id}: missing or invalid skills_embedding")
                    continue
                if not job_data.get('requirement_embedding') or not isinstance(job_data.get('requirement_embedding'), list):
                    logger.warning(f"Skipping job {job_id}: missing or invalid requirement_embedding")
                    continue
                
                # Truncate text fields if too long
                def truncate_text(text, max_length=10000):
                    if text is None:
                        return None
                    text_str = str(text)
                    if len(text_str) > max_length:
                        return text_str[:max_length]
                    return text_str
                
                validated_data.append({
                    'job_id': job_id,
                    'title': truncate_text(job_data.get('title'), max_length=500),
                    'skills': truncate_text(job_data.get('skills')),
                    'requirement': truncate_text(job_data.get('requirement')),
                    'title_embedding': job_data['title_embedding'],
                    'skills_embedding': job_data['skills_embedding'],
                    'requirement_embedding': job_data['requirement_embedding'],
                    'company': truncate_text(job_data.get('company'), max_length=200),
                    'location': truncate_text(job_data.get('location'), max_length=200)
                })
            except Exception as e:
                logger.error(f"Error validating job data at index {i}: {e}")
                continue
        
        if not validated_data:
            logger.warning("No valid job data to save")
            return 0
        
        # Insert in smaller batches
        batch_insert_size = 50
        total_inserted = 0
        
        for batch_start in range(0, len(validated_data), batch_insert_size):
            batch_end = min(batch_start + batch_insert_size, len(validated_data))
            batch = validated_data[batch_start:batch_end]
            
            try:
                job_embeddings = []
                for job_data in batch:
                    job_embedding = JobDescriptionMultiEmbedding(
                        job_id=job_data['job_id'],
                        title=job_data['title'],
                        skills=job_data['skills'],
                        requirement=job_data['requirement'],
                        title_embedding=job_data['title_embedding'],
                        skills_embedding=job_data['skills_embedding'],
                        requirement_embedding=job_data['requirement_embedding'],
                        company=job_data['company'],
                        location=job_data['location']
                    )
                    job_embeddings.append(job_embedding)
                
                self.db.bulk_save_objects(job_embeddings)
                self.db.commit()
                total_inserted += len(job_embeddings)
                
            except Exception as e:
                logger.error(f"Error inserting job batch {batch_start//batch_insert_size + 1}: {e}")
                self.db.rollback()
                # Fallback to one-by-one insert
                for job_data in batch:
                    try:
                        job_embedding = JobDescriptionMultiEmbedding(
                            job_id=job_data['job_id'],
                            title=job_data['title'],
                            skills=job_data['skills'],
                            requirement=job_data['requirement'],
                            title_embedding=job_data['title_embedding'],
                            skills_embedding=job_data['skills_embedding'],
                            requirement_embedding=job_data['requirement_embedding'],
                            company=job_data['company'],
                            location=job_data['location']
                        )
                        self.db.add(job_embedding)
                        self.db.commit()
                        total_inserted += 1
                    except Exception as single_error:
                        logger.error(f"Error inserting job {job_data.get('job_id')}: {single_error}")
                        self.db.rollback()
                        continue
        
        return total_inserted
    
    def get_job_multi_embedding(self, job_id: str) -> Optional[JobDescriptionMultiEmbedding]:
        """Get job multi-embedding by job_id."""
        return self.db.query(JobDescriptionMultiEmbedding).filter(
            JobDescriptionMultiEmbedding.job_id == job_id
        ).first()
    
    def get_all_job_multi_embeddings(self) -> List[JobDescriptionMultiEmbedding]:
        """Get all job multi-embeddings."""
        return self.db.query(JobDescriptionMultiEmbedding).all()
    
    def count_job_multi_embeddings(self) -> int:
        """Count total job multi-embeddings."""
        return self.db.query(JobDescriptionMultiEmbedding).count()
    
    # ========================================================================
    # Candidate Multi-Embedding Operations
    # ========================================================================
    
    def create_candidate_multi_embedding(
        self,
        candidate_id: str,
        title: Optional[str],
        skills: Optional[str],
        experience: Optional[str],
        title_embedding: List[float],
        skills_embedding: List[float],
        experience_embedding: List[float],
        name: Optional[str] = None,
        email: Optional[str] = None,
        replace_existing: bool = True
    ) -> CandidateMultiEmbedding:
        """
        Create or update a candidate multi-embedding.
        
        Args:
            candidate_id: Candidate ID
            title: Desired job title or current job title
            skills: Candidate skills
            experience: Work experience
            title_embedding: Title embedding vector
            skills_embedding: Skills embedding vector
            experience_embedding: Experience embedding vector
            name: Candidate name
            email: Candidate email
            replace_existing: If True, replace existing record
        
        Returns:
            Created or updated CandidateMultiEmbedding
        """
        if replace_existing:
            # Delete existing record
            existing = self.db.query(CandidateMultiEmbedding).filter(
                CandidateMultiEmbedding.candidate_id == candidate_id
            ).first()
            if existing:
                self.db.delete(existing)
                self.db.flush()
        
        candidate_embedding = CandidateMultiEmbedding(
            candidate_id=candidate_id,
            title=title,
            skills=skills,
            experience=experience,
            title_embedding=title_embedding,
            skills_embedding=skills_embedding,
            experience_embedding=experience_embedding,
            name=name,
            email=email
        )
        
        self.db.add(candidate_embedding)
        self.db.commit()
        self.db.refresh(candidate_embedding)
        
        return candidate_embedding
    
    def create_candidate_multi_embeddings_batch(
        self,
        candidates_data: List[Dict],
        replace_existing: bool = True
    ) -> int:
        """
        Create multiple candidate multi-embeddings in batch.
        
        Args:
            candidates_data: List of dicts with keys:
                candidate_id, title, skills, experience, title_embedding,
                skills_embedding, experience_embedding, name, email
            replace_existing: If True, replace existing records
        
        Returns:
            Number of records saved
        """
        if not candidates_data:
            return 0
        
        # Validate and clean data
        validated_data = []
        for i, candidate_data in enumerate(candidates_data):
            try:
                # Validate required fields
                candidate_id = str(candidate_data.get('candidate_id', '')).strip()
                if not candidate_id:
                    logger.warning(f"Skipping candidate at index {i}: missing candidate_id")
                    continue
                
                # Validate embeddings exist and are lists
                title_emb = candidate_data.get('title_embedding')
                skills_emb = candidate_data.get('skills_embedding')
                experience_emb = candidate_data.get('experience_embedding')
                
                if not title_emb or not isinstance(title_emb, list):
                    logger.warning(f"Skipping candidate {candidate_id}: missing or invalid title_embedding")
                    continue
                if not skills_emb or not isinstance(skills_emb, list):
                    logger.warning(f"Skipping candidate {candidate_id}: missing or invalid skills_embedding")
                    continue
                if not experience_emb or not isinstance(experience_emb, list):
                    logger.warning(f"Skipping candidate {candidate_id}: missing or invalid experience_embedding")
                    continue
                
                # Truncate text fields if too long
                def truncate_text(text, max_length=10000):
                    if text is None:
                        return None
                    text_str = str(text)
                    if len(text_str) > max_length:
                        logger.warning(f"Truncating text field for candidate {candidate_id} (length: {len(text_str)})")
                        return text_str[:max_length]
                    return text_str
                
                validated_data.append({
                    'candidate_id': candidate_id,
                    'title': truncate_text(candidate_data.get('title'), max_length=500),
                    'skills': truncate_text(candidate_data.get('skills')),
                    'experience': truncate_text(candidate_data.get('experience')),
                    'title_embedding': title_emb,
                    'skills_embedding': skills_emb,
                    'experience_embedding': experience_emb,
                    'name': truncate_text(candidate_data.get('name'), max_length=200),
                    'email': truncate_text(candidate_data.get('email'), max_length=200)
                })
            except Exception as e:
                logger.error(f"Error validating candidate data at index {i}: {e}")
                continue
        
        if not validated_data:
            logger.warning("No valid candidate data to save")
            return 0
        
        if replace_existing:
            # Delete existing records
            candidate_ids = [c['candidate_id'] for c in validated_data]
            try:
                self.db.query(CandidateMultiEmbedding).filter(
                    CandidateMultiEmbedding.candidate_id.in_(candidate_ids)
                ).delete(synchronize_session=False)
                self.db.flush()
            except Exception as e:
                logger.warning(f"Error deleting existing records: {e}")
                self.db.rollback()
        
        # Insert in smaller batches to avoid errors
        batch_insert_size = 50
        total_inserted = 0
        
        for batch_start in range(0, len(validated_data), batch_insert_size):
            batch_end = min(batch_start + batch_insert_size, len(validated_data))
            batch = validated_data[batch_start:batch_end]
            
            try:
                candidate_embeddings = []
                for candidate_data in batch:
                    candidate_embedding = CandidateMultiEmbedding(
                        candidate_id=candidate_data['candidate_id'],
                        title=candidate_data['title'],
                        skills=candidate_data['skills'],
                        experience=candidate_data['experience'],
                        title_embedding=candidate_data['title_embedding'],
                        skills_embedding=candidate_data['skills_embedding'],
                        experience_embedding=candidate_data['experience_embedding'],
                        name=candidate_data['name'],
                        email=candidate_data['email']
                    )
                    candidate_embeddings.append(candidate_embedding)
                
                self.db.bulk_save_objects(candidate_embeddings)
                self.db.commit()
                total_inserted += len(candidate_embeddings)
                
            except Exception as e:
                logger.error(f"Error inserting batch {batch_start//batch_insert_size + 1}: {e}")
                self.db.rollback()
                # Try inserting one by one as fallback
                for candidate_data in batch:
                    try:
                        candidate_embedding = CandidateMultiEmbedding(
                            candidate_id=candidate_data['candidate_id'],
                            title=candidate_data['title'],
                            skills=candidate_data['skills'],
                            experience=candidate_data['experience'],
                            title_embedding=candidate_data['title_embedding'],
                            skills_embedding=candidate_data['skills_embedding'],
                            experience_embedding=candidate_data['experience_embedding'],
                            name=candidate_data['name'],
                            email=candidate_data['email']
                        )
                        self.db.add(candidate_embedding)
                        self.db.commit()
                        total_inserted += 1
                    except Exception as single_error:
                        logger.error(f"Error inserting candidate {candidate_data.get('candidate_id')}: {single_error}")
                        self.db.rollback()
                        continue
        
        return total_inserted
    
    def get_candidate_multi_embedding(self, candidate_id: str) -> Optional[CandidateMultiEmbedding]:
        """Get candidate multi-embedding by candidate_id."""
        return self.db.query(CandidateMultiEmbedding).filter(
            CandidateMultiEmbedding.candidate_id == candidate_id
        ).first()
    
    def get_all_candidate_multi_embeddings(self) -> List[CandidateMultiEmbedding]:
        """Get all candidate multi-embeddings."""
        return self.db.query(CandidateMultiEmbedding).all()
    
    def count_candidate_multi_embeddings(self) -> int:
        """Count total candidate multi-embeddings."""
        return self.db.query(CandidateMultiEmbedding).count()
    
    # ========================================================================
    # Search Operations
    # ========================================================================
    
    def find_similar_jobs_by_field(
        self,
        query_embedding: List[float],
        field_type: str,
        limit: int = 100
    ) -> List[Tuple[str, float]]:
        """
        Find similar jobs by field embedding using cosine similarity.
        
        Args:
            query_embedding: Query vector
            field_type: 'title', 'skills', or 'requirement'
            limit: Maximum number of results
        
        Returns:
            List of (job_id, similarity_score) tuples
        """
        import numpy as np
        
        if not query_embedding:
            logger.warning(f"Query embedding is empty for field_type: {field_type}")
            return []
        
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        
        if query_norm == 0 or query_norm < 1e-10:
            logger.warning(f"Query embedding norm is zero or too small: {query_norm} for field_type: {field_type}")
            return []
        
        query_vec = query_vec / query_norm
        
        # Get appropriate field embedding
        if field_type == 'title':
            field_name = 'title_embedding'
        elif field_type == 'skills':
            field_name = 'skills_embedding'
        elif field_type == 'requirement':
            field_name = 'requirement_embedding'
        else:
            raise ValueError(f"Unknown field type: {field_type}")
        
        # Get all jobs with this field
        all_jobs = self.db.query(JobDescriptionMultiEmbedding).all()
        
        if not all_jobs:
            logger.warning(f"No jobs found in database")
            return []
        
        logger.debug(f"Searching in {len(all_jobs)} jobs for field_type: {field_type}")
        
        # Calculate similarities
        results = []
        zero_embedding_count = 0
        for job in all_jobs:
            job_emb = getattr(job, field_name, None)
            if not job_emb:
                zero_embedding_count += 1
                continue
                
            job_emb = np.array(job_emb, dtype=np.float32)
            job_norm = np.linalg.norm(job_emb)
            
            if job_norm == 0 or job_norm < 1e-10:
                zero_embedding_count += 1
                continue
            
            job_emb = job_emb / job_norm
            
            # Cosine similarity
            similarity = np.dot(query_vec, job_emb)
            similarity = max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
            results.append((job.job_id, float(similarity)))
        
        if zero_embedding_count > 0:
            logger.debug(f"Found {zero_embedding_count} jobs with zero/none embedding for field {field_type}")
        
        if not results:
            logger.warning(f"No valid job embeddings found for field_type: {field_type}")
            return []
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Found {len(results)} jobs with valid embeddings, returning top {limit}")
        if results:
            logger.debug(f"Top similarity: {results[0][1]:.4f}, Lowest: {results[-1][1]:.4f}")
        
        return results[:limit]
    
    def find_similar_jobs_by_field_filtered(
        self,
        query_embedding: List[float],
        field_type: str,
        job_ids: List[str],
        limit: int = 100
    ) -> List[Tuple[str, float]]:
        """
        Find similar jobs by field embedding, filtered to specific job IDs.
        
        Args:
            query_embedding: Query vector
            field_type: 'title', 'skills', or 'requirement'
            job_ids: List of job IDs to filter to
            limit: Maximum number of results
        
        Returns:
            List of (job_id, similarity_score) tuples
        """
        import numpy as np
        
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        query_vec = query_vec / query_norm
        
        # Get appropriate field embedding
        if field_type == 'title':
            field_name = 'title_embedding'
        elif field_type == 'skills':
            field_name = 'skills_embedding'
        elif field_type == 'requirement':
            field_name = 'requirement_embedding'
        else:
            raise ValueError(f"Unknown field type: {field_type}")
        
        # Get jobs by IDs
        jobs = self.db.query(JobDescriptionMultiEmbedding).filter(
            JobDescriptionMultiEmbedding.job_id.in_(job_ids)
        ).all()
        
        # Calculate similarities
        results = []
        for job in jobs:
            job_emb = np.array(getattr(job, field_name), dtype=np.float32)
            job_norm = np.linalg.norm(job_emb)
            if job_norm == 0:
                continue
            job_emb = job_emb / job_norm
            
            # Cosine similarity
            similarity = np.dot(query_vec, job_emb)
            similarity = max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
            results.append((job.job_id, float(similarity)))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:limit]
