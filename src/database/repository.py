"""Repository for database operations."""
from typing import List, Optional, Dict
from sqlalchemy.orm import Session
from sqlalchemy import func
import numpy as np
from .models import (
    JobDescriptionEmbedding, 
    CandidateEmbedding,
    ProcessedCandidateRecommendation
)


class EmbeddingRepository:
    """Repository for embedding operations."""
    
    def __init__(self, db: Session):
        self.db = db
    
    # Job Description Operations
    def create_jd_embedding(
        self,
        job_id: str,
        title: str,
        description: str,
        embedding: List[float],
        company: Optional[str] = None,
        requirements: Optional[str] = None,
        location: Optional[str] = None
    ) -> JobDescriptionEmbedding:
        """Create a new job description embedding."""
        jd_embedding = JobDescriptionEmbedding(
            job_id=job_id,
            title=title,
            company=company,
            description=description,
            requirements=requirements,
            location=location,
            embedding=embedding
        )
        self.db.add(jd_embedding)
        self.db.commit()
        self.db.refresh(jd_embedding)
        return jd_embedding
    
    def create_jd_embeddings_batch(
        self,
        embeddings_data: List[dict],
        replace_existing: bool = True
    ) -> int:
        """Create or update multiple job description embeddings in a batch.
        
        Args:
            embeddings_data: List of dicts with keys: job_id, title, description, 
                           embedding, company, requirements, location
            replace_existing: If True, replace existing embeddings. If False, skip existing.
        
        Returns:
            Number of embeddings created/updated
        
        Raises:
            Exception: If database error occurs
        """
        if not embeddings_data:
            return 0
        
        job_ids = [str(data['job_id']) for data in embeddings_data]
        
        if replace_existing:
            # Delete existing records first, then insert new ones
            existing_records = self.db.query(JobDescriptionEmbedding).filter(
                JobDescriptionEmbedding.job_id.in_(job_ids)
            ).all()
            if existing_records:
                for record in existing_records:
                    self.db.delete(record)
                self.db.commit()
        
        # Get existing job_ids to skip duplicates if not replacing
        if not replace_existing:
            existing_job_ids = set(
                self.db.query(JobDescriptionEmbedding.job_id)
                .filter(JobDescriptionEmbedding.job_id.in_(job_ids))
                .all()
            )
            existing_job_ids = {job_id[0] for job_id in existing_job_ids}
            
            # Filter out existing records
            embeddings_data = [
                data for data in embeddings_data 
                if str(data['job_id']) not in existing_job_ids
            ]
        
        if not embeddings_data:
            return 0
        
        jd_embeddings = []
        for data in embeddings_data:
            jd_embedding = JobDescriptionEmbedding(
                job_id=str(data['job_id']),
                title=data.get('title', ''),
                company=data.get('company'),
                description=data.get('description', ''),
                requirements=data.get('requirements'),
                location=data.get('location'),
                embedding=data['embedding']
            )
            jd_embeddings.append(jd_embedding)
        
        self.db.bulk_save_objects(jd_embeddings)
        self.db.commit()
        return len(jd_embeddings)
    
    def get_jd_embedding(self, job_id: str) -> Optional[JobDescriptionEmbedding]:
        """Get job description embedding by job_id."""
        return self.db.query(JobDescriptionEmbedding).filter(
            JobDescriptionEmbedding.job_id == job_id
        ).first()
    
    def get_all_jd_embeddings(self) -> List[JobDescriptionEmbedding]:
        """Get all job description embeddings."""
        return self.db.query(JobDescriptionEmbedding).all()
    
    def find_similar_jds(
        self,
        query_embedding: List[float],
        limit: int = 10
    ) -> List[JobDescriptionEmbedding]:
        """Find similar job descriptions using cosine similarity."""
        # Convert to numpy array for calculation
        query_vec = np.array(query_embedding)
        
        # Get all JD embeddings
        jds = self.get_all_jd_embeddings()
        
        # Calculate cosine similarities
        similarities = []
        for jd in jds:
            jd_vec = np.array(jd.embedding)
            cosine_sim = np.dot(query_vec, jd_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(jd_vec)
            )
            similarities.append((jd, cosine_sim))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [jd for jd, _ in similarities[:limit]]
    
    # Candidate Operations
    def create_candidate_embedding(
        self,
        candidate_id: str,
        embedding: List[float],
        name: Optional[str] = None,
        email: Optional[str] = None,
        skills: Optional[str] = None,
        experience: Optional[str] = None,
        education: Optional[str] = None,
        summary: Optional[str] = None,
        resume_text: Optional[str] = None
    ) -> CandidateEmbedding:
        """Create a new candidate embedding."""
        candidate_embedding = CandidateEmbedding(
            candidate_id=candidate_id,
            name=name,
            email=email,
            skills=skills,
            experience=experience,
            education=education,
            summary=summary,
            resume_text=resume_text,
            embedding=embedding
        )
        self.db.add(candidate_embedding)
        self.db.commit()
        self.db.refresh(candidate_embedding)
        return candidate_embedding
    
    def create_candidate_embeddings_batch(
        self,
        embeddings_data: List[dict],
        replace_existing: bool = True
    ) -> int:
        """Create or update multiple candidate embeddings in a batch.
        
        Args:
            embeddings_data: List of dicts with keys: candidate_id, embedding,
                           name, email, skills, experience, education, summary, resume_text
            replace_existing: If True, replace existing embeddings. If False, skip existing.
        
        Returns:
            Number of embeddings created/updated
        
        Raises:
            Exception: If database error occurs
        """
        if not embeddings_data:
            return 0
        
        candidate_ids = [str(data['candidate_id']) for data in embeddings_data]
        
        if replace_existing:
            # Delete existing records first, then insert new ones
            existing_records = self.db.query(CandidateEmbedding).filter(
                CandidateEmbedding.candidate_id.in_(candidate_ids)
            ).all()
            if existing_records:
                for record in existing_records:
                    self.db.delete(record)
                self.db.commit()
        
        # Get existing candidate_ids to skip duplicates if not replacing
        if not replace_existing:
            existing_candidate_ids = set(
                self.db.query(CandidateEmbedding.candidate_id)
                .filter(CandidateEmbedding.candidate_id.in_(candidate_ids))
                .all()
            )
            existing_candidate_ids = {candidate_id[0] for candidate_id in existing_candidate_ids}
            
            # Filter out existing records
            embeddings_data = [
                data for data in embeddings_data 
                if str(data['candidate_id']) not in existing_candidate_ids
            ]
        
        if not embeddings_data:
            return 0
        
        candidate_embeddings = []
        for data in embeddings_data:
            # Helper function to convert empty strings and 'None' strings to None
            def clean_value(value):
                if value is None:
                    return None
                
                # Check for pandas/numpy NaN
                try:
                    import pandas as pd
                    if pd.isna(value):
                        return None
                except (TypeError, ValueError, ImportError):
                    pass
                
                # Convert to string and check for NaN/None strings
                if isinstance(value, str):
                    value = value.strip()
                    str_lower = value.lower()
                    if value == '' or str_lower == 'none' or str_lower == 'nan' or str_lower == 'null':
                        return None
                    return value
                
                return value
            
            candidate_id = str(data.get('candidate_id', ''))
            name = clean_value(data.get('name'))
            email = clean_value(data.get('email'))
            skills = clean_value(data.get('skills'))
            experience = clean_value(data.get('experience'))
            education = clean_value(data.get('education'))
            summary = clean_value(data.get('summary'))
            resume_text = clean_value(data.get('resume_text'))
            embedding = data.get('embedding')
            
            # Truncate name and email to match database schema limits (String(200))
            if name and len(name) > 200:
                name = name[:200]
            if email and len(email) > 200:
                email = email[:200]
            
            # Validate that we have at least candidate_id and embedding
            if not candidate_id or embedding is None:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Skipping invalid record: candidate_id={candidate_id}, embedding={'present' if embedding else 'missing'}")
                continue
            
            candidate_embedding = CandidateEmbedding(
                candidate_id=candidate_id,
                name=name,
                email=email,
                skills=skills,
                experience=experience,
                education=education,
                summary=summary,
                resume_text=resume_text,
                embedding=embedding
            )
            candidate_embeddings.append(candidate_embedding)
        
        self.db.bulk_save_objects(candidate_embeddings)
        self.db.commit()
        return len(candidate_embeddings)
    
    def get_candidate_embedding(self, candidate_id: str) -> Optional[CandidateEmbedding]:
        """Get candidate embedding by candidate_id."""
        return self.db.query(CandidateEmbedding).filter(
            CandidateEmbedding.candidate_id == candidate_id
        ).first()
    
    def get_all_candidate_embeddings(self) -> List[CandidateEmbedding]:
        """Get all candidate embeddings."""
        return self.db.query(CandidateEmbedding).all()
    
    def find_similar_candidates(
        self,
        query_embedding: List[float],
        limit: int = 10
    ) -> List[CandidateEmbedding]:
        """Find similar candidates using cosine similarity."""
        # Convert to numpy array for calculation
        query_vec = np.array(query_embedding)
        
        # Get all candidate embeddings
        candidates = self.get_all_candidate_embeddings()
        
        # Calculate cosine similarities
        similarities = []
        for candidate in candidates:
            candidate_vec = np.array(candidate.embedding)
            cosine_sim = np.dot(query_vec, candidate_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(candidate_vec)
            )
            similarities.append((candidate, cosine_sim))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [candidate for candidate, _ in similarities[:limit]]
    
    # Recommendation Operations
    def recommend_jobs_for_candidate(
        self,
        candidate_id: str,
        limit: int = 10
    ) -> List[JobDescriptionEmbedding]:
        """Recommend jobs for a candidate based on their embedding."""
        candidate = self.get_candidate_embedding(candidate_id)
        if not candidate:
            return []
        
        return self.find_similar_jds(candidate.embedding, limit=limit)
    
    def recommend_candidates_for_job(
        self,
        job_id: str,
        limit: int = 10
    ) -> List[CandidateEmbedding]:
        """Recommend candidates for a job based on job embedding."""
        job = self.get_jd_embedding(job_id)
        if not job:
            return []
        
        return self.find_similar_candidates(job.embedding, limit=limit)
    
    # Processed Candidate Recommendations Operations
    def save_processed_recommendations(
        self,
        candidate_id: str,
        recommendations: List[Dict],
        replace_existing: bool = True
    ) -> int:
        """
        Save processed candidate recommendations (top 10 jobs).
        
        Args:
            candidate_id: Candidate ID
            recommendations: List of recommendation dicts with keys:
                           job_id, similarity_score, skills_similarity,
                           experience_similarity, desired_job_similarity
            replace_existing: If True, replace existing recommendations for this candidate
        
        Returns:
            Number of recommendations saved
        """
        if replace_existing:
            # Delete existing recommendations for this candidate
            existing = self.db.query(ProcessedCandidateRecommendation).filter(
                ProcessedCandidateRecommendation.candidate_id == candidate_id
            ).all()
            for rec in existing:
                self.db.delete(rec)
            self.db.commit()
        
        # Create new recommendations
        processed_recs = []
        for rank, rec in enumerate(recommendations[:10], 1):  # Top 10 only
            processed_rec = ProcessedCandidateRecommendation(
                candidate_id=candidate_id,
                job_id=str(rec['job_id']),
                similarity_score=rec.get('similarity_score', 0.0),
                skills_similarity=rec.get('field_similarities', {}).get('skills'),
                experience_similarity=rec.get('field_similarities', {}).get('experience'),
                desired_job_similarity=rec.get('field_similarities', {}).get('desired_job'),
                rank=rank
            )
            processed_recs.append(processed_rec)
        
        self.db.bulk_save_objects(processed_recs)
        self.db.commit()
        return len(processed_recs)
    
    def get_processed_recommendations(
        self,
        candidate_id: str
    ) -> List[ProcessedCandidateRecommendation]:
        """Get processed recommendations for a candidate."""
        return self.db.query(ProcessedCandidateRecommendation).filter(
            ProcessedCandidateRecommendation.candidate_id == candidate_id
        ).order_by(ProcessedCandidateRecommendation.rank).all()
    
    def has_processed_recommendations(self, candidate_id: str) -> bool:
        """Check if candidate has processed recommendations."""
        count = self.db.query(ProcessedCandidateRecommendation).filter(
            ProcessedCandidateRecommendation.candidate_id == candidate_id
        ).count()
        return count > 0
    
    def save_processed_recommendations_batch(
        self,
        all_recommendations: Dict[str, List[Dict]],
        replace_existing: bool = True
    ) -> int:
        """
        Save processed recommendations for multiple candidates in batch.
        
        Args:
            all_recommendations: Dict of candidate_id -> list of recommendations
            replace_existing: If True, replace existing recommendations
        
        Returns:
            Total number of recommendations saved
        """
        total_saved = 0
        
        for candidate_id, recommendations in all_recommendations.items():
            saved = self.save_processed_recommendations(
                candidate_id=candidate_id,
                recommendations=recommendations,
                replace_existing=replace_existing
            )
            total_saved += saved
        
        return total_saved

