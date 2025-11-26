"""Service for matching candidates to jobs using embeddings."""
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
import logging
from src.database.repository import EmbeddingRepository
from src.embeddings.generator import EmbeddingGenerator
from src.data_processing.candidate_processor import CandidateProcessor
from src.vector_search.faiss_manager import FAISSIndexManager
from src.services.reranking_service import RerankingService
from pathlib import Path

logger = logging.getLogger(__name__)


class MatchingService:
    """Service for candidate-to-job matching."""
    
    def __init__(self, db: Session, use_faiss: bool = True, use_reranking: bool = True):
        """
        Initialize matching service.
        
        Args:
            db: Database session
            use_faiss: Whether to use FAISS for fast search
            use_reranking: Whether to use cross-encoder re-ranking for 90%+ similarity
        """
        self.db = db
        self.repository = EmbeddingRepository(db)
        self.embedding_generator = EmbeddingGenerator()
        self.use_faiss = use_faiss
        self.use_reranking = use_reranking
        self.faiss_manager = None
        self.reranker = None
        
        # Initialize cross-encoder re-ranker for 90%+ similarity
        if use_reranking:
            try:
                self.reranker = RerankingService(use_cross_encoder=True)
                logger.info("Cross-encoder re-ranking enabled for 90%+ similarity")
            except Exception as e:
                logger.warning(f"Could not initialize cross-encoder re-ranker: {e}")
                self.reranker = RerankingService(use_cross_encoder=False)
                logger.info("Using bi-encoder only (no cross-encoder)")
        
        if use_faiss:
            try:
                dimension = self.embedding_generator.get_embedding_dimension()
                self.faiss_manager = FAISSIndexManager(
                    dimension=dimension,
                    index_type="HNSW",  # Use HNSW for fast approximate search
                    index_params={
                        "ef_search": 64,  # Higher ef_search for better precision (optimized for k=15)
                        "ef_construction": 200,  # Higher construction for better index quality
                        "M": 32  # Number of connections (good balance)
                    },
                    normalize=True
                )
                
                # Try to load existing index
                jd_index_path = Path("indices/jd_index.faiss")
                if jd_index_path.exists():
                    self.faiss_manager.load_index(str(jd_index_path), dataset_type='jd')
                    logger.info("Loaded existing JD FAISS index")
                else:
                    logger.warning("FAISS index not found. Building from database...")
                    self.faiss_manager.build_index_from_db(db, dataset_type='jd')
            except Exception as e:
                logger.warning(f"Could not initialize FAISS: {e}. Falling back to database search.")
                self.use_faiss = False
    
    def find_jobs_for_candidate(
        self,
        candidate_id: str,
        top_k: int = 50,
        use_faiss: Optional[bool] = None,
        use_processed: bool = True
    ) -> List[Dict]:
        """
        Find top matching jobs for a candidate.
        
        Args:
            candidate_id: Candidate ID from database
            top_k: Number of top matches to return
            use_faiss: Override FAISS usage (None = use instance setting)
            use_processed: If True, check processed recommendations first (default: True)
        
        Returns:
            List of job matches with similarity scores
        """
        # Check processed recommendations first if enabled
        if use_processed and self.repository.has_processed_recommendations(candidate_id):
            logger.info(f"Using processed recommendations for candidate {candidate_id}")
            processed_recs = self.repository.get_processed_recommendations(candidate_id)
            
            # Get job details from database
            results = []
            for rec in processed_recs[:top_k]:
                job = self.repository.get_jd_embedding(rec.job_id)
                if job:
                    job_dict = {
                        "job_id": job.job_id,
                        "title": job.title,
                        "company": job.company,
                        "location": job.location,
                        "description": job.description[:500] if job.description else None,
                        "requirements": job.requirements[:300] if job.requirements else None,
                        "similarity_score": round(rec.similarity_score, 4),
                        "field_similarities": {
                            "skills": rec.skills_similarity,
                            "experience": rec.experience_similarity,
                            "desired_job": rec.desired_job_similarity
                        } if rec.skills_similarity else None,
                        "rank": rec.rank
                    }
                    results.append(job_dict)
            
            logger.info(f"Retrieved {len(results)} processed recommendations")
            return results
    
    def get_job_ids_for_candidate(
        self,
        candidate_id: str,
        top_k: int = 10
    ) -> List[str]:
        """
        Get top K job IDs for a candidate from processed recommendations.
        This is a fast method that only returns job IDs without embedding computation.
        
        Args:
            candidate_id: Candidate ID from database
            top_k: Number of top job IDs to return (default: 10)
        
        Returns:
            List of job IDs (strings)
        """
        if not self.repository.has_processed_recommendations(candidate_id):
            logger.warning(f"No processed recommendations found for candidate {candidate_id}")
            return []
        
        processed_recs = self.repository.get_processed_recommendations(candidate_id)
        job_ids = [rec.job_id for rec in processed_recs[:top_k]]
        
        logger.info(f"Retrieved {len(job_ids)} job IDs for candidate {candidate_id}")
        return job_ids
    
    def find_jobs_for_candidate(
        self,
        candidate_id: str,
        top_k: int = 50,
        use_faiss: Optional[bool] = None,
        use_processed: bool = True
    ) -> List[Dict]:
        """
        Find top matching jobs for a candidate.
        
        Args:
            candidate_id: Candidate ID from database
            top_k: Number of top matches to return
            use_faiss: Override FAISS usage (None = use instance setting)
            use_processed: If True, check processed recommendations first (default: True)
        
        Returns:
            List of job matches with similarity scores
        """
        # Check processed recommendations first if enabled
        if use_processed and self.repository.has_processed_recommendations(candidate_id):
            logger.info(f"Using processed recommendations for candidate {candidate_id}")
            processed_recs = self.repository.get_processed_recommendations(candidate_id)
            
            # Get job details from database
            results = []
            for rec in processed_recs[:top_k]:
                job = self.repository.get_jd_embedding(rec.job_id)
                if job:
                    job_dict = {
                        "job_id": job.job_id,
                        "title": job.title,
                        "company": job.company,
                        "location": job.location,
                        "description": job.description[:500] if job.description else None,
                        "requirements": job.requirements[:300] if job.requirements else None,
                        "similarity_score": round(rec.similarity_score, 4),
                        "field_similarities": {
                            "skills": rec.skills_similarity,
                            "experience": rec.experience_similarity,
                            "desired_job": rec.desired_job_similarity
                        } if rec.skills_similarity else None,
                        "rank": rec.rank
                    }
                    results.append(job_dict)
            
            logger.info(f"Retrieved {len(results)} processed recommendations")
            return results
        
        use_faiss = use_faiss if use_faiss is not None else self.use_faiss
        
        # Get candidate embedding from PostgreSQL
        candidate = self.repository.get_candidate_embedding(candidate_id)
        if not candidate:
            logger.error(f"Candidate {candidate_id} not found in database")
            return []
        
        candidate_embedding = candidate.embedding
        
        # Log vector information to console
        import numpy as np
        vector_array = np.array(candidate_embedding)
        logger.info("=" * 80)
        logger.info(f"MATCHING CANDIDATE FROM POSTGRESQL")
        logger.info("=" * 80)
        logger.info(f"Candidate ID: {candidate_id}")
        logger.info(f"Candidate Name: {candidate.name or 'N/A'}")
        logger.info(f"Candidate Email: {candidate.email or 'N/A'}")
        logger.info(f"Vector Source: PostgreSQL Database")
        logger.info(f"Vector Dimension: {len(candidate_embedding)}")
        logger.info(f"Vector Shape: {vector_array.shape}")
        logger.info(f"Vector (first 10 values): {candidate_embedding[:10]}")
        logger.info(f"Vector Norm: {np.linalg.norm(vector_array):.4f}")
        logger.info(f"Top K: {top_k}")
        logger.info(f"Using FAISS: {use_faiss}")
        logger.info("=" * 80)
        
        # Search for similar jobs
        if use_faiss and self.faiss_manager:
            logger.info(f"Using FAISS to find top {top_k} jobs for candidate {candidate_id}")
            results = self.faiss_manager.search(
                query_embedding=candidate_embedding,
                k=top_k,
                dataset_type='jd'
            )
            
            # Get job details from database
            jobs = []
            job_texts = []  # For re-ranking
            for job_id, similarity in results:
                job = self.repository.get_jd_embedding(job_id)
                if job:
                    job_dict = {
                        "job_id": job.job_id,
                        "title": job.title,
                        "company": job.company,
                        "location": job.location,
                        "description": job.description[:500] if job.description else None,
                        "requirements": job.requirements[:300] if job.requirements else None,
                        "similarity_score": round(similarity, 4)
                    }
                    jobs.append(job_dict)
                    # Prepare text for re-ranking
                    job_text = f"{job.title or ''} {job.description or ''} {job.requirements or ''}".strip()
                    job_texts.append(job_text)
            
            # Apply cross-encoder re-ranking for 90%+ similarity
            if self.use_reranking and self.reranker and len(jobs) > 0:
                logger.info("Applying cross-encoder re-ranking for improved similarity...")
                # Get candidate text for re-ranking
                candidate_text = f"{candidate.name or ''} {candidate.skills or ''} {candidate.experience or ''}".strip()
                initial_scores = [job["similarity_score"] for job in jobs]
                reranked_indices = self.reranker.rerank_matches(
                    query_text=candidate_text,
                    candidate_texts=job_texts,
                    initial_scores=initial_scores,
                    top_k=len(jobs)
                )
                
                # Reorder jobs based on re-ranking
                reranked_jobs = [jobs[idx] for idx, _ in reranked_indices]
                
                # Also apply exact matching boost (increased for maximum similarity)
                boosted_scores = self.reranker.boost_exact_matches(
                    query_text=candidate_text,
                    candidate_texts=job_texts,
                    scores=[job["similarity_score"] for job in reranked_jobs],
                    boost_factor=1.3  # 30% boost for exact matches (increased from 15%)
                )
                
                # Update scores
                for i, job in enumerate(reranked_jobs):
                    job["similarity_score"] = round(boosted_scores[i], 4)
                
                logger.info(f"Re-ranking complete. Top similarity: {max(boosted_scores)*100:.2f}%")
                return reranked_jobs
            
            return jobs
        else:
            # Fallback to database search
            logger.info(f"Using database search to find top {top_k} jobs for candidate {candidate_id}")
            jobs = self.repository.find_similar_jds(candidate_embedding, limit=top_k)
            
            # Calculate similarity scores
            import numpy as np
            candidate_vec = np.array(candidate_embedding)
            
            results = []
            job_texts = []  # For re-ranking
            for job in jobs:
                job_vec = np.array(job.embedding)
                similarity = np.dot(candidate_vec, job_vec) / (
                    np.linalg.norm(candidate_vec) * np.linalg.norm(job_vec)
                )
                
                job_dict = {
                    "job_id": job.job_id,
                    "title": job.title,
                    "company": job.company,
                    "location": job.location,
                    "description": job.description[:500] if job.description else None,
                    "requirements": job.requirements[:300] if job.requirements else None,
                    "similarity_score": round(float(similarity), 4)
                }
                results.append(job_dict)
                # Prepare text for re-ranking
                job_text = f"{job.title or ''} {job.description or ''} {job.requirements or ''}".strip()
                job_texts.append(job_text)
            
            # Apply cross-encoder re-ranking for 90%+ similarity
            if self.use_reranking and self.reranker and len(results) > 0:
                logger.info("Applying cross-encoder re-ranking for improved similarity...")
                # Get candidate text for re-ranking
                candidate_text = f"{candidate.name or ''} {candidate.skills or ''} {candidate.experience or ''}".strip()
                initial_scores = [job["similarity_score"] for job in results]
                reranked_indices = self.reranker.rerank_matches(
                    query_text=candidate_text,
                    candidate_texts=job_texts,
                    initial_scores=initial_scores,
                    top_k=len(results)
                )
                
                # Reorder results based on re-ranking
                reranked_results = [results[idx] for idx, _ in reranked_indices]
                
                # Also apply exact matching boost
                boosted_scores = self.reranker.boost_exact_matches(
                    query_text=candidate_text,
                    candidate_texts=job_texts,
                    scores=[job["similarity_score"] for job in reranked_results],
                    boost_factor=1.15  # 15% boost for exact matches
                )
                
                # Update scores
                for i, job in enumerate(reranked_results):
                    job["similarity_score"] = round(boosted_scores[i], 4)
                
                logger.info(f"Re-ranking complete. Top similarity: {max(boosted_scores)*100:.2f}%")
                return reranked_results
            
            return results
    
    def combine_candidate_fields(
        self,
        name: Optional[str] = None,
        skills: Optional[str] = None,
        experience: Optional[str] = None,
        education: Optional[str] = None,
        summary: Optional[str] = None,
        resume_text: Optional[str] = None
    ) -> str:
        """
        Combine candidate fields into a single text for embedding.
        Uses same format as CandidateProcessor for consistency.
        
        Args:
            name: Candidate name
            skills: Skills and competencies
            experience: Work experience
            education: Education background
            summary: Professional summary
            resume_text: Full resume text
        
        Returns:
            Combined text string matching the format used during embedding generation
        """
        text_parts = []
        
        # Priority 1: Skills (most critical for matching) - match CandidateProcessor format
        if skills and skills.strip():
            text_parts.append(f"Skills: {skills.strip()}")
        
        # Priority 2: Experience (very important) - match CandidateProcessor format
        if experience and experience.strip():
            text_parts.append(f"Experience: {experience.strip()}")
        
        # Priority 3: Summary (provides context) - match CandidateProcessor format
        if summary and summary.strip():
            text_parts.append(f"Professional Summary: {summary.strip()}")
        
        # Priority 4: Education (supplementary) - match CandidateProcessor format
        if education and education.strip():
            text_parts.append(f"Education: {education.strip()}")
        
        # Priority 5: Resume text (if provided, use it as primary source)
        if resume_text and resume_text.strip():
            # If we have other fields, prepend them; otherwise just use resume
            if text_parts:
                return " ".join(text_parts) + " Resume: " + resume_text.strip()
            return resume_text.strip()
        
        # Join with spaces (matching CandidateProcessor format)
        return " ".join(text_parts) if text_parts else ""
    
    def find_jobs_for_candidate_text(
        self,
        candidate_text: str,
        top_k: int = 50
    ) -> List[Dict]:
        """
        Find top matching jobs for candidate text (not in database).
        
        Args:
            candidate_text: Candidate information as text
            top_k: Number of top matches to return
        
        Returns:
            List of job matches with similarity scores
        """
        # Generate embedding for candidate text
        logger.info("=" * 80)
        logger.info("MATCHING NEW CANDIDATE (GENERATING EMBEDDING)")
        logger.info("=" * 80)
        logger.info(f"Candidate Text Length: {len(candidate_text)} characters")
        logger.info(f"Candidate Text Preview: {candidate_text[:300]}...")
        logger.info(f"Full Candidate Text: {candidate_text}")
        logger.info("Generating embedding for candidate text...")
        candidate_embedding = self.embedding_generator.generate_embedding(candidate_text)
        
        # Log vector information to console
        import numpy as np
        vector_array = np.array(candidate_embedding)
        logger.info(f"Vector Source: Generated from text (SentenceTransformer)")
        logger.info(f"Vector Dimension: {len(candidate_embedding)}")
        logger.info(f"Vector Shape: {vector_array.shape}")
        logger.info(f"Vector (first 10 values): {candidate_embedding[:10]}")
        logger.info(f"Vector Norm: {np.linalg.norm(vector_array):.4f}")
        logger.info(f"Top K: {top_k}")
        logger.info(f"Using FAISS: {self.use_faiss}")
        logger.info("=" * 80)
        
        # Search for similar jobs
        if self.use_faiss and self.faiss_manager:
            logger.info(f"Using FAISS to find top {top_k} jobs")
            results = self.faiss_manager.search(
                query_embedding=candidate_embedding,
                k=top_k,
                dataset_type='jd'
            )
            
            # Get job details from database
            jobs = []
            job_texts = []  # For re-ranking
            for job_id, similarity in results:
                job = self.repository.get_jd_embedding(job_id)
                if job:
                    job_dict = {
                        "job_id": job.job_id,
                        "title": job.title,
                        "company": job.company,
                        "location": job.location,
                        "description": job.description[:500] if job.description else None,
                        "requirements": job.requirements[:300] if job.requirements else None,
                        "similarity_score": round(similarity, 4)
                    }
                    jobs.append(job_dict)
                    # Prepare text for re-ranking
                    job_text = f"{job.title or ''} {job.description or ''} {job.requirements or ''}".strip()
                    job_texts.append(job_text)
            
            # Apply cross-encoder re-ranking for 90%+ similarity
            if self.use_reranking and self.reranker and len(jobs) > 0:
                logger.info("Applying cross-encoder re-ranking for improved similarity...")
                initial_scores = [job["similarity_score"] for job in jobs]
                reranked_indices = self.reranker.rerank_matches(
                    query_text=candidate_text,
                    candidate_texts=job_texts,
                    initial_scores=initial_scores,
                    top_k=len(jobs)
                )
                
                # Reorder jobs based on re-ranking
                reranked_jobs = [jobs[idx] for idx, _ in reranked_indices]
                
                # Also apply exact matching boost (increased for maximum similarity)
                boosted_scores = self.reranker.boost_exact_matches(
                    query_text=candidate_text,
                    candidate_texts=job_texts,
                    scores=[job["similarity_score"] for job in reranked_jobs],
                    boost_factor=1.3  # 30% boost for exact matches (increased from 15%)
                )
                
                # Update scores
                for i, job in enumerate(reranked_jobs):
                    job["similarity_score"] = round(boosted_scores[i], 4)
                
                logger.info(f"Re-ranking complete. Top similarity: {max(boosted_scores)*100:.2f}%")
                return reranked_jobs
            
            return jobs
        else:
            # Fallback to database search
            logger.info(f"Using database search to find top {top_k} jobs")
            jobs = self.repository.find_similar_jds(candidate_embedding, limit=top_k)
            
            # Calculate similarity scores
            import numpy as np
            candidate_vec = np.array(candidate_embedding)
            
            results = []
            job_texts = []  # For re-ranking
            for job in jobs:
                job_vec = np.array(job.embedding)
                similarity = np.dot(candidate_vec, job_vec) / (
                    np.linalg.norm(candidate_vec) * np.linalg.norm(job_vec)
                )
                
                job_dict = {
                    "job_id": job.job_id,
                    "title": job.title,
                    "company": job.company,
                    "location": job.location,
                    "description": job.description[:500] if job.description else None,
                    "requirements": job.requirements[:300] if job.requirements else None,
                    "similarity_score": round(float(similarity), 4)
                }
                results.append(job_dict)
                # Prepare text for re-ranking
                job_text = f"{job.title or ''} {job.description or ''} {job.requirements or ''}".strip()
                job_texts.append(job_text)
            
            # Apply cross-encoder re-ranking for 90%+ similarity
            if self.use_reranking and self.reranker and len(results) > 0:
                logger.info("Applying cross-encoder re-ranking for improved similarity...")
                initial_scores = [job["similarity_score"] for job in results]
                reranked_indices = self.reranker.rerank_matches(
                    query_text=candidate_text,
                    candidate_texts=job_texts,
                    initial_scores=initial_scores,
                    top_k=len(results)
                )
                
                # Reorder results based on re-ranking
                reranked_results = [results[idx] for idx, _ in reranked_indices]
                
                # Also apply exact matching boost
                boosted_scores = self.reranker.boost_exact_matches(
                    query_text=candidate_text,
                    candidate_texts=job_texts,
                    scores=[job["similarity_score"] for job in reranked_results],
                    boost_factor=1.15  # 15% boost for exact matches
                )
                
                # Update scores
                for i, job in enumerate(reranked_results):
                    job["similarity_score"] = round(boosted_scores[i], 4)
                
                logger.info(f"Re-ranking complete. Top similarity: {max(boosted_scores)*100:.2f}%")
                return reranked_results
            
            return results
    
    def find_jobs_for_candidate_from_file(
        self,
        candidate_file: str,
        candidate_index: int = 0,
        top_k: int = 50
    ) -> List[Dict]:
        """
        Find top matching jobs for a candidate from processed file.
        
        Args:
            candidate_file: Path to processed candidate dataset
            candidate_index: Index of candidate in file (0-based)
            top_k: Number of top matches to return
        
        Returns:
            List of job matches with similarity scores
        """
        # Load candidate data
        processor = CandidateProcessor(auto_map_columns=True)
        processor.load_from_csv(candidate_file)
        
        records = processor.get_records()
        if candidate_index >= len(records):
            logger.error(f"Candidate index {candidate_index} out of range (total: {len(records)})")
            return []
        
        candidate_record = records[candidate_index]
        candidate_text = processor.get_combined_text(processor.data.iloc[candidate_index])
        
        logger.info(f"Finding jobs for candidate at index {candidate_index}")
        logger.info(f"Candidate ID: {candidate_record.get('candidate_id', 'N/A')}")
        
        # Find matching jobs
        return self.find_jobs_for_candidate_text(candidate_text, top_k=top_k)

