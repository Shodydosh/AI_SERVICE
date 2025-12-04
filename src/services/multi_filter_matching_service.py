"""Multi-filter matching service with 3-step pipeline."""
from typing import List, Dict, Tuple, Optional
from sqlalchemy.orm import Session
import logging
import numpy as np
from src.database.multi_field_repository import MultiFieldEmbeddingRepository
from src.embeddings.multi_field_generator import MultiFieldEmbeddingGenerator
from src.vector_search.multi_field_faiss_manager import MultiFieldFAISSManager
from src.services.title_matching_validator import TitleMatchingValidator

logger = logging.getLogger(__name__)


class MultiFilterMatchingService:
    """
    Multi-filter matching service with 3-step pipeline:
    1. Filter 1000 jobs by title similarity (Weight: 50-60%)
    2. Filter 100 jobs by skills similarity (Weight: 30-40%)
    3. Filter top 10 jobs by experience/requirement similarity (Weight: 10-20%)
    """
    
    def __init__(
        self,
        db: Session,
        use_faiss: bool = True,
        enable_title_validation: bool = True,
        min_title_similarity: float = 0.4,
        title_boost_threshold: float = 0.6,
        title_boost_factor: float = 1.2
    ):
        """
        Initialize multi-filter matching service.
        
        Args:
            db: Database session
            use_faiss: Whether to use FAISS for fast search
            enable_title_validation: Enable title matching validation and boosting
            min_title_similarity: Minimum title similarity threshold (0-1)
            title_boost_threshold: Title similarity threshold để boost score (0-1)
            title_boost_factor: Factor để boost score (ví dụ: 1.2 = tăng 20%)
        """
        self.db = db
        self.repository = MultiFieldEmbeddingRepository(db)
        # Only initialize embedding generator if needed (for find_jobs_for_candidate_text)
        # For find_jobs_for_candidate, we use embeddings from database
        self.embedding_generator = None  # Lazy initialization
        self.use_faiss = use_faiss
        self.faiss_manager = None
        
        # Initialize title matching validator
        self.enable_title_validation = enable_title_validation
        if enable_title_validation:
            self.title_validator = TitleMatchingValidator(
                min_title_similarity=min_title_similarity,
                boost_threshold=title_boost_threshold,
                boost_factor=title_boost_factor
            )
            logger.info(f"Title matching validation enabled (min: {min_title_similarity}, boost: {title_boost_threshold} with factor {title_boost_factor})")
        else:
            self.title_validator = None
        
        if use_faiss:
            try:
                # Initialize embedding generator only if needed for FAISS dimension
                if not self.embedding_generator:
                    self.embedding_generator = MultiFieldEmbeddingGenerator()
                dimension = self.embedding_generator.get_embedding_dimension()
                self.faiss_manager = MultiFieldFAISSManager(
                    dimension=dimension,
                    index_type="HNSW",
                    index_params={
                        "ef_search": 128,  # Higher for better precision at k=1000
                        "ef_construction": 200,
                        "M": 32
                    },
                    normalize=True
                )
                
                # Try to load existing indices
                from pathlib import Path
                base_path = Path("indices/multi_field")
                base_path.mkdir(parents=True, exist_ok=True)
                
                if (base_path / "jd_title_index.faiss").exists():
                    self.faiss_manager.load_indices(base_path)
                    logger.info("Loaded existing multi-field FAISS indices")
                else:
                    logger.warning("Multi-field FAISS indices not found. Building from database...")
                    self.faiss_manager.build_indices_from_db(db)
            except Exception as e:
                logger.warning(f"Could not initialize FAISS: {e}. Falling back to database search.")
                self.use_faiss = False
    
    def find_jobs_for_candidate(
        self,
        candidate_id: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Find top matching jobs for a candidate using 3-step filter pipeline.
        
        Pipeline:
        1. Find 1000 jobs by title similarity (Weight: 50-60%)
        2. Filter to 100 jobs by skills similarity (Weight: 30-40%)
        3. Filter to top 10 jobs by experience/requirement similarity (Weight: 10-20%)
        
        Args:
            candidate_id: Candidate ID from database
            top_k: Number of top matches to return (default: 10)
        
        Returns:
            List of job matches with similarity scores
        """
        logger.info("=" * 80)
        logger.info(f"MULTI-FILTER MATCHING for Candidate: {candidate_id}")
        logger.info("=" * 80)
        
        # Get candidate embeddings from database (DO NOT RE-EMBED)
        candidate = self.repository.get_candidate_multi_embedding(candidate_id)
        if not candidate:
            logger.error(f"Candidate {candidate_id} not found in database")
            return []
        
        candidate_experience_emb = candidate.experience_embedding
        candidate_skills_emb = candidate.skills_embedding
        candidate_title_emb = candidate.title_embedding
        
        # Validate embeddings (check if they're zero vectors or None)
        import numpy as np
        
        def validate_embedding(emb, name):
            if not emb:
                logger.warning(f"Candidate {name} embedding is None or empty - will skip filtering by this field")
                return False
            emb_array = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(emb_array)
            if norm == 0 or norm < 1e-10:
                logger.warning(f"Candidate {name} embedding is zero vector (norm: {norm:.6f}) - will skip filtering by this field")
                return False
            logger.debug(f"Candidate {name} embedding valid (dim: {len(emb)}, norm: {norm:.6f})")
            return True
        
        # Check which embeddings are valid
        exp_valid = validate_embedding(candidate_experience_emb, "experience")
        skills_valid = validate_embedding(candidate_skills_emb, "skills")
        title_valid = validate_embedding(candidate_title_emb, "title")
        
        # At least title or experience must be valid (skills can be optional)
        if not title_valid and not exp_valid:
            logger.error("Both title and experience embeddings are invalid - cannot proceed")
            return []
        
        if not title_valid:
            logger.warning("Title embedding invalid - will skip Step 1 filtering")
        if not skills_valid:
            logger.warning("Skills embedding invalid - will skip Step 2 filtering")
        if not exp_valid:
            logger.warning("Experience embedding invalid - will skip Step 3 filtering")
        
        logger.info(f"✓ Using embeddings from database (no re-embedding)")
        
        # STEP 1: Find 1000 jobs by title similarity (or skip if invalid/no results)
        if title_valid:
            logger.info(f"Step 1: Finding 1000 jobs by title similarity...")
            step1_results = self._filter_by_title(
                candidate_title_emb,
                job_ids=None,  # Search all jobs
                top_k=1000
            )
            
            if len(step1_results) == 0:
                logger.warning("No jobs found in step 1 - likely all job title embeddings are zero vectors")
                # Fallback: Use all jobs if Step 1 fails
                logger.info("Step 1: FALLBACK - using all jobs (jobs may have zero title embeddings)")
                all_jobs = self.repository.get_all_job_multi_embeddings()
                step1_results = [(job.job_id, 0.5) for job in all_jobs[:1000]]  # Neutral similarity score
                logger.info(f"✓ Step 1: Using {len(step1_results)} jobs (fallback - no title filtering)")
            else:
                logger.info(f"✓ Step 1: Found {len(step1_results)} jobs")
        else:
            # Skip Step 1 - use all jobs
            logger.info(f"Step 1: SKIPPED (invalid title embedding) - using all jobs")
            all_jobs = self.repository.get_all_job_multi_embeddings()
            step1_results = [(job.job_id, 0.5) for job in all_jobs[:1000]]  # Neutral similarity score
            logger.info(f"✓ Step 1: Using {len(step1_results)} jobs (no filtering)")
        
        # STEP 2: Filter to 100 jobs by skills similarity (or skip if invalid/no results)
        if skills_valid:
            logger.info(f"Step 2: Filtering to 100 jobs by skills similarity...")
            step2_results = self._filter_by_skills(
                candidate_skills_emb,
                job_ids=[job_id for job_id, _ in step1_results],
                top_k=100
            )
            
            if len(step2_results) == 0:
                logger.warning("No jobs found in step 2 - jobs may have zero skills embeddings")
                logger.info("Step 2: FALLBACK - using jobs from step 1")
                step2_results = [(job_id, 0.5) for job_id, _ in step1_results[:100]]  # Neutral score
                logger.info(f"✓ Step 2: Using {len(step2_results)} jobs (fallback - no skills filtering)")
            else:
                logger.info(f"✓ Step 2: Filtered to {len(step2_results)} jobs")
        else:
            # Skip Step 2 - use jobs from step 1
            logger.info(f"Step 2: SKIPPED (invalid skills embedding) - using jobs from step 1")
            step2_results = [(job_id, 0.5) for job_id, _ in step1_results[:100]]  # Neutral score
            logger.info(f"✓ Step 2: Using {len(step2_results)} jobs (no filtering)")
        
        # STEP 3: Filter to top 10 by experience/requirement similarity (or skip if invalid/no results)
        if exp_valid:
            logger.info(f"Step 3: Filtering to top {top_k} jobs by experience/requirement similarity...")
            step3_results = self._filter_by_experience_requirement(
                candidate_experience_emb,
                job_ids=[job_id for job_id, _ in step2_results],
                top_k=top_k
            )
            
            if len(step3_results) == 0:
                logger.warning("No jobs found in step 3 - jobs may have zero requirement embeddings")
                logger.info("Step 3: FALLBACK - using jobs from step 2")
                step3_results = [(job_id, 0.5) for job_id, _ in step2_results[:top_k]]  # Neutral score
                logger.info(f"✓ Step 3: Using {len(step3_results)} jobs (fallback - no experience filtering)")
            else:
                logger.info(f"✓ Step 3: Filtered to {len(step3_results)} final jobs")
        else:
            # Skip Step 3 - use jobs from step 2
            logger.info(f"Step 3: SKIPPED (invalid experience embedding) - using jobs from step 2")
            step3_results = [(job_id, 0.5) for job_id, _ in step2_results[:top_k]]  # Neutral score
            logger.info(f"✓ Step 3: Using {len(step3_results)} jobs (no filtering)")
        
        # Get job details and combine scores
        final_results = []
        for job_id, exp_similarity in step3_results:
            job = self.repository.get_job_multi_embedding(job_id)
            if not job:
                continue
            
            # Get similarity scores from previous steps (use 0.0 if field was skipped)
            title_sim_score = next((sim for jid, sim in step1_results if jid == job_id), 0.0) if title_valid else 0.5  # Neutral score if skipped
            skills_sim = next((sim for jid, sim in step2_results if jid == job_id), 0.0) if skills_valid else 0.5  # Neutral score if skipped
            exp_sim = exp_similarity if exp_valid else 0.5  # Neutral score if skipped
            
            # Combined score (weighted average)
            # New weights: Title (50%), Skills (35%), Experience (15%)
            # Adjust weights based on which fields are valid
            total_weight = (0.5 if title_valid else 0) + (0.35 if skills_valid else 0) + (0.15 if exp_valid else 0)
            if total_weight == 0:
                total_weight = 1.0  # Fallback
            
            combined_score = (
                (title_sim_score * 0.5 if title_valid else 0) +
                (skills_sim * 0.35 if skills_valid else 0) +
                (exp_sim * 0.15 if exp_valid else 0)
            ) / total_weight
            
            final_results.append({
                "job_id": job.job_id,
                "title": job.title,
                "company": job.company,
                "location": job.location,
                "similarity_score": round(combined_score, 4),
                "field_similarities": {
                    "title": round(title_sim_score, 4) if title_valid else None,
                    "skills": round(skills_sim, 4) if skills_valid else None,
                    "experience": round(exp_sim, 4) if exp_valid else None
                }
            })
        
        # Sort by combined score
        final_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # Apply title matching validation and boosting if enabled
        if self.enable_title_validation and self.title_validator and title_valid:
            logger.info("Applying title matching validation and boosting...")
            
            # Get job title embeddings for validation
            job_title_embeddings = {}
            for result in final_results:
                job_id = result["job_id"]
                job = self.repository.get_job_multi_embedding(job_id)
                if job and job.title_embedding:
                    job_title_embeddings[job_id] = job.title_embedding
            
            # Validate and boost
            final_results = self.title_validator.validate_and_boost(
                final_results,
                candidate_title_emb=candidate_title_emb,
                job_title_embeddings=job_title_embeddings if job_title_embeddings else None
            )
            
            # Log statistics
            stats = self.title_validator.get_title_matching_stats(final_results)
            logger.info(f"Title matching stats: {stats['with_title_similarity']}/{stats['total_matches']} with title similarity, "
                       f"avg: {stats['avg_title_similarity']:.4f}, boosted: {stats['boosted_count']}")
        
        logger.info("=" * 80)
        logger.info(f"Final Results: {len(final_results)} jobs")
        if final_results:
            logger.info(f"Top similarity: {final_results[0]['similarity_score']:.4f}")
        logger.info("=" * 80)
        
        return final_results
    
    def find_jobs_for_candidate_text(
        self,
        title: Optional[str] = None,
        skills: Optional[str] = None,
        experience: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Find top matching jobs for candidate text using 3-step filter pipeline.
        
        Args:
            title: Candidate desired job title
            skills: Candidate skills
            experience: Candidate experience
            top_k: Number of top matches to return
        
        Returns:
            List of job matches with similarity scores
        """
        logger.info("=" * 80)
        logger.info("MULTI-FILTER MATCHING for New Candidate (from text)")
        logger.info("=" * 80)
        
        # Initialize embedding generator only if needed (for text input)
        if not self.embedding_generator:
            self.embedding_generator = MultiFieldEmbeddingGenerator()
        
        # Generate embeddings for candidate (only for text input, not for database candidates)
        candidate_embeddings = self.embedding_generator.generate_candidate_embeddings(
            title=title,
            skills=skills,
            experience=experience
        )
        
        candidate_experience_emb = candidate_embeddings['experience_embedding']
        candidate_skills_emb = candidate_embeddings['skills_embedding']
        candidate_title_emb = candidate_embeddings['title_embedding']
        
        # Use same pipeline (Title → Skills → Experience)
        logger.info(f"Step 1: Finding 1000 jobs by title similarity...")
        step1_results = self._filter_by_title(
            candidate_title_emb,
            job_ids=None,  # Search all jobs
            top_k=1000
        )
        
        if len(step1_results) == 0:
            return []
        
        logger.info(f"✓ Step 1: Found {len(step1_results)} jobs")
        logger.info(f"Step 2: Filtering to 100 jobs by skills similarity...")
        
        step2_results = self._filter_by_skills(
            candidate_skills_emb,
            job_ids=[job_id for job_id, _ in step1_results],
            top_k=100
        )
        
        if len(step2_results) == 0:
            return []
        
        logger.info(f"✓ Step 2: Filtered to {len(step2_results)} jobs")
        logger.info(f"Step 3: Filtering to top {top_k} jobs by experience/requirement similarity...")
        
        step3_results = self._filter_by_experience_requirement(
            candidate_experience_emb,
            job_ids=[job_id for job_id, _ in step2_results],
            top_k=top_k
        )
        
        logger.info(f"✓ Step 3: Filtered to {len(step3_results)} final jobs")
        
        # Get job details
        final_results = []
        for job_id, exp_sim in step3_results:
            job = self.repository.get_job_multi_embedding(job_id)
            if not job:
                continue
            
            title_sim = next((sim for jid, sim in step1_results if jid == job_id), 0.0)
            skills_sim = next((sim for jid, sim in step2_results if jid == job_id), 0.0)
            
            # New weights: Title (50%), Skills (35%), Experience (15%)
            combined_score = (
                title_sim * 0.5 +
                skills_sim * 0.35 +
                exp_sim * 0.15
            )
            
            final_results.append({
                "job_id": job.job_id,
                "title": job.title,
                "company": job.company,
                "location": job.location,
                "similarity_score": round(combined_score, 4),
                "field_similarities": {
                    "title": round(title_sim, 4),
                    "skills": round(skills_sim, 4),
                    "experience": round(exp_sim, 4)
                }
            })
        
        final_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # Apply title matching validation and boosting if enabled
        if self.enable_title_validation and self.title_validator and candidate_title_emb:
            logger.info("Applying title matching validation and boosting...")
            
            # Get job title embeddings for validation
            job_title_embeddings = {}
            for result in final_results:
                job_id = result["job_id"]
                job = self.repository.get_job_multi_embedding(job_id)
                if job and job.title_embedding:
                    job_title_embeddings[job_id] = job.title_embedding
            
            # Validate and boost
            final_results = self.title_validator.validate_and_boost(
                final_results,
                candidate_title_emb=candidate_title_emb,
                job_title_embeddings=job_title_embeddings if job_title_embeddings else None
            )
            
            # Log statistics
            stats = self.title_validator.get_title_matching_stats(final_results)
            logger.info(f"Title matching stats: {stats['with_title_similarity']}/{stats['total_matches']} with title similarity, "
                       f"avg: {stats['avg_title_similarity']:.4f}, boosted: {stats['boosted_count']}")
        
        return final_results
    
    def _filter_by_title(
        self,
        candidate_title_emb: List[float],
        job_ids: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Step 1: Filter jobs by title similarity."""
        if job_ids is None:
            # Search all jobs (Step 1)
            if self.use_faiss and self.faiss_manager:
                results = self.faiss_manager.search(
                    query_embedding=candidate_title_emb,
                    field_type='title',
                    k=top_k
                )
                return results
            else:
                return self.repository.find_similar_jobs_by_field(
                    query_embedding=candidate_title_emb,
                    field_type='title',
                    limit=top_k
                )
        else:
            # Search in filtered job IDs (Step 3 in old order, but not used in new order)
            if self.use_faiss and self.faiss_manager:
                results = self.faiss_manager.search_filtered(
                    query_embedding=candidate_title_emb,
                    field_type='title',
                    candidate_ids=job_ids,
                    k=top_k
                )
                return results
            else:
                return self.repository.find_similar_jobs_by_field_filtered(
                    query_embedding=candidate_title_emb,
                    field_type='title',
                    job_ids=job_ids,
                    limit=top_k
                )
    
    def _filter_by_skills(
        self,
        candidate_skills_emb: List[float],
        job_ids: List[str],
        top_k: int = 100
    ) -> List[Tuple[str, float]]:
        """Step 2: Filter jobs by skills similarity from candidate list."""
        if self.use_faiss and self.faiss_manager:
            # Use FAISS to search in filtered job IDs
            results = self.faiss_manager.search_filtered(
                query_embedding=candidate_skills_emb,
                field_type='skills',
                candidate_ids=job_ids,
                k=top_k
            )
            return results
        else:
            # Use database search with filter
            return self.repository.find_similar_jobs_by_field_filtered(
                query_embedding=candidate_skills_emb,
                field_type='skills',
                job_ids=job_ids,
                limit=top_k
            )
    
    def _filter_by_experience_requirement(
        self,
        candidate_experience_emb: List[float],
        job_ids: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Step 3: Filter jobs by experience/requirement similarity from candidate list."""
        if job_ids is None:
            # Search all jobs (should not happen in new order, but keep for compatibility)
            if self.use_faiss and self.faiss_manager:
                results = self.faiss_manager.search(
                    query_embedding=candidate_experience_emb,
                    field_type='requirement',
                    k=top_k
                )
                return results
            else:
                return self.repository.find_similar_jobs_by_field(
                    query_embedding=candidate_experience_emb,
                    field_type='requirement',
                    limit=top_k
                )
        else:
            # Search in filtered job IDs (Step 3)
            if self.use_faiss and self.faiss_manager:
                results = self.faiss_manager.search_filtered(
                    query_embedding=candidate_experience_emb,
                    field_type='requirement',
                    candidate_ids=job_ids,
                    k=top_k
                )
                return results
            else:
                return self.repository.find_similar_jobs_by_field_filtered(
                    query_embedding=candidate_experience_emb,
                    field_type='requirement',
                    job_ids=job_ids,
                    limit=top_k
                )
