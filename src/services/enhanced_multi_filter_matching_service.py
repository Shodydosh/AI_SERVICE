"""Enhanced Multi-Filter Matching Service với tất cả cải tiến."""
from typing import List, Dict, Tuple, Optional
from sqlalchemy.orm import Session
import logging
import numpy as np
import hashlib

from src.database.multi_field_repository import MultiFieldEmbeddingRepository
from src.embeddings.multi_field_generator import MultiFieldEmbeddingGenerator
from src.vector_search.multi_field_faiss_manager import MultiFieldFAISSManager
from src.services.title_matching_validator import TitleMatchingValidator
from src.services.hybrid_search_service import HybridSearchService
from src.services.reranking_service import RerankingService
from src.services.dynamic_filtering_service import DynamicFilteringService
from src.services.contextual_embeddings_service import ContextualEmbeddingsService
from src.services.negative_signals_service import NegativeSignalsService
from src.services.caching_service import CachingService

logger = logging.getLogger(__name__)


class EnhancedMultiFilterMatchingService:
    """
    Enhanced Multi-Filter Matching Service với tất cả cải tiến:
    
    1. Hybrid Search (semantic + keyword)
    2. Reranking Layer (cross-encoder)
    3. Dynamic Filtering (adjust filter sizes)
    4. Contextual Embeddings
    5. Negative Signals (deal-breakers)
    6. Caching (Redis/in-memory)
    """
    
    def __init__(
        self,
        db: Session,
        use_faiss: bool = True,
        use_hybrid_search: bool = True,
        use_reranking: bool = True,
        use_dynamic_filtering: bool = True,
        use_contextual_embeddings: bool = True,
        use_negative_signals: bool = True,
        use_caching: bool = True,
        enable_title_validation: bool = True,
        min_title_similarity: float = 0.4,
        title_boost_threshold: float = 0.6,
        title_boost_factor: float = 1.2
    ):
        """
        Initialize enhanced matching service.
        
        Args:
            db: Database session
            use_faiss: Use FAISS for fast search
            use_hybrid_search: Use hybrid search (semantic + keyword)
            use_reranking: Use cross-encoder reranking
            use_dynamic_filtering: Use dynamic filter size adjustment
            use_contextual_embeddings: Use contextual embeddings
            use_negative_signals: Apply negative signals penalties
            use_caching: Use caching for frequent queries
            enable_title_validation: Enable title matching validation
            min_title_similarity: Minimum title similarity threshold
            title_boost_threshold: Title similarity threshold for boost
            title_boost_factor: Boost factor
        """
        self.db = db
        self.repository = MultiFieldEmbeddingRepository(db)
        self.embedding_generator = None
        self.use_faiss = use_faiss
        self.faiss_manager = None
        
        # Feature flags
        self.use_hybrid_search = use_hybrid_search
        self.use_reranking = use_reranking
        self.use_dynamic_filtering = use_dynamic_filtering
        self.use_contextual_embeddings = use_contextual_embeddings
        self.use_negative_signals = use_negative_signals
        self.use_caching = use_caching
        
        # Initialize services
        if use_hybrid_search:
            self.hybrid_search = HybridSearchService(keyword_boost=0.15)
        else:
            self.hybrid_search = None
        
        if use_reranking:
            self.reranker = RerankingService(use_cross_encoder=True, top_k_rerank=100)
        else:
            self.reranker = None
        
        if use_dynamic_filtering:
            self.dynamic_filter = DynamicFilteringService()
        else:
            self.dynamic_filter = None
        
        if use_contextual_embeddings:
            # Will be initialized after embedding_generator
            self.contextual_embeddings = None
        else:
            self.contextual_embeddings = None
        
        if use_negative_signals:
            self.negative_signals = NegativeSignalsService()
        else:
            self.negative_signals = None
        
        if use_caching:
            self.cache = CachingService(use_redis=True)
        else:
            self.cache = None
        
        # Title matching validator
        if enable_title_validation:
            self.title_validator = TitleMatchingValidator(
                min_title_similarity=min_title_similarity,
                boost_threshold=title_boost_threshold,
                boost_factor=title_boost_factor
            )
        else:
            self.title_validator = None
        
        # Initialize FAISS
        if use_faiss:
            try:
                if not self.embedding_generator:
                    self.embedding_generator = MultiFieldEmbeddingGenerator()
                
                if use_contextual_embeddings:
                    self.contextual_embeddings = ContextualEmbeddingsService(self.embedding_generator)
                
                dimension = self.embedding_generator.get_embedding_dimension()
                self.faiss_manager = MultiFieldFAISSManager(
                    dimension=dimension,
                    index_type="HNSW",
                    index_params={
                        "ef_search": 128,
                        "ef_construction": 200,
                        "M": 32
                    },
                    normalize=True
                )
                
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
        
        logger.info("EnhancedMultiFilterMatchingService initialized")
        logger.info(f"  - Hybrid Search: {use_hybrid_search}")
        logger.info(f"  - Reranking: {use_reranking}")
        logger.info(f"  - Dynamic Filtering: {use_dynamic_filtering}")
        logger.info(f"  - Contextual Embeddings: {use_contextual_embeddings}")
        logger.info(f"  - Negative Signals: {use_negative_signals}")
        logger.info(f"  - Caching: {use_caching}")
    
    def find_jobs_for_candidate(
        self,
        candidate_id: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Find jobs for candidate với enhanced pipeline.
        
        Pipeline:
        1. Check cache
        2. Title matching (1000 jobs) - với hybrid search nếu enabled
        3. Skills matching (100 jobs) - với dynamic filtering nếu enabled
        4. Experience matching (10 jobs)
        5. Reranking (nếu enabled)
        6. Negative signals (nếu enabled)
        7. Cache results
        
        Args:
            candidate_id: Candidate ID
            top_k: Number of top matches
            
        Returns:
            List of job matches
        """
        # Check cache
        if self.cache:
            cached = self.cache.get_cached_recommendations(candidate_id)
            if cached:
                logger.info(f"Cache hit for candidate {candidate_id}")
                return cached
        
        # Get candidate data
        candidate = self.repository.get_candidate_multi_embedding(candidate_id)
        if not candidate:
            logger.error(f"Candidate {candidate_id} not found")
            return []
        
        # Get embeddings
        candidate_title_emb = candidate.title_embedding
        candidate_skills_emb = candidate.skills_embedding
        candidate_experience_emb = candidate.experience_embedding
        
        # Validate embeddings
        def validate_embedding(emb, name):
            if not emb:
                return False
            emb_array = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(emb_array)
            return norm > 1e-10
        
        title_valid = validate_embedding(candidate_title_emb, "title")
        skills_valid = validate_embedding(candidate_skills_emb, "skills")
        exp_valid = validate_embedding(candidate_experience_emb, "experience")
        
        if not title_valid and not exp_valid:
            logger.error("Both title and experience embeddings are invalid")
            return []
        
        # Dynamic filter sizes (default)
        step1_size = 1000
        step2_size = 100
        step3_size = top_k
        
        # STEP 1: Title matching (1000 jobs)
        if title_valid:
            logger.info(f"Step 1: Finding {step1_size} jobs by title similarity...")
            
            # Use contextual embeddings if enabled
            if self.use_contextual_embeddings and self.contextual_embeddings:
                query_text = self.contextual_embeddings.create_candidate_contextual_text(
                    desired_job=candidate.title,
                    skills=candidate.skills,
                    experience=candidate.experience
                )
            else:
                query_text = candidate.title or ""
            
            step1_results = self._filter_by_title(
                candidate_title_emb,
                job_ids=None,
                top_k=step1_size
            )
            
            # Apply hybrid search if enabled
            if self.use_hybrid_search and self.hybrid_search and query_text:
                # Get job data for hybrid search
                job_embeddings = {}
                job_texts = {}
                job_dicts = []
                
                for job_id, score in step1_results[:step1_size]:
                    job = self.repository.get_job_multi_embedding(job_id)
                    if job:
                        job_embeddings[job_id] = job.title_embedding or []
                        if self.use_contextual_embeddings and self.contextual_embeddings:
                            job_text = self.contextual_embeddings.create_jd_contextual_text(
                                title=job.title,
                                skills=job.skills,
                                requirements=job.requirement
                            )
                        else:
                            job_text = job.title or ""
                        job_texts[job_id] = job_text
                        job_dicts.append({'id': job_id})
                
                # Hybrid search
                hybrid_results = self.hybrid_search.search_with_keywords(
                    query_text=query_text,
                    query_embedding=candidate_title_emb,
                    candidates=job_dicts,
                    candidate_embeddings=job_embeddings,
                    candidate_texts=job_texts,
                    top_k=step1_size
                )
                
                # Update step1_results với hybrid scores
                hybrid_dict = {job_id: score for job_id, score, _ in hybrid_results}
                step1_results = [
                    (job_id, hybrid_dict.get(job_id, sim))
                    for job_id, sim in step1_results
                ]
                step1_results.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"✓ Step 1: Found {len(step1_results)} jobs")
        else:
            # Fallback: use all jobs
            all_jobs = self.repository.get_all_job_multi_embeddings()
            step1_results = [(job.job_id, 0.5) for job in all_jobs[:step1_size]]
            logger.info(f"✓ Step 1: Using {len(step1_results)} jobs (no title filtering)")
        
        # Dynamic filtering adjustment
        if self.use_dynamic_filtering and self.dynamic_filter:
            # Assess data quality
            job_embeddings = {}
            job_texts = {}
            for job_id, _ in step1_results:
                job = self.repository.get_job_multi_embedding(job_id)
                if job:
                    job_embeddings[job_id] = job.title_embedding or []
                    job_texts[job_id] = job.title or ""
            
            quality_scores = self.dynamic_filter.assess_data_quality(job_embeddings, job_texts)
            diversity = self.dynamic_filter.calculate_diversity(
                [{'id': jid} for jid, _ in step1_results],
                job_embeddings
            )
            
            # Adjust filter sizes
            step1_size, step2_size, step3_size = self.dynamic_filter.adjust_filter_sizes(
                step1_size, step2_size, step3_size,
                quality_scores, diversity
            )
            logger.info(f"Dynamic filtering: Adjusted sizes to ({step1_size}, {step2_size}, {step3_size})")
        
        # STEP 2: Skills matching (100 jobs)
        if skills_valid:
            logger.info(f"Step 2: Filtering to {step2_size} jobs by skills similarity...")
            step2_results = self._filter_by_skills(
                candidate_skills_emb,
                job_ids=[job_id for job_id, _ in step1_results[:step1_size]],
                top_k=step2_size
            )
            logger.info(f"✓ Step 2: Filtered to {len(step2_results)} jobs")
        else:
            step2_results = [(job_id, 0.5) for job_id, _ in step1_results[:step2_size]]
            logger.info(f"✓ Step 2: Using {len(step2_results)} jobs (no skills filtering)")
        
        # STEP 3: Experience matching (top_k jobs)
        if exp_valid:
            logger.info(f"Step 3: Filtering to top {step3_size} jobs by experience similarity...")
            step3_results = self._filter_by_experience_requirement(
                candidate_experience_emb,
                job_ids=[job_id for job_id, _ in step2_results],
                top_k=step3_size
            )
            
            if len(step3_results) == 0:
                logger.warning("No jobs found in step 3 - jobs may have zero requirement embeddings")
                logger.info("Step 3: FALLBACK - using jobs from step 2")
                step3_results = [(job_id, 0.5) for job_id, _ in step2_results[:step3_size]]  # Neutral score
                logger.info(f"✓ Step 3: Using {len(step3_results)} jobs (fallback - no experience filtering)")
            else:
                logger.info(f"✓ Step 3: Filtered to {len(step3_results)} final jobs")
        else:
            step3_results = [(job_id, 0.5) for job_id, _ in step2_results[:step3_size]]
            logger.info(f"✓ Step 3: Using {len(step3_results)} jobs (no experience filtering)")
        
        # Build final results
        final_results = []
        for job_id, exp_sim in step3_results:
            job = self.repository.get_job_multi_embedding(job_id)
            if not job:
                continue
            
            # Get scores from all steps
            title_sim = next((sim for jid, sim in step1_results if jid == job_id), 0.0) if title_valid else 0.5
            skills_sim = next((sim for jid, sim in step2_results if jid == job_id), 0.0) if skills_valid else 0.5
            exp_sim = exp_sim if exp_valid else 0.5
            
            # Combined score
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
                    "title": round(title_sim, 4) if title_valid else None,
                    "skills": round(skills_sim, 4) if skills_valid else None,
                    "experience": round(exp_sim, 4) if exp_valid else None
                }
            })
        
        # Sort by combined score
        final_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # Apply reranking if enabled
        if self.use_reranking and self.reranker and len(final_results) > 0:
            logger.info("Applying reranking...")
            if self.use_contextual_embeddings and self.contextual_embeddings:
                query_text = self.contextual_embeddings.create_candidate_contextual_text(
                    desired_job=candidate.title,
                    skills=candidate.skills,
                    experience=candidate.experience
                )
            else:
                query_text = candidate.title or ""
            
            final_results = self.reranker.rerank_pipeline(
                query_text=query_text,
                faiss_results=final_results,
                top_k=top_k
            )
            logger.info(f"✓ Reranked to {len(final_results)} jobs")
        
        # Apply negative signals if enabled
        if self.use_negative_signals and self.negative_signals:
            logger.info("Applying negative signals...")
            # Get job data
            job_data_dict = {}
            for result in final_results:
                job_id = result["job_id"]
                job = self.repository.get_job_multi_embedding(job_id)
                if job:
                    job_data_dict[job_id] = {
                        'salary_min': getattr(job, 'salary_min', None),
                        'salary_max': getattr(job, 'salary_max', None),
                        'location': job.location,
                        'industry': getattr(job, 'industry', None),
                        'requirements': job.requirement
                    }
            
            candidate_data = {
                'expected_salary': getattr(candidate, 'expected_salary', None),
                'location': getattr(candidate, 'location', None),
                'industry': getattr(candidate, 'industry', None),
                'experience_years': getattr(candidate, 'experience_years', None)
            }
            
            final_results = self.negative_signals.apply_negative_signals(
                final_results,
                candidate_data,
                job_data_dict
            )
            logger.info(f"✓ Applied negative signals")
        
        # Apply title validation if enabled
        if self.title_validator and title_valid:
            logger.info("Applying title matching validation...")
            job_title_embeddings = {}
            for result in final_results:
                job_id = result["job_id"]
                job = self.repository.get_job_multi_embedding(job_id)
                if job and job.title_embedding:
                    job_title_embeddings[job_id] = job.title_embedding
            
            final_results = self.title_validator.validate_and_boost(
                final_results,
                candidate_title_emb=candidate_title_emb,
                job_title_embeddings=job_title_embeddings if job_title_embeddings else None
            )
        
        # Cache results
        if self.cache:
            self.cache.cache_candidate_recommendations(candidate_id, final_results)
        
        logger.info(f"Final Results: {len(final_results)} jobs")
        return final_results
    
    def _filter_by_title(
        self,
        candidate_title_emb: List[float],
        job_ids: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Filter by title similarity."""
        if self.use_faiss and self.faiss_manager:
            if job_ids is None:
                results = self.faiss_manager.search(
                    query_embedding=candidate_title_emb,
                    field_type='title',
                    k=top_k
                )
            else:
                results = self.faiss_manager.search_filtered(
                    query_embedding=candidate_title_emb,
                    field_type='title',
                    candidate_ids=job_ids,
                    k=top_k
                )
            return results
        else:
            if job_ids is None:
                return self.repository.find_similar_jobs_by_field(
                    query_embedding=candidate_title_emb,
                    field_type='title',
                    limit=top_k
                )
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
        """Filter by skills similarity."""
        if self.use_faiss and self.faiss_manager:
            results = self.faiss_manager.search_filtered(
                query_embedding=candidate_skills_emb,
                field_type='skills',
                candidate_ids=job_ids,
                k=top_k
            )
            return results
        else:
            return self.repository.find_similar_jobs_by_field_filtered(
                query_embedding=candidate_skills_emb,
                field_type='skills',
                job_ids=job_ids,
                limit=top_k
            )
    
    def _filter_by_experience_requirement(
        self,
        candidate_experience_emb: List[float],
        job_ids: List[str],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Filter by experience/requirement similarity."""
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

