"""Two-Tower Matching Service with 3-stage pipeline."""
from typing import List, Dict, Optional, Tuple
from sqlalchemy.orm import Session
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from src.database.two_tower_repository import TwoTowerRepository
from src.vector_search.two_tower_faiss_manager import TwoTowerFAISSManager
from src.embeddings.job_tower_encoder import JobTowerEncoder
from src.embeddings.candidate_tower_encoder import CandidateTowerEncoder
from config.settings import settings

logger = logging.getLogger(__name__)

# Default weights
DEFAULT_WEIGHTS = {
    'title': 0.2,
    'skills': 0.4,
    'experience': 0.4
}


class TwoTowerMatchingService:
    """Two-Tower Matching Service with 3-stage pipeline."""
    
    def __init__(
        self,
        db: Session,
        use_faiss: bool = True,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize Two-Tower matching service.
        
        Args:
            db: Database session
            use_faiss: Whether to use FAISS for fast search
            weights: Field weights (default: title 0.2, skills 0.4, experience 0.4)
        """
        self.db = db
        self.repository = TwoTowerRepository(db)
        self.use_faiss = use_faiss
        self.weights = weights or DEFAULT_WEIGHTS
        self.faiss_manager = None
        
        if use_faiss:
            try:
                dimension = settings.EMBEDDING_DIMENSION
                self.faiss_manager = TwoTowerFAISSManager(
                    dimension=dimension,
                    index_type="HNSW",
                    index_params={
                        "M": 32,
                        "ef_construction": 200,
                        "ef_search": 128
                    },
                    normalize=True
                )
                
                # Try to load existing indices
                from pathlib import Path
                base_path = Path("indices/two_tower")
                if (base_path / "job_title_index.faiss").exists():
                    self.faiss_manager.load_indices(base_path)
                    logger.info("Loaded existing Two-Tower FAISS indices")
                else:
                    logger.warning("Two-Tower FAISS indices not found. Building from database...")
                    self.faiss_manager.build_indices_from_db(db)
            except Exception as e:
                logger.warning(f"Could not initialize FAISS: {e}. Falling back to database search.")
                self.use_faiss = False
    
    def _stage1_per_field_search(
        self,
        candidate_embeddings: Dict[str, List[float]],
        top_n_per_field: int = 1000
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Stage 1: Per-field ANN search.
        
        Args:
            candidate_embeddings: Dict with 'title_embedding', 'skills_embedding', 'experience_embedding'
            top_n_per_field: Top N results per field
        
        Returns:
            Dict with keys: 'title_results', 'skills_results', 'experience_results'
        """
        if not self.use_faiss or not self.faiss_manager:
            # Fallback to database search (not implemented in this version)
            logger.warning("FAISS not available, cannot perform search")
            return {
                'title_results': [],
                'skills_results': [],
                'experience_results': []
            }
        
        results = {}
        
        # Parallel search across 3 fields
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                'title': executor.submit(
                    self.faiss_manager.search_job_by_field,
                    candidate_embeddings['title_embedding'],
                    'title',
                    top_n_per_field
                ),
                'skills': executor.submit(
                    self.faiss_manager.search_job_by_field,
                    candidate_embeddings['skills_embedding'],
                    'skills',
                    top_n_per_field
                ),
                'experience': executor.submit(
                    self.faiss_manager.search_job_by_field,
                    candidate_embeddings['experience_embedding'],
                    'requirement',  # Match candidate experience với job requirement
                    top_n_per_field
                )
            }
            
            results['title_results'] = futures['title'].result()
            results['skills_results'] = futures['skills'].result()
            results['experience_results'] = futures['experience'].result()
        
        return results
    
    def _stage2_merge_and_score(
        self,
        stage1_results: Dict[str, List[Tuple[str, float]]]
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Stage 2: Merge results and compute weighted score.
        
        Returns:
            Tuple of (job_scores, job_field_scores)
        """
        # Collect all unique job_ids
        all_job_ids = set()
        for field_results in stage1_results.values():
            all_job_ids.update([job_id for job_id, _ in field_results])
        
        # Compute weighted score for each job
        job_scores = {}
        job_field_scores = {}
        
        for job_id in all_job_ids:
            # Get scores from each field
            title_score = next(
                (score for jid, score in stage1_results['title_results'] if jid == job_id),
                0.0
            )
            skills_score = next(
                (score for jid, score in stage1_results['skills_results'] if jid == job_id),
                0.0
            )
            experience_score = next(
                (score for jid, score in stage1_results['experience_results'] if jid == job_id),
                0.0
            )
            
            # Weighted sum
            weighted_score = (
                title_score * self.weights['title'] +
                skills_score * self.weights['skills'] +
                experience_score * self.weights['experience']
            )
            
            job_scores[job_id] = weighted_score
            job_field_scores[job_id] = {
                'title': title_score,
                'skills': skills_score,
                'experience': experience_score
            }
        
        return job_scores, job_field_scores
    
    def find_jobs_for_candidate(
        self,
        candidate_id: str,
        top_k: int = 10,
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """
        Find top matching jobs for a candidate using 3-stage pipeline.
        
        Args:
            candidate_id: Candidate ID
            top_k: Number of top matches to return
            weights: Optional field weights (overrides default)
        
        Returns:
            List of job matches with similarity scores
        """
        if weights:
            self.weights = weights
        
        logger.info(f"Two-Tower matching for candidate: {candidate_id}")
        
        # Get candidate embeddings
        candidate = self.repository.get_candidate(candidate_id)
        if not candidate:
            logger.error(f"Candidate {candidate_id} not found")
            return []
        
        candidate_embeddings = {
            'title_embedding': candidate.title_embedding,
            'skills_embedding': candidate.skills_embedding,
            'experience_embedding': candidate.experience_embedding
        }
        
        # Stage 1: Per-field ANN search
        logger.debug("Stage 1: Per-field ANN search")
        stage1_results = self._stage1_per_field_search(
            candidate_embeddings,
            top_n_per_field=1000
        )
        
        # Stage 2: Merge and score
        logger.debug("Stage 2: Merge and score")
        job_scores, job_field_scores = self._stage2_merge_and_score(stage1_results)
        
        # Sort by score
        top_jobs = sorted(job_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Format results
        results = []
        for job_id, final_score in top_jobs[:top_k]:
            job = self.repository.get_job(job_id)
            if job:
                results.append({
                    'job_id': job_id,
                    'title': job.title,
                    'company': job.company,
                    'location': job.location,
                    'score': final_score,
                    'explain': job_field_scores[job_id]
                })
        
        logger.info(f"Found {len(results)} matching jobs")
        return results
    
    def find_candidates_for_job(
        self,
        job_id: str,
        top_k: int = 10,
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """
        Find top matching candidates for a job.
        
        Args:
            job_id: Job ID
            top_k: Number of top matches to return
            weights: Optional field weights
        
        Returns:
            List of candidate matches with similarity scores
        """
        if weights:
            self.weights = weights
        
        logger.info(f"Two-Tower matching for job: {job_id}")
        
        # Get job embeddings
        job = self.repository.get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return []
        
        job_embeddings = {
            'title_embedding': job.title_embedding,
            'skills_embedding': job.skills_embedding,
            'requirement_embedding': job.requirement_embedding
        }
        
        # Search candidates (similar to find_jobs_for_candidate but reverse)
        # For simplicity, we'll search each field and merge
        if not self.use_faiss or not self.faiss_manager:
            logger.warning("FAISS not available")
            return []
        
        # Search candidates by each field
        title_results = self.faiss_manager.search_candidate_by_field(
            job_embeddings['title_embedding'],
            'title',
            top_k * 10
        )
        skills_results = self.faiss_manager.search_candidate_by_field(
            job_embeddings['skills_embedding'],
            'skills',
            top_k * 10
        )
        experience_results = self.faiss_manager.search_candidate_by_field(
            job_embeddings['requirement_embedding'],
            'experience',
            top_k * 10
        )
        
        # Merge and score (similar to Stage 2)
        all_candidate_ids = set()
        for results in [title_results, skills_results, experience_results]:
            all_candidate_ids.update([cid for cid, _ in results])
        
        candidate_scores = {}
        candidate_field_scores = {}
        
        for candidate_id in all_candidate_ids:
            title_score = next((s for cid, s in title_results if cid == candidate_id), 0.0)
            skills_score = next((s for cid, s in skills_results if cid == candidate_id), 0.0)
            exp_score = next((s for cid, s in experience_results if cid == candidate_id), 0.0)
            
            weighted_score = (
                title_score * self.weights['title'] +
                skills_score * self.weights['skills'] +
                exp_score * self.weights['experience']
            )
            
            candidate_scores[candidate_id] = weighted_score
            candidate_field_scores[candidate_id] = {
                'title': title_score,
                'skills': skills_score,
                'experience': exp_score
            }
        
        # Sort and format
        top_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for candidate_id, final_score in top_candidates[:top_k]:
            candidate = self.repository.get_candidate(candidate_id)
            if candidate:
                results.append({
                    'candidate_id': candidate_id,
                    'name': candidate.name,
                    'email': candidate.email,
                    'score': final_score,
                    'explain': candidate_field_scores[candidate_id]
                })
        
        logger.info(f"Found {len(results)} matching candidates")
        return results


