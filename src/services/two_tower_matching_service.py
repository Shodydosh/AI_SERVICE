"""Two-Tower Matching Service with 3 separate embeddings per candidate/job."""
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
import logging
import numpy as np
from src.embeddings.candidate_tower_encoder import CandidateTowerEncoder
from src.embeddings.job_tower_encoder import JobTowerEncoder
from src.database.two_tower_repository import TwoTowerRepository

logger = logging.getLogger(__name__)


class TwoTowerMatchingService:
    """Two-Tower Matching Service with 3 embeddings per candidate/job."""
    
    def __init__(
        self,
        db: Session,
        model_name: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize Two-Tower matching service.
        
        Args:
            db: Database session
            model_name: Name of embedding model to use (default from settings)
            weights: Weights for combining field similarities. 
                     Default: {'title': 0.4, 'skills': 0.4, 'experience': 0.2}
        """
        self.db = db
        self.repository = TwoTowerRepository(db)
        
        # Initialize encoders
        logger.info("Initializing Candidate and Job Tower encoders...")
        self.candidate_encoder = CandidateTowerEncoder(model_name=model_name)
        self.job_encoder = JobTowerEncoder(model_name=model_name)
        logger.info("✓ Encoders initialized")
        
        # Set weights for combining similarities
        if weights is None:
            weights = {
                'title': 0.4,
                'skills': 0.4,
                'experience': 0.2  # experience vs requirement
            }
        self.weights = weights
        logger.info(f"Using weights: {weights}")
    
    def _encode_candidate_fields(
        self,
        title: Optional[str] = None,
        skills: Optional[str] = None,
        experience: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Encode candidate into 3 separate embeddings.
        
        Returns:
            Dict with keys: 'title', 'skills', 'experience' -> numpy arrays
        """
        embeddings = self.candidate_encoder.encode_candidate(
            title=title,
            skills=skills,
            experience=experience
        )
        
        return {
            'title': np.array(embeddings['title_embedding'], dtype=np.float32),
            'skills': np.array(embeddings['skills_embedding'], dtype=np.float32),
            'experience': np.array(embeddings['experience_embedding'], dtype=np.float32)
        }
    
    def _encode_job_fields(
        self,
        title: str,
        skills: Optional[str] = None,
        requirement: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Encode job into 3 separate embeddings.
        
        Returns:
            Dict with keys: 'title', 'skills', 'requirement' -> numpy arrays
        """
        embeddings = self.job_encoder.encode_job(
            title=title,
            skills=skills,
            requirements=requirement
        )
        
        return {
            'title': np.array(embeddings['title_embedding'], dtype=np.float32),
            'skills': np.array(embeddings['skills_embedding'], dtype=np.float32),
            'requirement': np.array(embeddings['requirement_embedding'], dtype=np.float32)
        }
    
    def _compute_field_similarities(
        self,
        candidate_embs: Dict[str, np.ndarray],
        job_embs: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute cosine similarity for each field.
        
        Returns:
            Dict with field similarities: {'title': 0.85, 'skills': 0.78, 'experience': 0.72}
        """
        # Title similarity
        title_sim = float(np.dot(candidate_embs['title'], job_embs['title']))
        
        # Skills similarity
        skills_sim = float(np.dot(candidate_embs['skills'], job_embs['skills']))
        
        # Experience (candidate) vs Requirement (job) similarity
        exp_sim = float(np.dot(candidate_embs['experience'], job_embs['requirement']))
        
        return {
            'title': title_sim,
            'skills': skills_sim,
            'experience': exp_sim
        }
    
    def _compute_combined_score(
        self,
        field_similarities: Dict[str, float]
    ) -> float:
        """
        Compute weighted combined score from field similarities.
        
        Args:
            field_similarities: Dict with 'title', 'skills', 'experience' similarities
        
        Returns:
            Combined weighted score
        """
        score = (
            self.weights['title'] * field_similarities['title'] +
            self.weights['skills'] * field_similarities['skills'] +
            self.weights['experience'] * field_similarities['experience']
        )
        return float(score)
    
    def find_jobs_for_candidate(
        self,
        candidate_id: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Find top matching jobs for a candidate using 3-field two-tower similarity.
        
        Args:
            candidate_id: Candidate ID
            top_k: Number of top matches to return
        
        Returns:
            List of job matches with similarity scores and field breakdowns
        """
        logger.info(f"Two-Tower matching (3 embeddings) for candidate: {candidate_id}")
        
        # Get candidate from database
        candidate = self.repository.get_candidate(candidate_id)
        if not candidate:
            logger.error(f"Candidate {candidate_id} not found")
            return []
        
        # Encode candidate into 3 embeddings
        logger.info("Encoding candidate into 3 embeddings...")
        candidate_embs = self._encode_candidate_fields(
            title=candidate.title,
            skills=candidate.skills,
            experience=candidate.experience
        )
        
        # Get all jobs
        all_jobs = self.repository.get_all_jobs()
        if not all_jobs:
            logger.warning("No jobs found in database")
            return []
        
        logger.info(f"Computing similarity with {len(all_jobs)} jobs using 3 embeddings...")
        
        # Compute similarities for all jobs
        all_scores = []
        all_field_similarities = []
        
        for job in all_jobs:
            # Encode job into 3 embeddings
            job_embs = self._encode_job_fields(
                title=job.title or "",
                skills=job.skills,
                requirement=job.requirement
            )
            
            # Compute field similarities
            field_sims = self._compute_field_similarities(candidate_embs, job_embs)
            
            # Compute combined score
            combined_score = self._compute_combined_score(field_sims)
            
            all_scores.append(combined_score)
            all_field_similarities.append({
                'job': job,
                'field_similarities': field_sims
            })
        
        # Get top-k
        scores_array = np.array(all_scores)
        top_indices = np.argsort(scores_array)[::-1][:top_k]
        
        # Format results
        results = []
        for idx in top_indices:
            job = all_field_similarities[idx]['job']
            score = all_scores[idx]
            field_sims = all_field_similarities[idx]['field_similarities']
            
            results.append({
                'job_id': job.job_id,
                'title': job.title,
                'company': job.company,
                'location': job.location,
                'score': score,
                'field_scores': {
                    'title': field_sims['title'],
                    'skills': field_sims['skills'],
                    'experience': field_sims['experience']
                }
            })
        
        logger.info(f"Found {len(results)} matching jobs")
        return results
    
    def find_candidates_for_job(
        self,
        job_id: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Find top matching candidates for a job using 3-field two-tower similarity.
        
        Args:
            job_id: Job ID
            top_k: Number of top matches to return
        
        Returns:
            List of candidate matches with similarity scores and field breakdowns
        """
        logger.info(f"Two-Tower matching (3 embeddings) for job: {job_id}")
        
        # Get job from database
        job = self.repository.get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return []
        
        # Encode job into 3 embeddings
        logger.info("Encoding job into 3 embeddings...")
        job_embs = self._encode_job_fields(
            title=job.title or "",
            skills=job.skills,
            requirement=job.requirement
        )
        
        # Get all candidates
        all_candidates = self.repository.get_all_candidates()
        if not all_candidates:
            logger.warning("No candidates found in database")
            return []
        
        logger.info(f"Computing similarity with {len(all_candidates)} candidates using 3 embeddings...")
        
        # Compute similarities for all candidates
        all_scores = []
        all_field_similarities = []
        
        for candidate in all_candidates:
            # Encode candidate into 3 embeddings
            candidate_embs = self._encode_candidate_fields(
                title=candidate.title,
                skills=candidate.skills,
                experience=candidate.experience
            )
            
            # Compute field similarities
            field_sims = self._compute_field_similarities(candidate_embs, job_embs)
            
            # Compute combined score
            combined_score = self._compute_combined_score(field_sims)
            
            all_scores.append(combined_score)
            all_field_similarities.append({
                'candidate': candidate,
                'field_similarities': field_sims
            })
        
        # Get top-k
        scores_array = np.array(all_scores)
        top_indices = np.argsort(scores_array)[::-1][:top_k]
        
        # Format results
        results = []
        for idx in top_indices:
            candidate = all_field_similarities[idx]['candidate']
            score = all_scores[idx]
            field_sims = all_field_similarities[idx]['field_similarities']
            
            results.append({
                'candidate_id': candidate.candidate_id,
                'name': candidate.name,
                'email': candidate.email,
                'score': score,
                'field_scores': {
                    'title': field_sims['title'],
                    'skills': field_sims['skills'],
                    'experience': field_sims['experience']
                }
            })
        
        logger.info(f"Found {len(results)} matching candidates")
        return results
