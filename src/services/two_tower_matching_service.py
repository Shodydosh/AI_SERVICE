"""Simple Two-Tower Matching Service - no multi-field, no weighted scoring."""
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
import logging
import torch
import numpy as np
from pathlib import Path
import sys

# Add two_tower to path
two_tower_path = Path(__file__).parent.parent.parent / "two_tower"
if str(two_tower_path.parent) not in sys.path:
    sys.path.insert(0, str(two_tower_path.parent))

from two_tower.model import TwoTowerModel
from src.database.two_tower_repository import TwoTowerRepository
from src.services.embedding_service import OptimizedEmbeddingService

logger = logging.getLogger(__name__)


class TwoTowerMatchingService:
    """Simple Two-Tower Matching Service - single embedding per candidate/job."""
    
    def __init__(
        self,
        db: Session,
        model_path: Optional[str] = None,
        device: str = 'cpu'
    ):
        """
        Initialize Two-Tower matching service.
        
        Args:
            db: Database session
            model_path: Path to trained two-tower model checkpoint
            device: Device to run model on
        """
        self.db = db
        self.repository = TwoTowerRepository(db)
        self.device = device
        
        # Load two-tower model
        if model_path is None:
            model_path = "outputs_improved/best_model_improved.pt"
        
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load two-tower model."""
        try:
            logger.info(f"Loading Two-Tower model from: {self.model_path}")
            self.model = TwoTowerModel(
                candidate_model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
                job_model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
                output_dim=768
            )
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            
            self.model.to(self.device)
            self.model.eval()
            logger.info("✓ Two-Tower model loaded")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _build_candidate_text(self, candidate) -> str:
        """Build candidate text from database record."""
        parts = []
        if candidate.title:
            parts.append(f"Title: {candidate.title}")
        if candidate.skills:
            parts.append(f"Skills: {candidate.skills}")
        if candidate.experience:
            parts.append(f"Experience: {candidate.experience}")
        return " | ".join(parts) if parts else ""
    
    def _build_job_text(self, job) -> str:
        """Build job text from database record."""
        parts = []
        if job.title:
            parts.append(f"Title: {job.title}")
        if job.skills:
            parts.append(f"Skills: {job.skills}")
        if job.requirement:
            parts.append(f"Requirements: {job.requirement}")
        return " | ".join(parts) if parts else ""
    
    def find_jobs_for_candidate(
        self,
        candidate_id: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Find top matching jobs for a candidate using simple two-tower similarity.
        
        Args:
            candidate_id: Candidate ID
            top_k: Number of top matches to return
        
        Returns:
            List of job matches with similarity scores
        """
        logger.info(f"Two-Tower matching for candidate: {candidate_id}")
        
        # Get candidate from database
        candidate = self.repository.get_candidate(candidate_id)
        if not candidate:
            logger.error(f"Candidate {candidate_id} not found")
            return []
        
        # Build candidate text
        candidate_text = self._build_candidate_text(candidate)
        if not candidate_text:
            logger.error(f"Candidate {candidate_id} has no text data")
            return []
        
        # Get all jobs
        all_jobs = self.repository.get_all_jobs()
        if not all_jobs:
            logger.warning("No jobs found in database")
            return []
        
        logger.info(f"Computing similarity with {len(all_jobs)} jobs...")
        
        # Build job texts
        job_texts = []
        job_records = []
        for job in all_jobs:
            job_text = self._build_job_text(job)
            if job_text:
                job_texts.append(job_text)
                job_records.append(job)
        
        if not job_texts:
            logger.warning("No valid job texts found")
            return []
        
        # Encode candidate
        with torch.no_grad():
            candidate_emb = self.model.encode_candidates([candidate_text])[0]  # [output_dim]
        
        # Encode all jobs in batches
        batch_size = 32
        all_job_embs = []
        with torch.no_grad():
            for i in range(0, len(job_texts), batch_size):
                batch_texts = job_texts[i:i+batch_size]
                batch_embs = self.model.encode_jobs(batch_texts)  # [batch_size, output_dim]
                all_job_embs.append(batch_embs.cpu().numpy())
        
        job_embs = np.vstack(all_job_embs)  # [num_jobs, output_dim]
        candidate_emb_np = candidate_emb.cpu().numpy()  # [output_dim]
        
        # Compute cosine similarity (embeddings are already normalized)
        similarities = np.dot(job_embs, candidate_emb_np)  # [num_jobs]
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Format results
        results = []
        for idx in top_indices:
            job = job_records[idx]
            score = float(similarities[idx])
            results.append({
                'job_id': job.job_id,
                'title': job.title,
                'company': job.company,
                'location': job.location,
                'score': score
            })
        
        logger.info(f"Found {len(results)} matching jobs")
        return results
    
    def find_candidates_for_job(
        self,
        job_id: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Find top matching candidates for a job using simple two-tower similarity.
        
        Args:
            job_id: Job ID
            top_k: Number of top matches to return
        
        Returns:
            List of candidate matches with similarity scores
        """
        logger.info(f"Two-Tower matching for job: {job_id}")
        
        # Get job from database
        job = self.repository.get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return []
        
        # Build job text
        job_text = self._build_job_text(job)
        if not job_text:
            logger.error(f"Job {job_id} has no text data")
            return []
        
        # Get all candidates
        all_candidates = self.repository.get_all_candidates()
        if not all_candidates:
            logger.warning("No candidates found in database")
            return []
        
        logger.info(f"Computing similarity with {len(all_candidates)} candidates...")
        
        # Build candidate texts
        candidate_texts = []
        candidate_records = []
        for candidate in all_candidates:
            candidate_text = self._build_candidate_text(candidate)
            if candidate_text:
                candidate_texts.append(candidate_text)
                candidate_records.append(candidate)
        
        if not candidate_texts:
            logger.warning("No valid candidate texts found")
            return []
        
        # Encode job
        with torch.no_grad():
            job_emb = self.model.encode_jobs([job_text])[0]  # [output_dim]
        
        # Encode all candidates in batches
        batch_size = 32
        all_candidate_embs = []
        with torch.no_grad():
            for i in range(0, len(candidate_texts), batch_size):
                batch_texts = candidate_texts[i:i+batch_size]
                batch_embs = self.model.encode_candidates(batch_texts)  # [batch_size, output_dim]
                all_candidate_embs.append(batch_embs.cpu().numpy())
        
        candidate_embs = np.vstack(all_candidate_embs)  # [num_candidates, output_dim]
        job_emb_np = job_emb.cpu().numpy()  # [output_dim]
        
        # Compute cosine similarity (embeddings are already normalized)
        similarities = np.dot(candidate_embs, job_emb_np)  # [num_candidates]
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Format results
        results = []
        for idx in top_indices:
            candidate = candidate_records[idx]
            score = float(similarities[idx])
            results.append({
                'candidate_id': candidate.candidate_id,
                'name': candidate.name,
                'email': candidate.email,
                'score': score
            })
        
        logger.info(f"Found {len(results)} matching candidates")
        return results
