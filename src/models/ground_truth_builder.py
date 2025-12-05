"""Ground Truth Dataset Builder for Two-Tower Model Training."""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sqlalchemy.orm import Session
import logging
from collections import defaultdict
import json

from src.database.multi_field_repository import MultiFieldEmbeddingRepository
from src.database.models import (
    JobDescriptionMultiEmbedding,
    CandidateMultiEmbedding
)

logger = logging.getLogger(__name__)


class GroundTruthBuilder:
    """
    Build ground truth dataset for training Two-Tower model.
    
    Creates positive and negative pairs based on:
    1. Field similarity thresholds
    2. Manual annotations (if available)
    3. Heuristic rules (title match, skills overlap, etc.)
    """
    
    def __init__(
        self,
        db: Session,
        title_similarity_threshold: float = 0.6,
        skills_similarity_threshold: float = 0.5,
        experience_similarity_threshold: float = 0.5,
        combined_threshold: float = 0.55
    ):
        """
        Initialize Ground Truth Builder.
        
        Args:
            db: Database session
            title_similarity_threshold: Minimum title similarity for positive pair
            skills_similarity_threshold: Minimum skills similarity for positive pair
            experience_similarity_threshold: Minimum experience similarity for positive pair
            combined_threshold: Minimum combined similarity for positive pair
        """
        self.db = db
        self.repository = MultiFieldEmbeddingRepository(db)
        self.title_threshold = title_similarity_threshold
        self.skills_threshold = skills_similarity_threshold
        self.experience_threshold = experience_similarity_threshold
        self.combined_threshold = combined_threshold
    
    def compute_field_similarities(
        self,
        candidate: CandidateMultiEmbedding,
        job: JobDescriptionMultiEmbedding
    ) -> Dict[str, float]:
        """
        Compute similarity scores for each field.
        
        Args:
            candidate: Candidate record
            job: Job record
        
        Returns:
            Dictionary with similarity scores for each field
        """
        similarities = {}
        
        # Title similarity
        if candidate.title_embedding and job.title_embedding:
            cand_title = np.array(candidate.title_embedding)
            job_title = np.array(job.title_embedding)
            cand_norm = np.linalg.norm(cand_title)
            job_norm = np.linalg.norm(job_title)
            if cand_norm > 0 and job_norm > 0:
                similarities['title'] = float(np.dot(cand_title, job_title))
            else:
                similarities['title'] = 0.0
        else:
            similarities['title'] = 0.0
        
        # Skills similarity
        if candidate.skills_embedding and job.skills_embedding:
            cand_skills = np.array(candidate.skills_embedding)
            job_skills = np.array(job.skills_embedding)
            cand_norm = np.linalg.norm(cand_skills)
            job_norm = np.linalg.norm(job_skills)
            if cand_norm > 0 and job_norm > 0:
                similarities['skills'] = float(np.dot(cand_skills, job_skills))
            else:
                similarities['skills'] = 0.0
        else:
            similarities['skills'] = 0.0
        
        # Experience-Requirement similarity
        if candidate.experience_embedding and job.requirement_embedding:
            cand_exp = np.array(candidate.experience_embedding)
            job_req = np.array(job.requirement_embedding)
            cand_norm = np.linalg.norm(cand_exp)
            job_norm = np.linalg.norm(job_req)
            if cand_norm > 0 and job_norm > 0:
                similarities['experience'] = float(np.dot(cand_exp, job_req))
            else:
                similarities['experience'] = 0.0
        else:
            similarities['experience'] = 0.0
        
        # Combined similarity (weighted average)
        similarities['combined'] = (
            similarities['title'] * 0.2 +
            similarities['skills'] * 0.4 +
            similarities['experience'] * 0.4
        )
        
        return similarities
    
    def is_positive_pair(
        self,
        similarities: Dict[str, float]
    ) -> bool:
        """
        Determine if a candidate-job pair is positive based on thresholds.
        
        Args:
            similarities: Dictionary of similarity scores
        
        Returns:
            True if positive pair, False otherwise
        """
        # Check individual thresholds
        title_ok = similarities['title'] >= self.title_threshold
        skills_ok = similarities['skills'] >= self.skills_threshold
        experience_ok = similarities['experience'] >= self.experience_threshold
        
        # Positive if combined score is high OR at least 2 fields pass threshold
        combined_ok = similarities['combined'] >= self.combined_threshold
        field_count = sum([title_ok, skills_ok, experience_ok])
        
        return combined_ok or field_count >= 2
    
    def build_ground_truth_dataset(
        self,
        max_candidates: Optional[int] = None,
        max_jobs: Optional[int] = None,
        positive_ratio: float = 0.3,
        negative_ratio: float = 0.7,
        min_positive_pairs: int = 100,
        min_negative_pairs: int = 200
    ) -> List[Dict]:
        """
        Build ground truth dataset with positive and negative pairs.
        
        Args:
            max_candidates: Maximum number of candidates to consider
            max_jobs: Maximum number of jobs to consider
            positive_ratio: Ratio of positive pairs
            negative_ratio: Ratio of negative pairs
            min_positive_pairs: Minimum number of positive pairs
            min_negative_pairs: Minimum number of negative pairs
        
        Returns:
            List of ground truth pairs with labels
        """
        logger.info("Building ground truth dataset...")
        
        # Load candidates and jobs
        all_candidates = self.repository.get_all_candidate_multi_embeddings()
        all_jobs = self.repository.get_all_job_multi_embeddings()
        
        if max_candidates:
            all_candidates = all_candidates[:max_candidates]
        if max_jobs:
            all_jobs = all_jobs[:max_jobs]
        
        logger.info(f"Loaded {len(all_candidates)} candidates and {len(all_jobs)} jobs")
        
        # Build positive pairs
        positive_pairs = []
        negative_pairs = []
        
        logger.info("Computing similarities and labeling pairs...")
        
        for candidate in all_candidates:
            candidate_similarities = []
            
            for job in all_jobs:
                similarities = self.compute_field_similarities(candidate, job)
                
                pair_data = {
                    'candidate_id': candidate.candidate_id,
                    'job_id': job.job_id,
                    'title_similarity': similarities['title'],
                    'skills_similarity': similarities['skills'],
                    'experience_similarity': similarities['experience'],
                    'combined_similarity': similarities['combined']
                }
                
                if self.is_positive_pair(similarities):
                    pair_data['label'] = 1
                    positive_pairs.append(pair_data)
                else:
                    pair_data['label'] = 0
                    candidate_similarities.append(pair_data)
            
            # Select negative pairs (low similarity)
            candidate_similarities.sort(key=lambda x: x['combined_similarity'])
            # Take bottom 20% as hard negatives
            n_negatives = max(1, len(candidate_similarities) // 5)
            negative_pairs.extend(candidate_similarities[:n_negatives])
        
        logger.info(f"Found {len(positive_pairs)} positive pairs")
        logger.info(f"Found {len(negative_pairs)} negative pairs")
        
        # Balance dataset
        if len(positive_pairs) < min_positive_pairs:
            logger.warning(f"Only {len(positive_pairs)} positive pairs, need at least {min_positive_pairs}")
        
        if len(negative_pairs) > len(positive_pairs) * 3:
            # Sample negatives to balance
            np.random.seed(42)
            indices = np.random.choice(
                len(negative_pairs),
                min(len(negative_pairs), len(positive_pairs) * 2),
                replace=False
            )
            negative_pairs = [negative_pairs[i] for i in indices]
        
        # Combine and shuffle
        all_pairs = positive_pairs + negative_pairs
        np.random.seed(42)
        np.random.shuffle(all_pairs)
        
        logger.info(f"Final dataset: {len(all_pairs)} pairs ({len(positive_pairs)} positive, {len(negative_pairs)} negative)")
        
        return all_pairs
    
    def save_ground_truth(
        self,
        ground_truth: List[Dict],
        output_path: str
    ):
        """
        Save ground truth dataset to JSON file.
        
        Args:
            ground_truth: List of ground truth pairs
            output_path: Path to save JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ground_truth, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved ground truth dataset to {output_path}")
        logger.info(f"Total pairs: {len(ground_truth)}")
        logger.info(f"Positive pairs: {sum(1 for p in ground_truth if p['label'] == 1)}")
        logger.info(f"Negative pairs: {sum(1 for p in ground_truth if p['label'] == 0)}")
    
    def load_ground_truth(self, file_path: str) -> List[Dict]:
        """
        Load ground truth dataset from JSON file.
        
        Args:
            file_path: Path to JSON file
        
        Returns:
            List of ground truth pairs
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        
        logger.info(f"Loaded ground truth dataset from {file_path}")
        logger.info(f"Total pairs: {len(ground_truth)}")
        
        return ground_truth


