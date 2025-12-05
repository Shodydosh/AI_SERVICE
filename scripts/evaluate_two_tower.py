"""Script to evaluate Two-Tower model and compare with baseline."""
import sys
import os
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader
from sqlalchemy.orm import Session

from src.database.connection import SessionLocal
from src.models.two_tower_model import TwoTowerModel
from src.models.training_pipeline import GroundTruthDataset
from src.models.evaluation_metrics import TwoTowerEvaluator
from src.database.multi_field_repository import MultiFieldEmbeddingRepository
from src.embeddings.multi_field_generator import MultiFieldEmbeddingGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_baseline_similarity(
    candidate_title_emb: np.ndarray,
    candidate_skills_emb: np.ndarray,
    candidate_experience_emb: np.ndarray,
    job_title_emb: np.ndarray,
    job_skills_emb: np.ndarray,
    job_requirement_emb: np.ndarray,
    weights: dict = {'title': 0.2, 'skills': 0.4, 'experience': 0.4}
) -> float:
    """
    Compute baseline similarity using weighted average of field similarities.
    
    Args:
        candidate_title_emb: Candidate title embedding
        candidate_skills_emb: Candidate skills embedding
        candidate_experience_emb: Candidate experience embedding
        job_title_emb: Job title embedding
        job_skills_emb: Job skills embedding
        job_requirement_emb: Job requirement embedding
        weights: Weights for each field
    
    Returns:
        Combined similarity score
    """
    # Title similarity
    title_sim = np.dot(candidate_title_emb, job_title_emb)
    
    # Skills similarity
    skills_sim = np.dot(candidate_skills_emb, job_skills_emb)
    
    # Experience-Requirement similarity
    exp_sim = np.dot(candidate_experience_emb, job_requirement_emb)
    
    # Weighted combination
    combined = (
        title_sim * weights['title'] +
        skills_sim * weights['skills'] +
        exp_sim * weights['experience']
    )
    
    return combined


def evaluate_baseline(
    ground_truth_path: str,
    embedding_dim: int = 768
) -> dict:
    """Evaluate baseline method."""
    logger.info("Evaluating baseline method...")
    
    from src.models.ground_truth_builder import GroundTruthBuilder
    db = SessionLocal()
    builder = GroundTruthBuilder(db)
    ground_truth = builder.load_ground_truth(ground_truth_path)
    
    repository = MultiFieldEmbeddingRepository(db)
    evaluator = TwoTowerEvaluator()
    
    all_predictions = []
    all_labels = []
    all_candidate_ids = []
    all_job_ids = []
    all_field_similarities = {
        'title': [],
        'skills': [],
        'experience': []
    }
    
    logger.info("Computing baseline similarities...")
    for pair in ground_truth:
        candidate = repository.get_candidate_multi_embedding(pair['candidate_id'])
        job = repository.get_job_multi_embedding(pair['job_id'])
        
        if not candidate or not job:
            continue
        
        cand_title = np.array(candidate.title_embedding, dtype=np.float32)
        cand_skills = np.array(candidate.skills_embedding, dtype=np.float32)
        cand_exp = np.array(candidate.experience_embedding, dtype=np.float32)
        
        job_title = np.array(job.title_embedding, dtype=np.float32)
        job_skills = np.array(job.skills_embedding, dtype=np.float32)
        job_req = np.array(job.requirement_embedding, dtype=np.float32)
        
        # Normalize
        cand_title = cand_title / (np.linalg.norm(cand_title) + 1e-8)
        cand_skills = cand_skills / (np.linalg.norm(cand_skills) + 1e-8)
        cand_exp = cand_exp / (np.linalg.norm(cand_exp) + 1e-8)
        job_title = job_title / (np.linalg.norm(job_title) + 1e-8)
        job_skills = job_skills / (np.linalg.norm(job_skills) + 1e-8)
        job_req = job_req / (np.linalg.norm(job_req) + 1e-8)
        
        # Compute similarities
        title_sim = float(np.dot(cand_title, job_title))
        skills_sim = float(np.dot(cand_skills, job_skills))
        exp_sim = float(np.dot(cand_exp, job_req))
        
        # Combined similarity
        combined = compute_baseline_similarity(
            cand_title, cand_skills, cand_exp,
            job_title, job_skills, job_req
        )
        
        all_predictions.append(combined)
        all_labels.append(pair['label'])
        all_candidate_ids.append(pair['candidate_id'])
        all_job_ids.append(pair['job_id'])
        all_field_similarities['title'].append(title_sim)
        all_field_similarities['skills'].append(skills_sim)
        all_field_similarities['experience'].append(exp_sim)
    
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    # Compute metrics
    metrics = evaluator.evaluate(
        predictions, labels, all_candidate_ids, all_job_ids,
        field_similarities=all_field_similarities
    )
    
    db.close()
    
    return metrics


def evaluate_two_tower(
    model_path: str,
    ground_truth_path: str,
    embedding_dim: int = 768,
    hidden_dims: list = [512, 256],
    output_dim: int = 256,
    batch_size: int = 32,
    device: str = 'cpu'
) -> dict:
    """Evaluate Two-Tower model."""
    logger.info("Evaluating Two-Tower model...")
    
    from src.models.ground_truth_builder import GroundTruthBuilder
    db = SessionLocal()
    builder = GroundTruthBuilder(db)
    ground_truth = builder.load_ground_truth(ground_truth_path)
    
    from src.models.training_pipeline import collate_fn
    
    repository = MultiFieldEmbeddingRepository(db)
    dataset = GroundTruthDataset(ground_truth, repository)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Load model
    model = TwoTowerModel(
        embedding_dim=embedding_dim,
        candidate_hidden_dims=hidden_dims,
        job_hidden_dims=hidden_dims,
        output_dim=output_dim
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    evaluator = TwoTowerEvaluator()
    
    all_predictions = []
    all_labels = []
    all_candidate_ids = []
    all_job_ids = []
    
    logger.info("Computing Two-Tower predictions...")
    with torch.no_grad():
        for batch in dataloader:
            candidate_title = batch['candidate_title'].to(device)
            candidate_skills = batch['candidate_skills'].to(device)
            candidate_experience = batch['candidate_experience'].to(device)
            job_title = batch['job_title'].to(device)
            job_skills = batch['job_skills'].to(device)
            job_requirement = batch['job_requirement'].to(device)
            labels = batch['label'].to(device)
            
            candidate_repr, job_repr = model(
                candidate_title, candidate_skills, candidate_experience,
                job_title, job_skills, job_requirement
            )
            
            similarity = model.compute_similarity(candidate_repr, job_repr)
            
            all_predictions.extend(similarity.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_candidate_ids.extend(batch['candidate_id'])
            all_job_ids.extend(batch['job_id'])
    
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    # Compute metrics
    metrics = evaluator.evaluate(
        predictions, labels, all_candidate_ids, all_job_ids
    )
    
    db.close()
    
    return metrics


def compare_methods(
    baseline_metrics: dict,
    two_tower_metrics: dict
):
    """Compare baseline and Two-Tower metrics."""
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON: Baseline vs Two-Tower")
    logger.info("=" * 60)
    
    metrics_to_compare = [
        'accuracy', 'precision', 'recall', 'f1',
        'auc_roc', 'auc_pr',
        'ndcg@10', 'mrr',
        'precision@5', 'precision@10',
        'recall@5', 'recall@10'
    ]
    
    logger.info(f"\n{'Metric':<20} {'Baseline':<15} {'Two-Tower':<15} {'Improvement':<15}")
    logger.info("-" * 65)
    
    for metric in metrics_to_compare:
        baseline_val = baseline_metrics.get(metric, 0.0)
        two_tower_val = two_tower_metrics.get(metric, 0.0)
        improvement = two_tower_val - baseline_val
        improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0.0
        
        logger.info(
            f"{metric:<20} {baseline_val:<15.4f} {two_tower_val:<15.4f} "
            f"{improvement:+.4f} ({improvement_pct:+.2f}%)"
        )


def main():
    parser = argparse.ArgumentParser(description='Evaluate Two-Tower Model')
    parser.add_argument(
        '--ground-truth-path',
        type=str,
        default='data/ground_truth.json',
        help='Path to ground truth JSON file'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/two_tower/best_model.pt',
        help='Path to trained Two-Tower model'
    )
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=768,
        help='Embedding dimension'
    )
    parser.add_argument(
        '--hidden-dims',
        type=int,
        nargs='+',
        default=[512, 256],
        help='Hidden layer dimensions'
    )
    parser.add_argument(
        '--output-dim',
        type=int,
        default=256,
        help='Output representation dimension'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use'
    )
    parser.add_argument(
        '--baseline-only',
        action='store_true',
        help='Only evaluate baseline'
    )
    parser.add_argument(
        '--two-tower-only',
        action='store_true',
        help='Only evaluate Two-Tower'
    )
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    try:
        # Evaluate baseline
        if not args.two_tower_only:
            logger.info("\n" + "=" * 60)
            logger.info("BASELINE EVALUATION")
            logger.info("=" * 60)
            baseline_metrics = evaluate_baseline(
                args.ground_truth_path,
                args.embedding_dim
            )
            evaluator = TwoTowerEvaluator()
            evaluator.print_metrics(baseline_metrics)
        else:
            baseline_metrics = None
        
        # Evaluate Two-Tower
        if not args.baseline_only:
            logger.info("\n" + "=" * 60)
            logger.info("TWO-TOWER EVALUATION")
            logger.info("=" * 60)
            two_tower_metrics = evaluate_two_tower(
                args.model_path,
                args.ground_truth_path,
                args.embedding_dim,
                args.hidden_dims,
                args.output_dim,
                args.batch_size,
                args.device
            )
            evaluator = TwoTowerEvaluator()
            evaluator.print_metrics(two_tower_metrics)
        else:
            two_tower_metrics = None
        
        # Compare
        if baseline_metrics and two_tower_metrics:
            compare_methods(baseline_metrics, two_tower_metrics)
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

