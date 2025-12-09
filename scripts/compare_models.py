"""Compare all available models and select the one with highest similarity."""
import sys
import os
import argparse
import logging
from pathlib import Path
import json
import csv
import glob

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_ground_truth_csv(csv_path: str) -> list:
    """Load ground truth from CSV."""
    logger.info(f"Loading ground truth from: {csv_path}")
    ground_truth = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            similarity_type = str(row.get('similarity_type', 'random')).lower()
            if similarity_type == 'high':
                label = 1.0
            elif similarity_type == 'medium':
                label = 0.7
            else:
                label = 0.0
            
            ground_truth.append({
                'candidate_id': str(row['candidate_id']),
                'job_id': str(row['job_id']),
                'label': label,
                'similarity_type': similarity_type,
            })
    
    logger.info(f"Loaded {len(ground_truth)} pairs")
    logger.info(f"  - High: {sum(1 for p in ground_truth if p['similarity_type'] == 'high')}")
    logger.info(f"  - Medium: {sum(1 for p in ground_truth if p['similarity_type'] == 'medium')}")
    logger.info(f"  - Random: {sum(1 for p in ground_truth if p['similarity_type'] == 'random')}")
    
    return ground_truth


def evaluate_model(model_path: str, ground_truth: list, batch_size: int = 4, device: str = 'cpu'):
    """Evaluate a single model."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating: {model_path}")
    logger.info(f"{'='*80}")
    
    # Load model
    from pathlib import Path
    two_tower_path = Path(__file__).parent.parent / "two_tower"
    if str(two_tower_path) not in sys.path:
        sys.path.insert(0, str(two_tower_path.parent))
    
    from two_tower.model import TwoTowerModel
    
    model = TwoTowerModel(
        candidate_model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
        job_model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
        output_dim=768
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    model.eval()
    logger.info("✓ Model loaded")
    
    # Load text data from database
    from src.database.connection import SessionLocal
    from src.database.models import CandidateMultiEmbedding, JobDescriptionMultiEmbedding
    
    db = SessionLocal()
    
    candidate_texts = []
    job_texts = []
    all_labels = []
    all_similarity_types = []
    
    for pair in ground_truth:
        cand_id = pair['candidate_id']
        job_id = pair['job_id']
        
        cand = db.query(CandidateMultiEmbedding).filter(
            CandidateMultiEmbedding.candidate_id == cand_id
        ).first()
        
        job = db.query(JobDescriptionMultiEmbedding).filter(
            JobDescriptionMultiEmbedding.job_id == job_id
        ).first()
        
        if cand and job:
            cand_text_parts = []
            if cand.title:
                cand_text_parts.append(f"Title: {cand.title}")
            if cand.skills:
                cand_text_parts.append(f"Skills: {cand.skills}")
            if cand.experience:
                cand_text_parts.append(f"Experience: {cand.experience}")
            candidate_text = " | ".join(cand_text_parts) if cand_text_parts else ""
            
            job_text_parts = []
            if job.title:
                job_text_parts.append(f"Title: {job.title}")
            if job.skills:
                job_text_parts.append(f"Skills: {job.skills}")
            if job.requirement:
                job_text_parts.append(f"Requirements: {job.requirement}")
            job_text = " | ".join(job_text_parts) if job_text_parts else ""
            
            if candidate_text and job_text:
                candidate_texts.append(candidate_text)
                job_texts.append(job_text)
                all_labels.append(pair['label'])
                all_similarity_types.append(pair.get('similarity_type', 'unknown'))
    
    db.close()
    
    if len(candidate_texts) == 0:
        raise ValueError("No valid pairs found")
    
    logger.info(f"✓ Loaded {len(candidate_texts)} valid pairs")
    
    # Compute predictions
    all_predictions = []
    
    with torch.no_grad():
        for i in range(0, len(candidate_texts), batch_size):
            batch_cand_texts = candidate_texts[i:i+batch_size]
            batch_job_texts = job_texts[i:i+batch_size]
            
            candidate_emb = model.encode_candidates(batch_cand_texts)
            job_emb = model.encode_jobs(batch_job_texts)
            
            similarity = torch.sum(candidate_emb * job_emb, dim=1).cpu().numpy()
            all_predictions.extend(similarity)
    
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    # Compute key metrics
    binary_labels = (labels >= 0.7).astype(int)
    
    try:
        auc_roc = roc_auc_score(binary_labels, predictions)
    except:
        auc_roc = 0.0
    
    # Mean similarity by type
    high_mask = np.array([t == 'high' for t in all_similarity_types])
    medium_mask = np.array([t == 'medium' for t in all_similarity_types])
    random_mask = np.array([t == 'random' for t in all_similarity_types])
    
    high_mean = float(np.mean(predictions[high_mask])) if np.any(high_mask) else 0.0
    medium_mean = float(np.mean(predictions[medium_mask])) if np.any(medium_mask) else 0.0
    random_mean = float(np.mean(predictions[random_mask])) if np.any(random_mask) else 0.0
    
    # Overall mean similarity
    overall_mean = float(np.mean(predictions))
    
    # Separation score (difference between high and random)
    separation = high_mean - random_mean
    
    metrics = {
        'model_path': model_path,
        'auc_roc': auc_roc,
        'overall_mean_similarity': overall_mean,
        'high_mean': high_mean,
        'medium_mean': medium_mean,
        'random_mean': random_mean,
        'separation': separation,  # How well it separates high from random
        'pred_min': float(np.min(predictions)),
        'pred_max': float(np.max(predictions)),
        'pred_std': float(np.std(predictions)),
    }
    
    logger.info(f"  AUC-ROC: {auc_roc:.4f}")
    logger.info(f"  Overall Mean Similarity: {overall_mean:.4f}")
    logger.info(f"  High Mean: {high_mean:.4f}")
    logger.info(f"  Medium Mean: {medium_mean:.4f}")
    logger.info(f"  Random Mean: {random_mean:.4f}")
    logger.info(f"  Separation (High - Random): {separation:.4f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Compare models and select best')
    parser.add_argument('--ground-truth-csv', type=str, default='ground_truth_500_pairs.csv',
                        help='Path to ground truth CSV')
    parser.add_argument('--model-dir', type=str, default='outputs_improved',
                        help='Directory containing model checkpoints')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for evaluation')
    parser.add_argument('--output-file', type=str, default='model_comparison.json',
                        help='Output file for comparison results')
    
    args = parser.parse_args()
    
    # Find all model files
    model_dir = Path(args.model_dir)
    model_files = list(model_dir.glob("*.pt"))
    
    if not model_files:
        logger.error(f"No model files found in {model_dir}")
        return
    
    logger.info(f"Found {len(model_files)} model(s):")
    for mf in model_files:
        logger.info(f"  - {mf}")
    
    # Load ground truth
    ground_truth = load_ground_truth_csv(args.ground_truth_csv)
    
    # Evaluate each model
    all_metrics = []
    for model_path in model_files:
        try:
            metrics = evaluate_model(str(model_path), ground_truth, args.batch_size)
            all_metrics.append(metrics)
        except Exception as e:
            logger.error(f"Error evaluating {model_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    if not all_metrics:
        logger.error("No models evaluated successfully")
        return
    
    # Sort by separation score (best separation = best model)
    all_metrics.sort(key=lambda x: x['separation'], reverse=True)
    
    # Print comparison
    logger.info("\n" + "="*80)
    logger.info("📊 MODEL COMPARISON RESULTS")
    logger.info("="*80)
    
    for i, metrics in enumerate(all_metrics, 1):
        logger.info(f"\n{i}. {Path(metrics['model_path']).name}")
        logger.info(f"   Separation: {metrics['separation']:.4f}")
        logger.info(f"   AUC-ROC: {metrics['auc_roc']:.4f}")
        logger.info(f"   Overall Mean: {metrics['overall_mean_similarity']:.4f}")
        logger.info(f"   High Mean: {metrics['high_mean']:.4f}")
        logger.info(f"   Random Mean: {metrics['random_mean']:.4f}")
    
    # Select best model
    best_model = all_metrics[0]
    logger.info("\n" + "="*80)
    logger.info("🏆 BEST MODEL (Highest Separation)")
    logger.info("="*80)
    logger.info(f"Model: {best_model['model_path']}")
    logger.info(f"Separation: {best_model['separation']:.4f}")
    logger.info(f"AUC-ROC: {best_model['auc_roc']:.4f}")
    logger.info(f"High Mean Similarity: {best_model['high_mean']:.4f}")
    logger.info(f"Random Mean Similarity: {best_model['random_mean']:.4f}")
    
    # Save results
    results = {
        'best_model': best_model['model_path'],
        'comparison': all_metrics
    }
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✓ Results saved to: {args.output_file}")
    
    return best_model['model_path']


if __name__ == '__main__':
    main()








