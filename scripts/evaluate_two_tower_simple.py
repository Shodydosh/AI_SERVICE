"""Simple Two-Tower evaluation without database dependency - uses embeddings from CSV if available."""
import sys
import os
import argparse
import logging
from pathlib import Path
import json
import csv

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score
)

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
                'predicted_similarity': float(row.get('predicted_similarity', 0.0))
            })
    
    logger.info(f"Loaded {len(ground_truth)} pairs")
    logger.info(f"  - High: {sum(1 for p in ground_truth if p['label'] == 1.0)}")
    logger.info(f"  - Medium: {sum(1 for p in ground_truth if p['label'] == 0.7)}")
    logger.info(f"  - Random: {sum(1 for p in ground_truth if p['label'] == 0.0)}")
    
    return ground_truth


def compute_ndcg(labels: np.ndarray, predictions: np.ndarray, k: int = 10) -> float:
    """Compute NDCG@K."""
    if len(labels) == 0:
        return 0.0
    
    k = min(k, len(labels))
    
    # Sort by predictions
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_labels = labels[sorted_indices]
    
    # DCG
    dcg = 0.0
    for i in range(k):
        rel = sorted_labels[i]
        dcg += rel / np.log2(i + 2)
    
    # Ideal DCG
    ideal_labels = np.sort(labels)[::-1]
    idcg = 0.0
    for i in range(k):
        rel = ideal_labels[i]
        idcg += rel / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0


def compute_mrr(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Compute Mean Reciprocal Rank."""
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_labels = labels[sorted_indices]
    
    for i, label in enumerate(sorted_labels):
        if label > 0.5:  # Positive
            return 1.0 / (i + 1)
    
    return 0.0


def evaluate_with_database(
    model_path: str,
    ground_truth: list,
    embedding_dim: int = 768,
    hidden_dims: list = [512, 256],
    output_dim: int = 768,
    batch_size: int = 16,
    device: str = 'cpu'
):
    """Evaluate using database text data."""
    logger.info("\n" + "=" * 80)
    logger.info("TWO-TOWER MODEL EVALUATION - DETAILED")
    logger.info("=" * 80)
    
    # Load model from two_tower module
    logger.info(f"\n1. Loading model: {model_path}")
    import sys
    from pathlib import Path
    two_tower_path = Path(__file__).parent.parent / "two_tower"
    if str(two_tower_path) not in sys.path:
        sys.path.insert(0, str(two_tower_path.parent))
    
    from two_tower.model import TwoTowerModel
    
    model = TwoTowerModel(
        candidate_model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
        job_model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
        output_dim=768  # Match checkpoint output_dim
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
    logger.info("\n2. Loading text data from database...")
    try:
        from src.database.connection import SessionLocal
        from src.database.models import CandidateMultiEmbedding, JobDescriptionMultiEmbedding
        
        db = SessionLocal()
        
        # Build text representations for all pairs
        candidate_texts = []
        job_texts = []
        all_labels = []
        all_similarity_types = []
        
        logger.info(f"Processing {len(ground_truth)} pairs...")
        for pair in ground_truth:
            cand_id = pair['candidate_id']
            job_id = pair['job_id']
            
            # Get candidate text
            cand = db.query(CandidateMultiEmbedding).filter(
                CandidateMultiEmbedding.candidate_id == cand_id
            ).first()
            
            # Get job text
            job = db.query(JobDescriptionMultiEmbedding).filter(
                JobDescriptionMultiEmbedding.job_id == job_id
            ).first()
            
            if cand and job:
                # Combine candidate fields into text
                cand_text_parts = []
                if cand.title:
                    cand_text_parts.append(f"Title: {cand.title}")
                if cand.skills:
                    cand_text_parts.append(f"Skills: {cand.skills}")
                if cand.experience:
                    cand_text_parts.append(f"Experience: {cand.experience}")
                candidate_text = " | ".join(cand_text_parts) if cand_text_parts else ""
                
                # Combine job fields into text
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
        logger.info(f"✓ Loaded {len(candidate_texts)} valid pairs")
        
        # Evaluate in batches
        logger.info("\n3. Computing predictions...")
        all_predictions = []
        
        with torch.no_grad():
            for i in range(0, len(candidate_texts), batch_size):
                if (i // batch_size + 1) % 10 == 0:
                    logger.info(f"  Batch {i // batch_size + 1}/{(len(candidate_texts) + batch_size - 1) // batch_size}")
                
                batch_cand_texts = candidate_texts[i:i+batch_size]
                batch_job_texts = job_texts[i:i+batch_size]
                
                # Encode candidates and jobs
                candidate_emb = model.encode_candidates(batch_cand_texts)
                job_emb = model.encode_jobs(batch_job_texts)
                
                # Compute cosine similarity
                similarity = torch.sum(candidate_emb * job_emb, dim=1).cpu().numpy()
                all_predictions.extend(similarity)
        
        if len(all_predictions) == 0:
            raise ValueError("No valid pairs found in database")
        
    except ImportError as e:
        logger.error(f"Cannot import database modules: {e}")
        logger.error("Please install dependencies: pip install sqlalchemy psycopg2-binary")
        raise
    except Exception as e:
        logger.error(f"Error loading from database: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    # Compute metrics
    logger.info("\n4. Computing metrics...")
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    # Binary labels for classification (high = 1, others = 0)
    binary_labels = (labels >= 0.7).astype(int)
    binary_predictions = (predictions >= 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(binary_labels, binary_predictions),
        'precision': precision_score(binary_labels, binary_predictions, zero_division=0),
        'recall': recall_score(binary_labels, binary_predictions, zero_division=0),
        'f1': f1_score(binary_labels, binary_predictions, zero_division=0),
    }
    
    try:
        metrics['auc_roc'] = roc_auc_score(binary_labels, predictions)
    except:
        metrics['auc_roc'] = 0.0
    
    try:
        metrics['auc_pr'] = average_precision_score(binary_labels, predictions)
    except:
        metrics['auc_pr'] = 0.0
    
    # Ranking metrics
    metrics['ndcg@10'] = compute_ndcg(labels, predictions, k=10)
    metrics['mrr'] = compute_mrr(labels, predictions)
    
    # Precision@K, Recall@K
    for k in [5, 10, 20]:
        if len(predictions) >= k:
            top_k_indices = np.argsort(predictions)[::-1][:k]
            top_k_labels = labels[top_k_indices]
            top_k_binary = (top_k_labels >= 0.7).astype(int)
            
            metrics[f'precision@{k}'] = precision_score(
                top_k_binary, np.ones_like(top_k_binary), zero_division=0
            ) if np.sum(top_k_binary) > 0 else 0.0
            
            metrics[f'recall@{k}'] = np.sum(top_k_binary) / np.sum(binary_labels) if np.sum(binary_labels) > 0 else 0.0
    
    # Analysis by type
    high_mask = np.array([t == 'high' for t in all_similarity_types])
    medium_mask = np.array([t == 'medium' for t in all_similarity_types])
    random_mask = np.array([t == 'random' for t in all_similarity_types])
    
    if np.any(high_mask):
        metrics['high_mean'] = float(np.mean(predictions[high_mask]))
        metrics['high_std'] = float(np.std(predictions[high_mask]))
        metrics['high_count'] = int(np.sum(high_mask))
    
    if np.any(medium_mask):
        metrics['medium_mean'] = float(np.mean(predictions[medium_mask]))
        metrics['medium_std'] = float(np.std(predictions[medium_mask]))
        metrics['medium_count'] = int(np.sum(medium_mask))
    
    if np.any(random_mask):
        metrics['random_mean'] = float(np.mean(predictions[random_mask]))
        metrics['random_std'] = float(np.std(predictions[random_mask]))
        metrics['random_count'] = int(np.sum(random_mask))
    
    # Distribution
    metrics['pred_mean'] = float(np.mean(predictions))
    metrics['pred_std'] = float(np.std(predictions))
    metrics['pred_min'] = float(np.min(predictions))
    metrics['pred_max'] = float(np.max(predictions))
    metrics['pred_median'] = float(np.median(predictions))
    
    return metrics, predictions, labels, all_similarity_types


def print_results(metrics: dict, predictions: np.ndarray, labels: np.ndarray, similarity_types: list):
    """Print detailed results."""
    logger.info("\n" + "=" * 80)
    logger.info("📊 DETAILED EVALUATION RESULTS")
    logger.info("=" * 80)
    
    # Classification
    logger.info("\n📊 CLASSIFICATION METRICS")
    logger.info("-" * 80)
    logger.info(f"  Accuracy:        {metrics.get('accuracy', 0):.4f}")
    logger.info(f"  Precision:       {metrics.get('precision', 0):.4f}")
    logger.info(f"  Recall:          {metrics.get('recall', 0):.4f}")
    logger.info(f"  F1-Score:        {metrics.get('f1', 0):.4f}")
    
    auc_roc = metrics.get('auc_roc', 0)
    auc_status = "⭐ Excellent" if auc_roc > 0.8 else "✓ Good" if auc_roc > 0.7 else "⚠ Needs Improvement"
    logger.info(f"  AUC-ROC:         {auc_roc:.4f}  {auc_status}")
    logger.info(f"  AUC-PR:          {metrics.get('auc_pr', 0):.4f}")
    
    # Ranking
    logger.info("\n📈 RANKING METRICS")
    logger.info("-" * 80)
    ndcg = metrics.get('ndcg@10', 0)
    ndcg_status = "⭐ Excellent" if ndcg > 0.8 else "✓ Good" if ndcg > 0.6 else "⚠ Needs Improvement"
    logger.info(f"  NDCG@10:         {ndcg:.4f}  {ndcg_status}")
    logger.info(f"  MRR:             {metrics.get('mrr', 0):.4f}")
    logger.info(f"  Precision@5:     {metrics.get('precision@5', 0):.4f}")
    logger.info(f"  Precision@10:    {metrics.get('precision@10', 0):.4f}")
    logger.info(f"  Recall@5:       {metrics.get('recall@5', 0):.4f}")
    logger.info(f"  Recall@10:      {metrics.get('recall@10', 0):.4f}")
    
    # Distribution
    logger.info("\n📉 PREDICTION SCORE DISTRIBUTION")
    logger.info("-" * 80)
    logger.info(f"  Mean:            {metrics.get('pred_mean', 0):.4f}")
    logger.info(f"  Std:             {metrics.get('pred_std', 0):.4f}")
    logger.info(f"  Min:             {metrics.get('pred_min', 0):.4f}")
    logger.info(f"  Max:             {metrics.get('pred_max', 0):.4f}")
    logger.info(f"  Median:          {metrics.get('pred_median', 0):.4f}")
    
    # By type
    logger.info("\n🔍 ANALYSIS BY SIMILARITY TYPE")
    logger.info("-" * 80)
    
    if 'high_count' in metrics:
        logger.info(f"\n  HIGH Similarity ({metrics['high_count']} pairs):")
        logger.info(f"    Mean Score:    {metrics.get('high_mean', 0):.4f}")
        logger.info(f"    Std:           {metrics.get('high_std', 0):.4f}")
        logger.info(f"    Expected:     0.8-1.0 ⭐")
    
    if 'medium_count' in metrics:
        logger.info(f"\n  MEDIUM Similarity ({metrics['medium_count']} pairs):")
        logger.info(f"    Mean Score:    {metrics.get('medium_mean', 0):.4f}")
        logger.info(f"    Std:           {metrics.get('medium_std', 0):.4f}")
        logger.info(f"    Expected:     0.5-0.8 ✓")
    
    if 'random_count' in metrics:
        logger.info(f"\n  RANDOM/LOW Similarity ({metrics['random_count']} pairs):")
        logger.info(f"    Mean Score:    {metrics.get('random_mean', 0):.4f}")
        logger.info(f"    Std:           {metrics.get('random_std', 0):.4f}")
        logger.info(f"    Expected:     0.0-0.4 ⚠")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📋 SUMMARY")
    logger.info("=" * 80)
    
    if auc_roc > 0.8 and ndcg > 0.7:
        logger.info("✅ EXCELLENT: Model performs very well!")
    elif auc_roc > 0.7 and ndcg > 0.6:
        logger.info("✓ GOOD: Model performs well")
    elif auc_roc > 0.6:
        logger.info("⚠ MODERATE: Model needs improvement")
    else:
        logger.info("❌ POOR: Model needs significant improvement")
    
    logger.info(f"\nKey Metrics:")
    logger.info(f"  - AUC-ROC: {auc_roc:.4f} (Higher is better, >0.8 is excellent)")
    logger.info(f"  - NDCG@10: {ndcg:.4f} (Higher is better, >0.7 is excellent)")
    logger.info(f"  - Precision@10: {metrics.get('precision@10', 0):.4f}")
    logger.info(f"  - Recall@10: {metrics.get('recall@10', 0):.4f}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Simple Two-Tower Evaluation')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model .pt file')
    parser.add_argument('--ground-truth-csv', type=str, required=True, help='Path to ground truth CSV')
    parser.add_argument('--embedding-dim', type=int, default=768)
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[512, 256])
    parser.add_argument('--output-dim', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--output-file', type=str, help='Save results to JSON')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load ground truth
    ground_truth = load_ground_truth_csv(args.ground_truth_csv)
    
    # Evaluate
    try:
        metrics, predictions, labels, similarity_types = evaluate_with_database(
            model_path=args.model_path,
            ground_truth=ground_truth,
            embedding_dim=args.embedding_dim,
            hidden_dims=args.hidden_dims,
            output_dim=args.output_dim,
            batch_size=args.batch_size,
            device=args.device
        )
        
        print_results(metrics, predictions, labels, similarity_types)
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"\n✓ Results saved to: {args.output_file}")
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        logger.error("\nTroubleshooting:")
        logger.error("1. Ensure database is running and embeddings are processed")
        logger.error("2. Install dependencies: pip install sqlalchemy psycopg2-binary")
        logger.error("3. Check model path is correct")
        raise


if __name__ == '__main__':
    main()

