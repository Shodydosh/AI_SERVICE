"""Detailed evaluation script for Two-Tower model with comprehensive metrics."""
import sys
import os
import argparse
import logging
from pathlib import Path
import json
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    import csv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader
from sqlalchemy.orm import Session

from src.database.connection import SessionLocal
from src.models.two_tower_model import TwoTowerModel
from src.models.training_pipeline import GroundTruthDataset, collate_fn
from src.models.evaluation_metrics import TwoTowerEvaluator
from src.database.multi_field_repository import MultiFieldEmbeddingRepository
from src.models.ground_truth_builder import GroundTruthBuilder

# Setup logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_ground_truth_from_csv(csv_path: str) -> list:
    """Load ground truth pairs from CSV file."""
    logger.info(f"Loading ground truth from CSV: {csv_path}")
    
    ground_truth = []
    
    if PANDAS_AVAILABLE:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            # Convert similarity_type to label
            similarity_type = str(row.get('similarity_type', 'random')).lower()
            if similarity_type == 'high':
                label = 1.0
            elif similarity_type == 'medium':
                label = 0.7  # Medium similarity
            else:  # random
                label = 0.0
            
            ground_truth.append({
                'candidate_id': str(row['candidate_id']),
                'job_id': str(row['job_id']),
                'label': label,
                'similarity_type': similarity_type,
                'predicted_similarity': float(row.get('predicted_similarity', 0.0))
            })
    else:
        # Use csv module if pandas not available
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
    
    logger.info(f"Loaded {len(ground_truth)} pairs from CSV")
    logger.info(f"  - High similarity: {sum(1 for p in ground_truth if p['label'] == 1.0)}")
    logger.info(f"  - Medium similarity: {sum(1 for p in ground_truth if p['label'] == 0.7)}")
    logger.info(f"  - Random/Low similarity: {sum(1 for p in ground_truth if p['label'] == 0.0)}")
    
    return ground_truth


def evaluate_two_tower_detailed(
    model_path: str,
    ground_truth: list,
    embedding_dim: int = 768,
    hidden_dims: list = [512, 256],
    output_dim: int = 256,
    batch_size: int = 32,
    device: str = 'cpu'
) -> dict:
    """Evaluate Two-Tower model with detailed metrics."""
    logger.info("\n" + "=" * 80)
    logger.info("TWO-TOWER MODEL EVALUATION - DETAILED")
    logger.info("=" * 80)
    
    # Load model
    logger.info(f"\n1. Loading model from: {model_path}")
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = TwoTowerModel(
        embedding_dim=embedding_dim,
        candidate_hidden_dims=hidden_dims,
        job_hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout=0.1,
        use_batch_norm=True
    )
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    model.to(device)
    model.eval()
    
    # Load embeddings from database
    logger.info("\n2. Loading embeddings from database...")
    db = SessionLocal()
    try:
        repository = MultiFieldEmbeddingRepository(db)
        
        # Create dataset
        dataset = GroundTruthDataset(ground_truth, repository)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        logger.info(f"✓ Dataset created: {len(dataset)} samples")
        
        # Evaluate
        logger.info("\n3. Computing predictions...")
        evaluator = TwoTowerEvaluator()
        
        all_predictions = []
        all_labels = []
        all_candidate_ids = []
        all_job_ids = []
        all_similarity_types = []
        all_predicted_similarities = []
        
        num_batches = len(dataloader)
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"  Processing batch {batch_idx + 1}/{num_batches}...")
                
                candidate_title = batch['candidate_title'].to(device)
                candidate_skills = batch['candidate_skills'].to(device)
                candidate_experience = batch['candidate_experience'].to(device)
                job_title = batch['job_title'].to(device)
                job_skills = batch['job_skills'].to(device)
                job_requirement = batch['job_requirement'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                candidate_repr, job_repr = model(
                    candidate_title, candidate_skills, candidate_experience,
                    job_title, job_skills, job_requirement
                )
                
                # Compute similarity
                similarity = model.compute_similarity(candidate_repr, job_repr)
                
                # Store results
                all_predictions.extend(similarity.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_candidate_ids.extend(batch['candidate_id'])
                all_job_ids.extend(batch['job_id'])
                
                # Get similarity types from ground truth
                for i, cand_id in enumerate(batch['candidate_id']):
                    job_id = batch['job_id'][i]
                    pair = next(
                        (p for p in ground_truth 
                         if p['candidate_id'] == cand_id and p['job_id'] == job_id),
                        None
                    )
                    if pair:
                        all_similarity_types.append(pair.get('similarity_type', 'unknown'))
                        all_predicted_similarities.append(pair.get('predicted_similarity', 0.0))
                    else:
                        all_similarity_types.append('unknown')
                        all_predicted_similarities.append(0.0)
        
        logger.info(f"✓ Computed predictions for {len(all_predictions)} pairs")
        
        # Convert to numpy
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        
        # Compute metrics
        logger.info("\n4. Computing evaluation metrics...")
        metrics = evaluator.evaluate(
            predictions, labels, all_candidate_ids, all_job_ids
        )
        
        # Additional detailed analysis
        logger.info("\n5. Detailed Analysis...")
        
        # Analysis by similarity type
        high_mask = np.array([t == 'high' for t in all_similarity_types])
        medium_mask = np.array([t == 'medium' for t in all_similarity_types])
        random_mask = np.array([t == 'random' for t in all_similarity_types])
        
        if np.any(high_mask):
            metrics['high_similarity_mean'] = float(np.mean(predictions[high_mask]))
            metrics['high_similarity_std'] = float(np.std(predictions[high_mask]))
            metrics['high_similarity_count'] = int(np.sum(high_mask))
        
        if np.any(medium_mask):
            metrics['medium_similarity_mean'] = float(np.mean(predictions[medium_mask]))
            metrics['medium_similarity_std'] = float(np.std(predictions[medium_mask]))
            metrics['medium_similarity_count'] = int(np.sum(medium_mask))
        
        if np.any(random_mask):
            metrics['random_similarity_mean'] = float(np.mean(predictions[random_mask]))
            metrics['random_similarity_std'] = float(np.std(predictions[random_mask]))
            metrics['random_similarity_count'] = int(np.sum(random_mask))
        
        # Correlation with predicted similarity
        if len(all_predicted_similarities) > 0:
            pred_sim_array = np.array(all_predicted_similarities)
            if len(np.unique(pred_sim_array)) > 1:
                correlation = np.corrcoef(predictions, pred_sim_array)[0, 1]
                metrics['predicted_similarity_correlation'] = (
                    float(correlation) if not np.isnan(correlation) else 0.0
                )
        
        # Score distribution
        metrics['prediction_mean'] = float(np.mean(predictions))
        metrics['prediction_std'] = float(np.std(predictions))
        metrics['prediction_min'] = float(np.min(predictions))
        metrics['prediction_max'] = float(np.max(predictions))
        metrics['prediction_median'] = float(np.median(predictions))
        
        # Label distribution
        metrics['label_mean'] = float(np.mean(labels))
        metrics['label_std'] = float(np.std(labels))
        
    finally:
        db.close()
    
    return metrics, predictions, labels, all_similarity_types


def print_detailed_results(metrics: dict, predictions: np.ndarray, labels: np.ndarray, similarity_types: list):
    """Print detailed evaluation results."""
    logger.info("\n" + "=" * 80)
    logger.info("DETAILED EVALUATION RESULTS")
    logger.info("=" * 80)
    
    # Classification Metrics
    logger.info("\n📊 CLASSIFICATION METRICS")
    logger.info("-" * 80)
    logger.info(f"  Accuracy:        {metrics.get('accuracy', 0):.4f}")
    logger.info(f"  Precision:       {metrics.get('precision', 0):.4f}")
    logger.info(f"  Recall:          {metrics.get('recall', 0):.4f}")
    logger.info(f"  F1-Score:        {metrics.get('f1', 0):.4f}")
    logger.info(f"  AUC-ROC:         {metrics.get('auc_roc', 0):.4f}  {'⭐ Excellent' if metrics.get('auc_roc', 0) > 0.8 else '✓ Good' if metrics.get('auc_roc', 0) > 0.7 else '⚠ Needs Improvement'}")
    logger.info(f"  AUC-PR:          {metrics.get('auc_pr', 0):.4f}")
    
    # Ranking Metrics
    logger.info("\n📈 RANKING METRICS")
    logger.info("-" * 80)
    logger.info(f"  NDCG@10:         {metrics.get('ndcg@10', 0):.4f}  {'⭐ Excellent' if metrics.get('ndcg@10', 0) > 0.8 else '✓ Good' if metrics.get('ndcg@10', 0) > 0.6 else '⚠ Needs Improvement'}")
    logger.info(f"  MRR:             {metrics.get('mrr', 0):.4f}")
    logger.info(f"  Precision@5:     {metrics.get('precision@5', 0):.4f}")
    logger.info(f"  Precision@10:    {metrics.get('precision@10', 0):.4f}")
    logger.info(f"  Recall@5:        {metrics.get('recall@5', 0):.4f}")
    logger.info(f"  Recall@10:       {metrics.get('recall@10', 0):.4f}")
    
    # Score Distribution
    logger.info("\n📉 PREDICTION SCORE DISTRIBUTION")
    logger.info("-" * 80)
    logger.info(f"  Mean:            {metrics.get('prediction_mean', 0):.4f}")
    logger.info(f"  Std:             {metrics.get('prediction_std', 0):.4f}")
    logger.info(f"  Min:             {metrics.get('prediction_min', 0):.4f}")
    logger.info(f"  Max:             {metrics.get('prediction_max', 0):.4f}")
    logger.info(f"  Median:          {metrics.get('prediction_median', 0):.4f}")
    
    # Analysis by Similarity Type
    logger.info("\n🔍 ANALYSIS BY SIMILARITY TYPE")
    logger.info("-" * 80)
    
    if 'high_similarity_count' in metrics:
        logger.info(f"\n  HIGH Similarity Pairs ({metrics['high_similarity_count']} pairs):")
        logger.info(f"    Mean Score:    {metrics.get('high_similarity_mean', 0):.4f}")
        logger.info(f"    Std:           {metrics.get('high_similarity_std', 0):.4f}")
        logger.info(f"    Expected:     0.8-1.0 (High similarity should have high scores)")
    
    if 'medium_similarity_count' in metrics:
        logger.info(f"\n  MEDIUM Similarity Pairs ({metrics['medium_similarity_count']} pairs):")
        logger.info(f"    Mean Score:    {metrics.get('medium_similarity_mean', 0):.4f}")
        logger.info(f"    Std:           {metrics.get('medium_similarity_std', 0):.4f}")
        logger.info(f"    Expected:     0.5-0.8 (Medium similarity should have medium scores)")
    
    if 'random_similarity_count' in metrics:
        logger.info(f"\n  RANDOM/LOW Similarity Pairs ({metrics['random_similarity_count']} pairs):")
        logger.info(f"    Mean Score:    {metrics.get('random_similarity_mean', 0):.4f}")
        logger.info(f"    Std:           {metrics.get('random_similarity_std', 0):.4f}")
        logger.info(f"    Expected:     0.0-0.4 (Low similarity should have low scores)")
    
    # Correlation
    if 'predicted_similarity_correlation' in metrics:
        logger.info("\n🔗 CORRELATION ANALYSIS")
        logger.info("-" * 80)
        corr = metrics['predicted_similarity_correlation']
        logger.info(f"  Correlation with Predicted Similarity: {corr:.4f}")
        if abs(corr) > 0.7:
            logger.info(f"    {'✓ Strong correlation' if corr > 0 else '⚠ Strong negative correlation'}")
        elif abs(corr) > 0.4:
            logger.info(f"    ✓ Moderate correlation")
        else:
            logger.info(f"    ⚠ Weak correlation")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📋 SUMMARY")
    logger.info("=" * 80)
    
    auc_roc = metrics.get('auc_roc', 0)
    ndcg = metrics.get('ndcg@10', 0)
    
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
    logger.info(f"  - Precision@10: {metrics.get('precision@10', 0):.4f} (Higher is better)")
    logger.info(f"  - Recall@10: {metrics.get('recall@10', 0):.4f} (Higher is better)")
    
    logger.info("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Detailed Evaluation of Two-Tower Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with CSV ground truth
  python scripts/evaluate_two_tower_detailed.py \\
      --model-path outputs_improved/best_model_improved.pt \\
      --ground-truth-csv ground_truth_500_pairs.csv
  
  # Evaluate with JSON ground truth
  python scripts/evaluate_two_tower_detailed.py \\
      --model-path models/two_tower/best_model.pt \\
      --ground-truth-json data/ground_truth.json
        """
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained Two-Tower model (.pt file)'
    )
    parser.add_argument(
        '--ground-truth-csv',
        type=str,
        help='Path to ground truth CSV file'
    )
    parser.add_argument(
        '--ground-truth-json',
        type=str,
        help='Path to ground truth JSON file'
    )
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=768,
        help='Embedding dimension (default: 768)'
    )
    parser.add_argument(
        '--hidden-dims',
        type=int,
        nargs='+',
        default=[512, 256],
        help='Hidden layer dimensions (default: [512, 256])'
    )
    parser.add_argument(
        '--output-dim',
        type=int,
        default=256,
        help='Output representation dimension (default: 256)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation (default: 32)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use (default: cpu)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        help='Save detailed results to JSON file'
    )
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load ground truth
    if args.ground_truth_csv:
        ground_truth = load_ground_truth_from_csv(args.ground_truth_csv)
    elif args.ground_truth_json:
        logger.info(f"Loading ground truth from JSON: {args.ground_truth_json}")
        db = SessionLocal()
        builder = GroundTruthBuilder(db)
        ground_truth = builder.load_ground_truth(args.ground_truth_json)
        db.close()
    else:
        # Try default CSV
        default_csv = 'ground_truth_500_pairs.csv'
        if os.path.exists(default_csv):
            logger.info(f"Using default ground truth CSV: {default_csv}")
            ground_truth = load_ground_truth_from_csv(default_csv)
        else:
            logger.error("Please provide --ground-truth-csv or --ground-truth-json")
            return
    
    # Evaluate
    try:
        metrics, predictions, labels, similarity_types = evaluate_two_tower_detailed(
            model_path=args.model_path,
            ground_truth=ground_truth,
            embedding_dim=args.embedding_dim,
            hidden_dims=args.hidden_dims,
            output_dim=args.output_dim,
            batch_size=args.batch_size,
            device=args.device
        )
        
        # Print results
        print_detailed_results(metrics, predictions, labels, similarity_types)
        
        # Save to file if requested
        if args.output_file:
            output_data = {
                'metrics': metrics,
                'model_path': args.model_path,
                'num_samples': len(predictions),
                'config': {
                    'embedding_dim': args.embedding_dim,
                    'hidden_dims': args.hidden_dims,
                    'output_dim': args.output_dim,
                    'batch_size': args.batch_size,
                    'device': args.device
                }
            }
            with open(args.output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"\n✓ Detailed results saved to: {args.output_file}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

