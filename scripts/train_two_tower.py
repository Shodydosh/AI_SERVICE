"""Script to train Two-Tower model with ground truth dataset."""
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
from src.models.ground_truth_builder import GroundTruthBuilder
from src.models.two_tower_model import TwoTowerModel
from src.models.training_pipeline import TwoTowerTrainer, GroundTruthDataset
from src.models.evaluation_metrics import TwoTowerEvaluator
from src.database.multi_field_repository import MultiFieldEmbeddingRepository

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_ground_truth(
    db: Session,
    output_path: str,
    max_candidates: int = 500,
    max_jobs: int = 2000
):
    """Build ground truth dataset."""
    logger.info("Building ground truth dataset...")
    
    builder = GroundTruthBuilder(
        db=db,
        title_similarity_threshold=0.6,
        skills_similarity_threshold=0.5,
        experience_similarity_threshold=0.5,
        combined_threshold=0.55
    )
    
    ground_truth = builder.build_ground_truth_dataset(
        max_candidates=max_candidates,
        max_jobs=max_jobs,
        min_positive_pairs=100,
        min_negative_pairs=200
    )
    
    builder.save_ground_truth(ground_truth, output_path)
    
    return ground_truth


def train_model(
    ground_truth_path: str,
    output_dir: str,
    embedding_dim: int = 768,
    hidden_dims: list = [512, 256],
    output_dim: int = 256,
    batch_size: int = 32,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    train_ratio: float = 0.8,
    device: str = 'cpu'
):
    """Train Two-Tower model."""
    logger.info("Initializing training...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ground truth
    from src.models.ground_truth_builder import GroundTruthBuilder
    db = SessionLocal()
    builder = GroundTruthBuilder(db)
    ground_truth = builder.load_ground_truth(ground_truth_path)
    
    # Split train/val
    np.random.seed(42)
    np.random.shuffle(ground_truth)
    split_idx = int(len(ground_truth) * train_ratio)
    train_data = ground_truth[:split_idx]
    val_data = ground_truth[split_idx:]
    
    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Val samples: {len(val_data)}")
    
    # Create datasets
    repository = MultiFieldEmbeddingRepository(db)
    train_dataset = GroundTruthDataset(train_data, repository)
    val_dataset = GroundTruthDataset(val_data, repository)
    
    # Import collate function
    from src.models.training_pipeline import collate_fn
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Create model
    model = TwoTowerModel(
        embedding_dim=embedding_dim,
        candidate_hidden_dims=hidden_dims,
        job_hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout=0.1,
        use_batch_norm=True
    )
    
    # Create trainer
    trainer = TwoTowerTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=0.0001
    )
    
    # Train
    logger.info("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        save_dir=output_dir,
        save_best=True
    )
    
    # Final evaluation
    logger.info("\nFinal Evaluation:")
    logger.info("=" * 60)
    val_metrics = trainer.evaluate(val_loader)
    evaluator = TwoTowerEvaluator()
    evaluator.print_metrics(val_metrics)
    
    # Save history
    import json
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"\nSaved training history to {history_path}")
    
    db.close()
    
    return trainer, val_metrics


def main():
    parser = argparse.ArgumentParser(description='Train Two-Tower Model')
    parser.add_argument(
        '--ground-truth-path',
        type=str,
        default='data/ground_truth.json',
        help='Path to ground truth JSON file'
    )
    parser.add_argument(
        '--build-ground-truth',
        action='store_true',
        help='Build ground truth dataset before training'
    )
    parser.add_argument(
        '--max-candidates',
        type=int,
        default=500,
        help='Maximum number of candidates for ground truth'
    )
    parser.add_argument(
        '--max-jobs',
        type=int,
        default=2000,
        help='Maximum number of jobs for ground truth'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/two_tower',
        help='Output directory for model checkpoints'
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
        '--num-epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to train on'
    )
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    db = SessionLocal()
    
    try:
        # Build ground truth if requested
        if args.build_ground_truth:
            logger.info("Building ground truth dataset...")
            build_ground_truth(
                db=db,
                output_path=args.ground_truth_path,
                max_candidates=args.max_candidates,
                max_jobs=args.max_jobs
            )
        
        # Train model
        trainer, metrics = train_model(
            ground_truth_path=args.ground_truth_path,
            output_dir=args.output_dir,
            embedding_dim=args.embedding_dim,
            hidden_dims=args.hidden_dims,
            output_dim=args.output_dim,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            device=args.device
        )
        
        logger.info("\nTraining completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise
    finally:
        db.close()


if __name__ == '__main__':
    main()

