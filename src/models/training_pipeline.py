"""Training Pipeline for Two-Tower Model."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from tqdm import tqdm
import os
import json

from src.models.two_tower_model import TwoTowerModel
from src.models.evaluation_metrics import TwoTowerEvaluator
from src.database.multi_field_repository import MultiFieldEmbeddingRepository
from src.database.connection import SessionLocal

logger = logging.getLogger(__name__)


def collate_fn(batch):
    """Custom collate function for batching."""
    candidate_title = torch.stack([torch.tensor(item['candidate_title']) for item in batch])
    candidate_skills = torch.stack([torch.tensor(item['candidate_skills']) for item in batch])
    candidate_experience = torch.stack([torch.tensor(item['candidate_experience']) for item in batch])
    job_title = torch.stack([torch.tensor(item['job_title']) for item in batch])
    job_skills = torch.stack([torch.tensor(item['job_skills']) for item in batch])
    job_requirement = torch.stack([torch.tensor(item['job_requirement']) for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32)
    candidate_ids = [item['candidate_id'] for item in batch]
    job_ids = [item['job_id'] for item in batch]
    
    return {
        'candidate_title': candidate_title,
        'candidate_skills': candidate_skills,
        'candidate_experience': candidate_experience,
        'job_title': job_title,
        'job_skills': job_skills,
        'job_requirement': job_requirement,
        'label': labels,
        'candidate_id': candidate_ids,
        'job_id': job_ids
    }


class GroundTruthDataset(Dataset):
    """Dataset for ground truth pairs."""
    
    def __init__(
        self,
        ground_truth: List[Dict],
        repository: MultiFieldEmbeddingRepository
    ):
        """
        Initialize dataset.
        
        Args:
            ground_truth: List of ground truth pairs
            repository: Repository to fetch embeddings
        """
        self.ground_truth = ground_truth
        self.repository = repository
        
        # Pre-load all embeddings
        logger.info("Pre-loading embeddings...")
        self.candidate_embeddings = {}
        self.job_embeddings = {}
        
        candidate_ids = set(p['candidate_id'] for p in ground_truth)
        job_ids = set(p['job_id'] for p in ground_truth)
        
        for cand_id in tqdm(candidate_ids, desc="Loading candidate embeddings"):
            candidate = repository.get_candidate_multi_embedding(cand_id)
            if candidate:
                self.candidate_embeddings[cand_id] = {
                    'title': np.array(candidate.title_embedding, dtype=np.float32),
                    'skills': np.array(candidate.skills_embedding, dtype=np.float32),
                    'experience': np.array(candidate.experience_embedding, dtype=np.float32)
                }
        
        for job_id in tqdm(job_ids, desc="Loading job embeddings"):
            job = repository.get_job_multi_embedding(job_id)
            if job:
                self.job_embeddings[job_id] = {
                    'title': np.array(job.title_embedding, dtype=np.float32),
                    'skills': np.array(job.skills_embedding, dtype=np.float32),
                    'requirement': np.array(job.requirement_embedding, dtype=np.float32)
                }
        
        logger.info(f"Loaded {len(self.candidate_embeddings)} candidate embeddings")
        logger.info(f"Loaded {len(self.job_embeddings)} job embeddings")
    
    def __len__(self):
        return len(self.ground_truth)
    
    def __getitem__(self, idx):
        pair = self.ground_truth[idx]
        
        cand_id = pair['candidate_id']
        job_id = pair['job_id']
        label = pair['label']
        
        # Get embeddings
        cand_emb = self.candidate_embeddings.get(cand_id)
        job_emb = self.job_embeddings.get(job_id)
        
        if cand_emb is None or job_emb is None:
            # Return zeros if missing
            embedding_dim = 768  # Default
            return {
                'candidate_title': np.zeros(embedding_dim, dtype=np.float32),
                'candidate_skills': np.zeros(embedding_dim, dtype=np.float32),
                'candidate_experience': np.zeros(embedding_dim, dtype=np.float32),
                'job_title': np.zeros(embedding_dim, dtype=np.float32),
                'job_skills': np.zeros(embedding_dim, dtype=np.float32),
                'job_requirement': np.zeros(embedding_dim, dtype=np.float32),
                'label': float(label),
                'candidate_id': cand_id,
                'job_id': job_id
            }
        
        return {
            'candidate_title': cand_emb['title'],
            'candidate_skills': cand_emb['skills'],
            'candidate_experience': cand_emb['experience'],
            'job_title': job_emb['title'],
            'job_skills': job_emb['skills'],
            'job_requirement': job_emb['requirement'],
            'label': float(label),
            'candidate_id': cand_id,
            'job_id': job_id
        }


class TwoTowerTrainer:
    """Trainer for Two-Tower model."""
    
    def __init__(
        self,
        model: TwoTowerModel,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001
    ):
        """
        Initialize trainer.
        
        Args:
            model: Two-Tower model
            device: Device to train on ('cpu' or 'cuda')
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.evaluator = TwoTowerEvaluator()
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader for training data
            epoch: Current epoch number
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            # Move to device
            candidate_title = batch['candidate_title'].to(self.device)
            candidate_skills = batch['candidate_skills'].to(self.device)
            candidate_experience = batch['candidate_experience'].to(self.device)
            job_title = batch['job_title'].to(self.device)
            job_skills = batch['job_skills'].to(self.device)
            job_requirement = batch['job_requirement'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            candidate_repr, job_repr = self.model(
                candidate_title, candidate_skills, candidate_experience,
                job_title, job_skills, job_requirement
            )
            
            # Compute similarity
            similarity = self.model.compute_similarity(candidate_repr, job_repr)
            
            # Compute loss (using BCE with logits)
            loss = self.criterion(similarity, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * len(labels)
            total_samples += len(labels)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        return {'loss': avg_loss}
    
    def evaluate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate model on validation/test set.
        
        Args:
            dataloader: DataLoader for evaluation data
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_candidate_ids = []
        all_job_ids = []
        all_field_similarities = {
            'title': [],
            'skills': [],
            'experience': []
        }
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                candidate_title = batch['candidate_title'].to(self.device)
                candidate_skills = batch['candidate_skills'].to(self.device)
                candidate_experience = batch['candidate_experience'].to(self.device)
                job_title = batch['job_title'].to(self.device)
                job_skills = batch['job_skills'].to(self.device)
                job_requirement = batch['job_requirement'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                candidate_repr, job_repr = self.model(
                    candidate_title, candidate_skills, candidate_experience,
                    job_title, job_skills, job_requirement
                )
                
                # Compute similarity
                similarity = self.model.compute_similarity(candidate_repr, job_repr)
                
                # Store results
                all_predictions.extend(similarity.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_candidate_ids.extend(batch['candidate_id'])
                all_job_ids.extend(batch['job_id'])
        
        # Convert to numpy
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        
        # Compute metrics
        metrics = self.evaluator.evaluate(
            predictions, labels, all_candidate_ids, all_job_ids
        )
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        save_dir: Optional[str] = None,
        save_best: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train model for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            num_epochs: Number of epochs to train
            save_dir: Directory to save model checkpoints
            save_best: Whether to save best model based on validation metrics
        
        Returns:
            Dictionary of training history
        """
        history = {
            'train_loss': [],
            'val_auc_roc': [],
            'val_ndcg@10': []
        }
        
        best_val_auc = 0.0
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{num_epochs}")
            logger.info("-" * 60)
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            history['train_loss'].append(train_metrics['loss'])
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            
            # Validate
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                history['val_auc_roc'].append(val_metrics.get('auc_roc', 0.0))
                history['val_ndcg@10'].append(val_metrics.get('ndcg@10', 0.0))
                
                logger.info(f"Val AUC-ROC: {val_metrics.get('auc_roc', 0):.4f}")
                logger.info(f"Val NDCG@10: {val_metrics.get('ndcg@10', 0):.4f}")
                
                # Save best model
                if save_best and save_dir:
                    current_auc = val_metrics.get('auc_roc', 0.0)
                    if current_auc > best_val_auc:
                        best_val_auc = current_auc
                        self.save_model(os.path.join(save_dir, 'best_model.pt'))
                        logger.info(f"Saved best model (AUC-ROC: {best_val_auc:.4f})")
            
            # Save checkpoint
            if save_dir:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
                self.save_model(checkpoint_path)
        
        return history
    
    def save_model(self, path: str):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f"Saved model to {path}")
    
    def load_model(self, path: str):
        """
        Load model checkpoint.
        
        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded model from {path}")

