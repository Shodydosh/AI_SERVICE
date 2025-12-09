"""Evaluation Metrics for Two-Tower Model."""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import logging

logger = logging.getLogger(__name__)


class TwoTowerEvaluator:
    """Evaluator for Two-Tower model performance."""
    
    def __init__(self):
        """Initialize evaluator."""
        pass
    
    def compute_ranking_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        candidate_ids: List[str],
        job_ids: List[str]
    ) -> Dict[str, float]:
        """
        Compute ranking metrics (NDCG, MRR, Precision@K, Recall@K).
        
        Args:
            predictions: Predicted similarity scores
            labels: Ground truth labels
            candidate_ids: List of candidate IDs
            job_ids: List of job IDs
        
        Returns:
            Dictionary of ranking metrics
        """
        metrics = {}
        
        # Group by candidate
        candidate_to_pairs = {}
        for i, cand_id in enumerate(candidate_ids):
            if cand_id not in candidate_to_pairs:
                candidate_to_pairs[cand_id] = []
            candidate_to_pairs[cand_id].append({
                'job_id': job_ids[i],
                'prediction': predictions[i],
                'label': labels[i]
            })
        
        # Compute metrics per candidate
        ndcg_scores = []
        mrr_scores = []
        precision_at_k_scores = {5: [], 10: [], 20: []}
        recall_at_k_scores = {5: [], 10: [], 20: []}
        
        for cand_id, pairs in candidate_to_pairs.items():
            # Sort by prediction score
            pairs.sort(key=lambda x: x['prediction'], reverse=True)
            
            # Extract labels
            sorted_labels = np.array([p['label'] for p in pairs])
            sorted_predictions = np.array([p['prediction'] for p in pairs])
            
            # NDCG@10
            ndcg = self._compute_ndcg(sorted_labels, k=10)
            if ndcg is not None:
                ndcg_scores.append(ndcg)
            
            # MRR
            mrr = self._compute_mrr(sorted_labels)
            if mrr is not None:
                mrr_scores.append(mrr)
            
            # Precision@K and Recall@K
            for k in [5, 10, 20]:
                if len(sorted_labels) >= k:
                    precision_k = precision_score(
                        sorted_labels[:k],
                        (sorted_predictions[:k] > 0.5).astype(int),
                        zero_division=0
                    )
                    recall_k = recall_score(
                        sorted_labels[:k],
                        (sorted_predictions[:k] > 0.5).astype(int),
                        zero_division=0
                    )
                    precision_at_k_scores[k].append(precision_k)
                    recall_at_k_scores[k].append(recall_k)
        
        # Average metrics
        metrics['ndcg@10'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
        metrics['mrr'] = np.mean(mrr_scores) if mrr_scores else 0.0
        
        for k in [5, 10, 20]:
            metrics[f'precision@{k}'] = (
                np.mean(precision_at_k_scores[k]) if precision_at_k_scores[k] else 0.0
            )
            metrics[f'recall@{k}'] = (
                np.mean(recall_at_k_scores[k]) if recall_at_k_scores[k] else 0.0
            )
        
        return metrics
    
    def _compute_ndcg(self, labels: np.ndarray, k: int = 10) -> Optional[float]:
        """
        Compute Normalized Discounted Cumulative Gain.
        
        Args:
            labels: Sorted relevance labels
            k: Top K items to consider
        
        Returns:
            NDCG@K score
        """
        if len(labels) == 0:
            return None
        
        k = min(k, len(labels))
        dcg = 0.0
        
        for i in range(k):
            rel = labels[i]
            dcg += rel / np.log2(i + 2)
        
        # Ideal DCG
        ideal_labels = np.sort(labels)[::-1]
        idcg = 0.0
        for i in range(k):
            rel = ideal_labels[i]
            idcg += rel / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def _compute_mrr(self, labels: np.ndarray) -> Optional[float]:
        """
        Compute Mean Reciprocal Rank.
        
        Args:
            labels: Sorted relevance labels
        
        Returns:
            MRR score
        """
        if len(labels) == 0:
            return None
        
        for i, label in enumerate(labels):
            if label == 1:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def compute_classification_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            predictions: Predicted similarity scores
            labels: Ground truth labels
            threshold: Classification threshold
        
        Returns:
            Dictionary of classification metrics
        """
        binary_predictions = (predictions >= threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(labels, binary_predictions),
            'precision': precision_score(labels, binary_predictions, zero_division=0),
            'recall': recall_score(labels, binary_predictions, zero_division=0),
            'f1': f1_score(labels, binary_predictions, zero_division=0)
        }
        
        # AUC-ROC
        try:
            metrics['auc_roc'] = roc_auc_score(labels, predictions)
        except ValueError:
            metrics['auc_roc'] = 0.0
        
        # AUC-PR
        try:
            metrics['auc_pr'] = average_precision_score(labels, predictions)
        except ValueError:
            metrics['auc_pr'] = 0.0
        
        return metrics
    
    def compute_similarity_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        field_similarities: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        Compute similarity correlation metrics.
        
        Args:
            predictions: Predicted similarity scores
            labels: Ground truth labels
            field_similarities: Optional field-wise similarities
        
        Returns:
            Dictionary of similarity metrics
        """
        metrics = {}
        
        # Correlation with labels
        if len(np.unique(labels)) > 1:
            correlation = np.corrcoef(predictions, labels)[0, 1]
            metrics['label_correlation'] = correlation if not np.isnan(correlation) else 0.0
        else:
            metrics['label_correlation'] = 0.0
        
        # Field correlations if provided
        if field_similarities:
            for field_name, field_sim in field_similarities.items():
                if len(field_sim) == len(predictions):
                    corr = np.corrcoef(predictions, field_sim)[0, 1]
                    metrics[f'{field_name}_correlation'] = (
                        corr if not np.isnan(corr) else 0.0
                    )
        
        return metrics
    
    def evaluate(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        candidate_ids: List[str],
        job_ids: List[str],
        field_similarities: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of model performance.
        
        Args:
            predictions: Predicted similarity scores
            labels: Ground truth labels
            candidate_ids: List of candidate IDs
            job_ids: List of job IDs
            field_similarities: Optional field-wise similarities
        
        Returns:
            Dictionary of all metrics
        """
        all_metrics = {}
        
        # Classification metrics
        classification_metrics = self.compute_classification_metrics(predictions, labels)
        all_metrics.update(classification_metrics)
        
        # Ranking metrics
        ranking_metrics = self.compute_ranking_metrics(
            predictions, labels, candidate_ids, job_ids
        )
        all_metrics.update(ranking_metrics)
        
        # Similarity metrics
        similarity_metrics = self.compute_similarity_metrics(
            predictions, labels, field_similarities
        )
        all_metrics.update(similarity_metrics)
        
        return all_metrics
    
    def print_metrics(self, metrics: Dict[str, float]):
        """
        Print metrics in a formatted way.
        
        Args:
            metrics: Dictionary of metrics
        """
        logger.info("=" * 60)
        logger.info("EVALUATION METRICS")
        logger.info("=" * 60)
        
        logger.info("\nClassification Metrics:")
        logger.info(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
        logger.info(f"  Precision: {metrics.get('precision', 0):.4f}")
        logger.info(f"  Recall:    {metrics.get('recall', 0):.4f}")
        logger.info(f"  F1-Score:  {metrics.get('f1', 0):.4f}")
        logger.info(f"  AUC-ROC:   {metrics.get('auc_roc', 0):.4f}")
        logger.info(f"  AUC-PR:    {metrics.get('auc_pr', 0):.4f}")
        
        logger.info("\nRanking Metrics:")
        logger.info(f"  NDCG@10:     {metrics.get('ndcg@10', 0):.4f}")
        logger.info(f"  MRR:         {metrics.get('mrr', 0):.4f}")
        logger.info(f"  Precision@5: {metrics.get('precision@5', 0):.4f}")
        logger.info(f"  Precision@10:{metrics.get('precision@10', 0):.4f}")
        logger.info(f"  Recall@5:    {metrics.get('recall@5', 0):.4f}")
        logger.info(f"  Recall@10:   {metrics.get('recall@10', 0):.4f}")
        
        logger.info("\nSimilarity Metrics:")
        logger.info(f"  Label Correlation: {metrics.get('label_correlation', 0):.4f}")
        if 'title_similarity_correlation' in metrics:
            logger.info(f"  Title Correlation:  {metrics.get('title_similarity_correlation', 0):.4f}")
        if 'skills_similarity_correlation' in metrics:
            logger.info(f"  Skills Correlation: {metrics.get('skills_similarity_correlation', 0):.4f}")
        if 'experience_similarity_correlation' in metrics:
            logger.info(f"  Exp Correlation:    {metrics.get('experience_similarity_correlation', 0):.4f}")
        
        logger.info("=" * 60)


