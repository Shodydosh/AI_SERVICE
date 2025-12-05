"""Evaluation metrics."""
import torch
import numpy as np
from typing import List, Tuple


def recall_at_k(
    scores: np.ndarray,
    labels: np.ndarray,
    k: int
) -> float:
    """
    Compute recall@k.
    
    Args:
        scores: [num_candidates, num_jobs] similarity scores
        labels: [num_candidates, num_jobs] binary labels
        k: top k
    
    Returns:
        recall@k
    """
    num_candidates = scores.shape[0]
    recalls = []
    
    for i in range(num_candidates):
        top_k_indices = np.argsort(scores[i])[-k:][::-1]
        relevant = np.sum(labels[i][top_k_indices])
        total_relevant = np.sum(labels[i])
        
        if total_relevant > 0:
            recalls.append(relevant / total_relevant)
    
    return np.mean(recalls) if recalls else 0.0


def precision_at_k(
    scores: np.ndarray,
    labels: np.ndarray,
    k: int
) -> float:
    """Compute precision@k."""
    num_candidates = scores.shape[0]
    precisions = []
    
    for i in range(num_candidates):
        top_k_indices = np.argsort(scores[i])[-k:][::-1]
        relevant = np.sum(labels[i][top_k_indices])
        precisions.append(relevant / k)
    
    return np.mean(precisions) if precisions else 0.0


def mrr(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute Mean Reciprocal Rank."""
    num_candidates = scores.shape[0]
    reciprocal_ranks = []
    
    for i in range(num_candidates):
        sorted_indices = np.argsort(scores[i])[::-1]
        for rank, idx in enumerate(sorted_indices, 1):
            if labels[i][idx] > 0:
                reciprocal_ranks.append(1.0 / rank)
                break
    
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def ndcg_at_k(
    scores: np.ndarray,
    labels: np.ndarray,
    k: int
) -> float:
    """Compute NDCG@k."""
    num_candidates = scores.shape[0]
    ndcgs = []
    
    for i in range(num_candidates):
        top_k_indices = np.argsort(scores[i])[-k:][::-1]
        top_k_labels = labels[i][top_k_indices]
        
        dcg = np.sum(top_k_labels / np.log2(np.arange(2, len(top_k_labels) + 2)))
        
        ideal_labels = np.sort(labels[i])[-k:][::-1]
        idcg = np.sum(ideal_labels / np.log2(np.arange(2, len(ideal_labels) + 2)))
        
        if idcg > 0:
            ndcgs.append(dcg / idcg)
    
    return np.mean(ndcgs) if ndcgs else 0.0


def hit_at_k(
    scores: np.ndarray,
    labels: np.ndarray,
    k: int
) -> float:
    """Compute hit rate@k."""
    num_candidates = scores.shape[0]
    hits = 0
    
    for i in range(num_candidates):
        top_k_indices = np.argsort(scores[i])[-k:][::-1]
        if np.any(labels[i][top_k_indices] > 0):
            hits += 1
    
    return hits / num_candidates if num_candidates > 0 else 0.0


def evaluate(
    model,
    candidate_texts: List[str],
    job_texts: List[str],
    positive_pairs: List[Tuple[int, int]],
    k_values: List[int] = [1, 5, 10]
) -> dict:
    """
    Evaluate model.
    
    Args:
        model: TwoTowerModel
        candidate_texts: List of candidate texts
        job_texts: List of job texts
        positive_pairs: List of (candidate_idx, job_idx) pairs
        k_values: List of k values for metrics
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    with torch.no_grad():
        candidate_emb = model.encode_candidates(candidate_texts)
        job_emb = model.encode_jobs(job_texts)
        
        scores = torch.matmul(candidate_emb, job_emb.T).cpu().numpy()
    
    labels = np.zeros((len(candidate_texts), len(job_texts)))
    for c_idx, j_idx in positive_pairs:
        labels[c_idx, j_idx] = 1
    
    results = {}
    
    for k in k_values:
        results[f'recall@{k}'] = recall_at_k(scores, labels, k)
        results[f'precision@{k}'] = precision_at_k(scores, labels, k)
        results[f'ndcg@{k}'] = ndcg_at_k(scores, labels, k)
        results[f'hit@{k}'] = hit_at_k(scores, labels, k)
    
    results['mrr'] = mrr(scores, labels)
    
    return results

