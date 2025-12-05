"""Utility functions."""
import torch
import numpy as np
from typing import List, Tuple
import pickle
from pathlib import Path


def set_seed(seed: int = 42):
    """Set random seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_embeddings(embeddings: np.ndarray, ids: List[str], path: Path):
    """Save embeddings with IDs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'ids': ids}, f)


def load_embeddings(path: Path) -> Tuple[np.ndarray, List[str]]:
    """Load embeddings with IDs."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data['ids']


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embeddings / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity."""
    a_norm = normalize_embeddings(a)
    b_norm = normalize_embeddings(b)
    return np.dot(a_norm, b_norm.T)


