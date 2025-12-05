"""Inference with FAISS index."""
import torch
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict
import pickle

from two_tower.model import TwoTowerModel
from two_tower.utils import load_embeddings, save_embeddings, normalize_embeddings


class JobRecommender:
    """Job recommender with FAISS index."""
    
    def __init__(
        self,
        model_path: str,
        job_embeddings_path: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dim: int = 256,
        index_type: str = "HNSW",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        
        model = TwoTowerModel(
            candidate_model_name=model_name,
            output_dim=output_dim
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        self.model = model.to(device)
        
        job_embeddings, job_ids = load_embeddings(Path(job_embeddings_path))
        self.job_embeddings = normalize_embeddings(job_embeddings.astype(np.float32))
        self.job_ids = job_ids
        
        self.index = self._build_index(index_type)
        self.index.add(self.job_embeddings)
    
    def _build_index(self, index_type: str) -> faiss.Index:
        """Build FAISS index."""
        dim = self.job_embeddings.shape[1]
        
        if index_type == "HNSW":
            index = faiss.IndexHNSWFlat(dim, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 128
        elif index_type == "Flat":
            index = faiss.IndexFlatIP(dim)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        return index
    
    def recommend(
        self,
        candidate_text: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Recommend jobs for candidate.
        
        Args:
            candidate_text: Candidate text
            top_k: Number of recommendations
        
        Returns:
            List of job results with id and score
        """
        with torch.no_grad():
            candidate_emb = self.model.encode_candidates([candidate_text])
            candidate_emb = candidate_emb.cpu().numpy()
            candidate_emb = normalize_embeddings(candidate_emb)
        
        self.index.hnsw.efSearch = 128
        
        distances, indices = self.index.search(candidate_emb, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.job_ids):
                score = float(dist)
                results.append({
                    'job_id': self.job_ids[idx],
                    'score': score
                })
        
        return results


def build_job_index(
    model_path: str,
    job_texts: List[str],
    job_ids: List[str],
    output_path: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_dim: int = 256,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Build and save job embeddings index.
    
    Args:
        model_path: Path to trained model
        job_texts: List of job texts
        job_ids: List of job IDs
        output_path: Path to save embeddings
        model_name: Backbone model name
        output_dim: Output dimension
        batch_size: Batch size for encoding
        device: Device
    """
    device = torch.device(device)
    
    model = TwoTowerModel(
        candidate_model_name=model_name,
        output_dim=output_dim
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model = model.to(device)
    
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(job_texts), batch_size):
            batch_texts = job_texts[i:i + batch_size]
            batch_emb = model.encode_jobs(batch_texts)
            embeddings.append(batch_emb.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    
    save_embeddings(embeddings, job_ids, Path(output_path))
    
    print(f"Saved {len(job_ids)} job embeddings to {output_path}")

