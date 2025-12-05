"""Two-Tower model architecture."""
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from typing import Optional, List


class Tower(nn.Module):
    """Single tower encoder."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.backbone = SentenceTransformer(model_name)
        backbone_dim = self.backbone.get_sentence_embedding_dimension()
        
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        self.output_dim = output_dim
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to embeddings."""
        with torch.no_grad():
            embeddings = self.backbone.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        embeddings = embeddings.clone().detach().requires_grad_(True)
        embeddings = self.projection(embeddings)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


class TwoTowerModel(nn.Module):
    """Two-Tower retrieval model."""
    
    def __init__(
        self,
        candidate_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        job_model_name: Optional[str] = None,
        output_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        job_model_name = job_model_name or candidate_model_name
        
        self.candidate_tower = Tower(candidate_model_name, output_dim, dropout)
        self.job_tower = Tower(job_model_name, output_dim, dropout)
        self.output_dim = output_dim
    
    def encode_candidates(self, texts: List[str]) -> torch.Tensor:
        """Encode candidate texts."""
        return self.candidate_tower(texts)
    
    def encode_jobs(self, texts: List[str]) -> torch.Tensor:
        """Encode job texts."""
        return self.job_tower(texts)
    
    def forward(
        self,
        candidate_texts: List[str],
        job_texts: List[str]
    ) -> torch.Tensor:
        """Compute similarity matrix."""
        candidate_emb = self.encode_candidates(candidate_texts)
        job_emb = self.encode_jobs(job_texts)
        similarity = torch.matmul(candidate_emb, job_emb.T)
        return similarity
    
    def encode_candidate_onnx(self, text: str) -> torch.Tensor:
        """Encode single candidate for ONNX export."""
        return self.encode_candidates([text])[0]
    
    def encode_job_onnx(self, text: str) -> torch.Tensor:
        """Encode single job for ONNX export."""
        return self.encode_jobs([text])[0]

