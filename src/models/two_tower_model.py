"""Two-Tower Neural Network Architecture for Multi-Field Embeddings."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CandidateTower(nn.Module):
    """
    Candidate Tower: Encodes candidate features (title, skills, experience)
    into a unified representation.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dims: List[int] = [512, 256],
        output_dim: int = 256,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        """
        Initialize Candidate Tower.
        
        Args:
            embedding_dim: Dimension of input embeddings (from pre-trained model)
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of final output representation
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super(CandidateTower, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        
        # Input layer: concatenate 3 embeddings
        input_dim = embedding_dim * 3  # title + skills + experience
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        title_emb: torch.Tensor,
        skills_emb: torch.Tensor,
        experience_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through candidate tower.
        
        Args:
            title_emb: Title embedding [batch_size, embedding_dim]
            skills_emb: Skills embedding [batch_size, embedding_dim]
            experience_emb: Experience embedding [batch_size, embedding_dim]
        
        Returns:
            Candidate representation [batch_size, output_dim]
        """
        # Concatenate three embeddings
        combined = torch.cat([title_emb, skills_emb, experience_emb], dim=1)
        
        # Pass through network
        output = self.network(combined)
        
        # L2 normalize output
        output = F.normalize(output, p=2, dim=1)
        
        return output


class JobTower(nn.Module):
    """
    Job Tower: Encodes job features (title, skills, requirements)
    into a unified representation.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dims: List[int] = [512, 256],
        output_dim: int = 256,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        """
        Initialize Job Tower.
        
        Args:
            embedding_dim: Dimension of input embeddings (from pre-trained model)
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of final output representation
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super(JobTower, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        
        # Input layer: concatenate 3 embeddings
        input_dim = embedding_dim * 3  # title + skills + requirement
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        title_emb: torch.Tensor,
        skills_emb: torch.Tensor,
        requirement_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through job tower.
        
        Args:
            title_emb: Title embedding [batch_size, embedding_dim]
            skills_emb: Skills embedding [batch_size, embedding_dim]
            requirement_emb: Requirement embedding [batch_size, embedding_dim]
        
        Returns:
            Job representation [batch_size, output_dim]
        """
        # Concatenate three embeddings
        combined = torch.cat([title_emb, skills_emb, requirement_emb], dim=1)
        
        # Pass through network
        output = self.network(combined)
        
        # L2 normalize output
        output = F.normalize(output, p=2, dim=1)
        
        return output


class TwoTowerModel(nn.Module):
    """
    Two-Tower Model: Combines Candidate Tower and Job Tower
    for learning optimal representations for matching.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        candidate_hidden_dims: List[int] = [512, 256],
        job_hidden_dims: List[int] = [512, 256],
        output_dim: int = 256,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        """
        Initialize Two-Tower Model.
        
        Args:
            embedding_dim: Dimension of input embeddings
            candidate_hidden_dims: Hidden dimensions for candidate tower
            job_hidden_dims: Hidden dimensions for job tower
            output_dim: Dimension of output representations
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super(TwoTowerModel, self).__init__()
        
        self.candidate_tower = CandidateTower(
            embedding_dim=embedding_dim,
            hidden_dims=candidate_hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )
        
        self.job_tower = JobTower(
            embedding_dim=embedding_dim,
            hidden_dims=job_hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )
        
        self.output_dim = output_dim
    
    def forward(
        self,
        candidate_title_emb: torch.Tensor,
        candidate_skills_emb: torch.Tensor,
        candidate_experience_emb: torch.Tensor,
        job_title_emb: torch.Tensor,
        job_skills_emb: torch.Tensor,
        job_requirement_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both towers.
        
        Args:
            candidate_title_emb: Candidate title embedding
            candidate_skills_emb: Candidate skills embedding
            candidate_experience_emb: Candidate experience embedding
            job_title_emb: Job title embedding
            job_skills_emb: Job skills embedding
            job_requirement_emb: Job requirement embedding
        
        Returns:
            Tuple of (candidate_repr, job_repr)
        """
        candidate_repr = self.candidate_tower(
            candidate_title_emb,
            candidate_skills_emb,
            candidate_experience_emb
        )
        
        job_repr = self.job_tower(
            job_title_emb,
            job_skills_emb,
            job_requirement_emb
        )
        
        return candidate_repr, job_repr
    
    def compute_similarity(
        self,
        candidate_repr: torch.Tensor,
        job_repr: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity score between candidate and job representations.
        
        Args:
            candidate_repr: Candidate representation [batch_size, output_dim]
            job_repr: Job representation [batch_size, output_dim]
        
        Returns:
            Similarity scores [batch_size]
        """
        # Dot product (since both are L2 normalized, this is cosine similarity)
        similarity = torch.sum(candidate_repr * job_repr, dim=1)
        return similarity
    
    def encode_candidate(
        self,
        title_emb: torch.Tensor,
        skills_emb: torch.Tensor,
        experience_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode candidate into representation (for inference).
        
        Args:
            title_emb: Title embedding
            skills_emb: Skills embedding
            experience_emb: Experience embedding
        
        Returns:
            Candidate representation
        """
        return self.candidate_tower(title_emb, skills_emb, experience_emb)
    
    def encode_job(
        self,
        title_emb: torch.Tensor,
        skills_emb: torch.Tensor,
        requirement_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode job into representation (for inference).
        
        Args:
            title_emb: Title embedding
            skills_emb: Skills embedding
            requirement_emb: Requirement embedding
        
        Returns:
            Job representation
        """
        return self.job_tower(title_emb, skills_emb, requirement_emb)


