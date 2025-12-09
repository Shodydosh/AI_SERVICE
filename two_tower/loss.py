"""Loss functions."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss."""
    
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        candidate_emb: torch.Tensor,
        job_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            candidate_emb: [batch_size, dim]
            job_emb: [batch_size, dim] (positive pairs)
        
        Returns:
            loss scalar
        """
        batch_size = candidate_emb.size(0)
        
        similarity = torch.matmul(candidate_emb, job_emb.T) / self.temperature
        
        labels = torch.arange(batch_size, device=candidate_emb.device)
        
        loss = F.cross_entropy(similarity, labels)
        
        return loss


class InfoNCEWithNegativesLoss(nn.Module):
    """InfoNCE loss with explicit negatives."""
    
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        candidate_emb: torch.Tensor,
        positive_job_emb: torch.Tensor,
        negative_job_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss with negatives.
        
        Args:
            candidate_emb: [batch_size, dim]
            positive_job_emb: [batch_size, dim]
            negative_job_emb: [num_negatives, dim]
        """
        batch_size = candidate_emb.size(0)
        
        pos_sim = torch.sum(candidate_emb * positive_job_emb, dim=1) / self.temperature
        
        neg_sim = torch.matmul(candidate_emb, negative_job_emb.T) / self.temperature
        
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        
        labels = torch.zeros(batch_size, dtype=torch.long, device=candidate_emb.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss


