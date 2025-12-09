"""Dataset and data loading."""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import random


class JobRecommendationDataset(Dataset):
    """Dataset for job recommendation."""
    
    def __init__(
        self,
        candidate_texts: List[str],
        job_texts: List[str],
        positive_pairs: List[Tuple[int, int]],
        all_job_texts: Optional[List[str]] = None,
        num_negatives: int = 0
    ):
        self.candidate_texts = candidate_texts
        self.job_texts = job_texts
        self.positive_pairs = positive_pairs
        self.all_job_texts = all_job_texts or job_texts
        self.num_negatives = num_negatives
        self.job_indices = list(range(len(self.all_job_texts)))
    
    def __len__(self) -> int:
        return len(self.positive_pairs)
    
    def __getitem__(self, idx: int) -> dict:
        candidate_idx, positive_job_idx = self.positive_pairs[idx]
        
        candidate_text = self.candidate_texts[candidate_idx]
        positive_job_text = self.job_texts[positive_job_idx]
        
        result = {
            'candidate_text': candidate_text,
            'positive_job_text': positive_job_text
        }
        
        if self.num_negatives > 0:
            negative_indices = random.sample(
                self.job_indices,
                min(self.num_negatives, len(self.job_indices))
            )
            negative_texts = [self.all_job_texts[i] for i in negative_indices]
            result['negative_job_texts'] = negative_texts
        
        return result


def collate_fn(batch: List[dict]) -> dict:
    """Collate function for DataLoader."""
    candidate_texts = [item['candidate_text'] for item in batch]
    positive_job_texts = [item['positive_job_text'] for item in batch]
    
    result = {
        'candidate_texts': candidate_texts,
        'positive_job_texts': positive_job_texts
    }
    
    if 'negative_job_texts' in batch[0]:
        negative_job_texts = [item['negative_job_texts'] for item in batch]
        result['negative_job_texts'] = negative_job_texts
    
    return result


def create_dataloader(
    candidate_texts: List[str],
    job_texts: List[str],
    positive_pairs: List[Tuple[int, int]],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    all_job_texts: Optional[List[str]] = None,
    num_negatives: int = 0
) -> DataLoader:
    """Create DataLoader."""
    dataset = JobRecommendationDataset(
        candidate_texts=candidate_texts,
        job_texts=job_texts,
        positive_pairs=positive_pairs,
        all_job_texts=all_job_texts,
        num_negatives=num_negatives
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


