"""Dynamic Filtering Service: Điều chỉnh số lượng filter theo data quality."""
from typing import List, Dict, Tuple, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class DynamicFilteringService:
    """
    Dynamic Filtering Service điều chỉnh số lượng filter dựa trên:
    1. Data quality (embedding norms, field completeness)
    2. Similarity distribution
    3. Result diversity
    """
    
    def __init__(
        self,
        min_quality_threshold: float = 0.3,
        diversity_threshold: float = 0.1
    ):
        """
        Initialize dynamic filtering service.
        
        Args:
            min_quality_threshold: Minimum quality score để giữ lại result
            diversity_threshold: Minimum diversity để đảm bảo kết quả đa dạng
        """
        self.min_quality_threshold = min_quality_threshold
        self.diversity_threshold = diversity_threshold
        logger.info(f"DynamicFilteringService initialized (min_quality={min_quality_threshold}, diversity={diversity_threshold})")
    
    def assess_data_quality(
        self,
        embeddings: Dict[str, List[float]],
        texts: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Đánh giá chất lượng data cho mỗi candidate/job.
        
        Args:
            embeddings: Dict of id -> embedding
            texts: Dict of id -> text
            
        Returns:
            Dict of id -> quality_score (0-1)
        """
        quality_scores = {}
        
        for item_id, emb in embeddings.items():
            score = 1.0
            
            # Check embedding quality
            if emb:
                emb_array = np.array(emb, dtype=np.float32)
                norm = np.linalg.norm(emb_array)
                
                # Penalize zero or very small embeddings
                if norm < 1e-10:
                    score *= 0.1  # Very low quality
                elif norm < 0.5:
                    score *= 0.5  # Low quality
                elif norm > 2.0:
                    score *= 0.8  # Possibly over-normalized
            else:
                score *= 0.1  # No embedding
            
            # Check text quality
            text = texts.get(item_id, '')
            if not text or len(text.strip()) < 10:
                score *= 0.5  # Low text quality
            elif len(text) > 5000:
                score *= 0.9  # Very long text (may be noisy)
            
            quality_scores[item_id] = score
        
        return quality_scores
    
    def calculate_diversity(
        self,
        results: List[Dict],
        embeddings: Dict[str, List[float]]
    ) -> float:
        """
        Tính diversity của results (đảm bảo không quá similar).
        
        Args:
            results: List of result dicts
            embeddings: Dict of id -> embedding
            
        Returns:
            Diversity score (0-1), higher = more diverse
        """
        if len(results) < 2:
            return 1.0
        
        # Get embeddings for results
        result_embeddings = []
        for result in results:
            item_id = result.get('id') or result.get('job_id') or result.get('candidate_id')
            if item_id and item_id in embeddings:
                result_embeddings.append(np.array(embeddings[item_id], dtype=np.float32))
        
        if len(result_embeddings) < 2:
            return 1.0
        
        # Calculate pairwise similarities
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = []
        for i in range(len(result_embeddings)):
            for j in range(i + 1, len(result_embeddings)):
                sim = cosine_similarity([result_embeddings[i]], [result_embeddings[j]])[0][0]
                similarities.append(sim)
        
        if not similarities:
            return 1.0
        
        # Diversity = 1 - average similarity
        avg_similarity = np.mean(similarities)
        diversity = 1.0 - avg_similarity
        
        return max(0.0, min(1.0, diversity))
    
    def adjust_filter_sizes(
        self,
        step1_count: int,
        step2_count: int,
        step3_count: int,
        quality_scores: Dict[str, float],
        results_diversity: float
    ) -> Tuple[int, int, int]:
        """
        Điều chỉnh số lượng filter dựa trên data quality và diversity.
        
        Args:
            step1_count: Số lượng results từ step 1
            step2_count: Số lượng results từ step 2
            step3_count: Số lượng results từ step 3
            quality_scores: Quality scores cho results
            results_diversity: Diversity score của results
            
        Returns:
            Adjusted (step1_count, step2_count, step3_count)
        """
        # Calculate average quality
        if quality_scores:
            avg_quality = np.mean(list(quality_scores.values()))
        else:
            avg_quality = 0.5
        
        # Adjust based on quality
        if avg_quality < 0.3:
            # Low quality: Increase filter sizes để có nhiều candidates hơn
            step1_count = int(step1_count * 1.5)
            step2_count = int(step2_count * 1.3)
            step3_count = int(step3_count * 1.2)
            logger.info(f"Low quality detected (avg={avg_quality:.2f}), increasing filter sizes")
        elif avg_quality > 0.8:
            # High quality: Có thể giảm filter sizes
            step1_count = int(step1_count * 0.9)
            step2_count = int(step2_count * 0.9)
            logger.info(f"High quality detected (avg={avg_quality:.2f}), optimizing filter sizes")
        
        # Adjust based on diversity
        if results_diversity < self.diversity_threshold:
            # Low diversity: Increase filter sizes để có kết quả đa dạng hơn
            step1_count = int(step1_count * 1.2)
            step2_count = int(step2_count * 1.1)
            logger.info(f"Low diversity detected ({results_diversity:.2f}), increasing filter sizes")
        
        # Ensure minimum sizes
        step1_count = max(step1_count, 500)
        step2_count = max(step2_count, 50)
        step3_count = max(step3_count, 10)
        
        return (step1_count, step2_count, step3_count)
    
    def filter_by_quality(
        self,
        results: List[Dict],
        quality_scores: Dict[str, float],
        min_quality: Optional[float] = None
    ) -> List[Dict]:
        """
        Filter results dựa trên quality scores.
        
        Args:
            results: List of result dicts
            quality_scores: Quality scores
            min_quality: Minimum quality threshold (default: self.min_quality_threshold)
            
        Returns:
            Filtered results
        """
        if min_quality is None:
            min_quality = self.min_quality_threshold
        
        filtered = []
        for result in results:
            item_id = result.get('id') or result.get('job_id') or result.get('candidate_id')
            quality = quality_scores.get(item_id, 0.5)
            
            if quality >= min_quality:
                result['quality_score'] = quality
                filtered.append(result)
        
        return filtered

