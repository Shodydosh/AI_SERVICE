"""Diversity & Fairness Service: Debiasing, diverse results, fairness metrics."""
from typing import List, Dict, Optional, Set, Tuple
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class DiversityFairnessService:
    """
    Diversity & Fairness Service để:
    1. Debiasing embeddings
    2. Ensure diverse result set
    3. Monitor fairness metrics
    """
    
    def __init__(
        self,
        diversity_threshold: float = 0.3,
        max_similar_results: int = 3
    ):
        """
        Initialize diversity & fairness service.
        
        Args:
            diversity_threshold: Minimum diversity score (0-1)
            max_similar_results: Max number of similar results in top K
        """
        self.diversity_threshold = diversity_threshold
        self.max_similar_results = max_similar_results
        logger.info(f"DiversityFairnessService initialized (diversity_threshold={diversity_threshold})")
    
    def ensure_diverse_results(
        self,
        results: List[Dict],
        embeddings: Dict[str, List[float]],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Ensure diverse result set (không chỉ top similar).
        
        Args:
            results: List of result dicts (sorted by score)
            embeddings: Dict of id -> embedding
            top_k: Number of diverse results to return
            
        Returns:
            Diverse result set
        """
        if len(results) <= top_k:
            return results
        
        diverse_results = []
        used_indices = set()
        
        # Start with top result
        if results:
            diverse_results.append(results[0])
            used_indices.add(0)
        
        # Greedy selection: pick results that are diverse from already selected
        while len(diverse_results) < top_k and len(used_indices) < len(results):
            best_idx = None
            best_diversity = -1
            
            for i, result in enumerate(results):
                if i in used_indices:
                    continue
                
                # Calculate minimum distance to already selected results
                result_id = result.get('job_id') or result.get('id')
                if not result_id or result_id not in embeddings:
                    continue
                
                result_emb = np.array(embeddings[result_id], dtype=np.float32)
                
                min_similarity = 1.0
                for selected_result in diverse_results:
                    selected_id = selected_result.get('job_id') or selected_result.get('id')
                    if selected_id and selected_id in embeddings:
                        selected_emb = np.array(embeddings[selected_id], dtype=np.float32)
                        sim = cosine_similarity([result_emb], [selected_emb])[0][0]
                        min_similarity = min(min_similarity, sim)
                
                # Diversity = 1 - similarity
                diversity = 1.0 - min_similarity
                
                # Combine diversity với original score
                original_score = result.get('similarity_score', 0.0)
                combined_score = original_score * 0.7 + diversity * 0.3
                
                if combined_score > best_diversity:
                    best_diversity = combined_score
                    best_idx = i
            
            if best_idx is not None:
                diverse_results.append(results[best_idx])
                used_indices.add(best_idx)
            else:
                break
        
        # Sort by original score
        diverse_results.sort(key=lambda x: x.get('similarity_score', 0.0), reverse=True)
        
        logger.info(f"Selected {len(diverse_results)} diverse results from {len(results)} candidates")
        
        return diverse_results
    
    def calculate_diversity_score(
        self,
        results: List[Dict],
        embeddings: Dict[str, List[float]]
    ) -> float:
        """
        Calculate diversity score của result set.
        
        Args:
            results: List of results
            embeddings: Dict of id -> embedding
            
        Returns:
            Diversity score (0-1), higher = more diverse
        """
        if len(results) < 2:
            return 1.0
        
        # Get embeddings
        result_embeddings = []
        for result in results:
            result_id = result.get('job_id') or result.get('id')
            if result_id and result_id in embeddings:
                result_embeddings.append(np.array(embeddings[result_id], dtype=np.float32))
        
        if len(result_embeddings) < 2:
            return 1.0
        
        # Calculate pairwise similarities
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
    
    def debias_embeddings(
        self,
        embeddings: Dict[str, List[float]],
        bias_vectors: Optional[List[List[float]]] = None
    ) -> Dict[str, List[float]]:
        """
        Debias embeddings bằng cách remove bias vectors.
        
        Args:
            embeddings: Dict of id -> embedding
            bias_vectors: List of bias vectors to remove (optional)
            
        Returns:
            Debiased embeddings
        """
        if not bias_vectors:
            # Default: remove gender/age bias (can be extended)
            logger.info("No bias vectors provided, skipping debiasing")
            return embeddings
        
        debiased = {}
        
        for item_id, emb in embeddings.items():
            emb_array = np.array(emb, dtype=np.float32)
            
            # Remove each bias vector
            for bias_vec in bias_vectors:
                bias_array = np.array(bias_vec, dtype=np.float32)
                
                # Project và subtract bias component
                bias_norm = np.linalg.norm(bias_array)
                if bias_norm > 1e-10:
                    projection = np.dot(emb_array, bias_array) / (bias_norm ** 2)
                    emb_array = emb_array - projection * bias_array
            
            # Renormalize
            norm = np.linalg.norm(emb_array)
            if norm > 1e-10:
                emb_array = emb_array / norm
            
            debiased[item_id] = emb_array.tolist()
        
        logger.info(f"Debiased {len(debiased)} embeddings")
        
        return debiased
    
    def calculate_fairness_metrics(
        self,
        results: List[Dict],
        protected_attributes: Dict[str, str]
    ) -> Dict:
        """
        Calculate fairness metrics.
        
        Args:
            results: List of results
            protected_attributes: Dict of result_id -> protected attribute value
            
        Returns:
            Fairness metrics dict
        """
        if not protected_attributes:
            return {}
        
        # Count by protected attribute
        counts = {}
        total = len(results)
        
        for result in results:
            result_id = result.get('job_id') or result.get('id')
            attr_value = protected_attributes.get(result_id, 'unknown')
            counts[attr_value] = counts.get(attr_value, 0) + 1
        
        # Calculate representation percentages
        representation = {
            attr: count / total if total > 0 else 0.0
            for attr, count in counts.items()
        }
        
        # Calculate fairness score (higher = more fair)
        if len(representation) > 1:
            # Fairness = 1 - std deviation of representation
            values = list(representation.values())
            std_dev = np.std(values)
            fairness_score = 1.0 - min(std_dev, 1.0)
        else:
            fairness_score = 1.0
        
        metrics = {
            'representation': representation,
            'fairness_score': fairness_score,
            'total_results': total,
            'attribute_counts': counts
        }
        
        return metrics
    
    def apply_diversity_fairness(
        self,
        results: List[Dict],
        embeddings: Dict[str, List[float]],
        protected_attributes: Optional[Dict[str, str]] = None,
        top_k: int = 10
    ) -> Tuple[List[Dict], Dict]:
        """
        Apply diversity và fairness filtering.
        
        Args:
            results: List of results
            embeddings: Dict of id -> embedding
            protected_attributes: Optional protected attributes
            top_k: Number of results to return
            
        Returns:
            Tuple of (diverse_results, fairness_metrics)
        """
        # Ensure diversity
        diverse_results = self.ensure_diverse_results(results, embeddings, top_k)
        
        # Calculate fairness metrics
        fairness_metrics = {}
        if protected_attributes:
            fairness_metrics = self.calculate_fairness_metrics(
                diverse_results,
                protected_attributes
            )
        
        # Calculate diversity score
        diversity_score = self.calculate_diversity_score(diverse_results, embeddings)
        
        metrics = {
            'diversity_score': diversity_score,
            'fairness_metrics': fairness_metrics
        }
        
        return diverse_results, metrics

