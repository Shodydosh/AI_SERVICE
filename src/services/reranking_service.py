"""Reranking Service với Cross-Encoder model."""
from typing import List, Dict, Tuple, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class RerankingService:
    """
    Reranking Service sử dụng Cross-Encoder để rerank kết quả từ FAISS.
    
    Cross-Encoder tốt hơn Bi-Encoder cho reranking vì:
    - Xem xét cả query và candidate cùng lúc
    - Chính xác hơn nhưng chậm hơn
    - Phù hợp cho reranking top 100-1000 results
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        use_cross_encoder: bool = True,
        top_k_rerank: int = 100
    ):
        """
        Initialize reranking service.
        
        Args:
            model_name: Cross-encoder model name (default: use sentence-transformers cross-encoder)
            use_cross_encoder: Whether to use cross-encoder (if False, use simple weighted reranking)
            top_k_rerank: Number of candidates to rerank (rerank top N from FAISS)
        """
        self.use_cross_encoder = use_cross_encoder
        self.top_k_rerank = top_k_rerank
        self.model = None
        
        if use_cross_encoder:
            try:
                from sentence_transformers import CrossEncoder
                # Use Vietnamese cross-encoder if available, else use multilingual
                self.model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
                self.model = CrossEncoder(self.model_name)
                logger.info(f"RerankingService initialized with cross-encoder: {self.model_name}")
            except ImportError:
                logger.warning("sentence-transformers not available, falling back to weighted reranking")
                self.use_cross_encoder = False
        else:
            logger.info("RerankingService initialized with weighted reranking (no cross-encoder)")
    
    def rerank_with_cross_encoder(
        self,
        query_text: str,
        candidates: List[Dict],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Rerank candidates sử dụng cross-encoder.
        
        Args:
            query_text: Query text (e.g., candidate desired job + skills)
            candidates: List of candidate dicts with 'text' or 'title' + 'description'
            top_k: Number of top results to return
            
        Returns:
            List of (candidate_id, reranked_score) tuples
        """
        if not self.model or not self.use_cross_encoder:
            # Fallback to weighted reranking
            return self.rerank_weighted(candidates, top_k)
        
        # Prepare pairs for cross-encoder
        pairs = []
        candidate_ids = []
        
        for candidate in candidates:
            candidate_id = candidate.get('id') or candidate.get('job_id') or candidate.get('candidate_id')
            if not candidate_id:
                continue
            
            # Combine candidate text fields
            candidate_text = self._get_candidate_text(candidate)
            if not candidate_text:
                continue
            
            pairs.append([query_text, candidate_text])
            candidate_ids.append(candidate_id)
        
        if not pairs:
            return []
        
        # Get scores from cross-encoder
        try:
            scores = self.model.predict(pairs)
            scores = scores.tolist() if hasattr(scores, 'tolist') else list(scores)
        except Exception as e:
            logger.error(f"Error in cross-encoder prediction: {e}")
            return self.rerank_weighted(candidates, top_k)
        
        # Combine IDs and scores
        results = list(zip(candidate_ids, scores))
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def _get_candidate_text(self, candidate: Dict) -> str:
        """Extract text from candidate dict for reranking."""
        parts = []
        
        if candidate.get('title'):
            parts.append(f"Title: {candidate['title']}")
        if candidate.get('description'):
            parts.append(f"Description: {candidate['description']}")
        if candidate.get('requirements'):
            parts.append(f"Requirements: {candidate['requirements']}")
        if candidate.get('skills'):
            parts.append(f"Skills: {candidate['skills']}")
        
        return " | ".join(parts) if parts else ""
    
    def rerank_weighted(
        self,
        candidates: List[Dict],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Weighted reranking (fallback khi không có cross-encoder).
        
        Args:
            candidates: List of candidate dicts with similarity_score
            top_k: Number of top results
            
        Returns:
            List of (candidate_id, weighted_score) tuples
        """
        results = []
        
        for candidate in candidates:
            candidate_id = candidate.get('id') or candidate.get('job_id') or candidate.get('candidate_id')
            if not candidate_id:
                continue
            
            # Use existing similarity_score if available
            score = candidate.get('similarity_score', 0.0)
            
            # Apply weights based on field similarities
            field_sims = candidate.get('field_similarities', {})
            if field_sims:
                # Weighted average: title 50%, skills 35%, experience 15%
                title_sim = field_sims.get('title', 0.0) or 0.0
                skills_sim = field_sims.get('skills', 0.0) or 0.0
                exp_sim = field_sims.get('experience', 0.0) or 0.0
                
                weighted_score = (
                    title_sim * 0.5 +
                    skills_sim * 0.35 +
                    exp_sim * 0.15
                )
                
                # Combine với existing score
                score = (score * 0.7 + weighted_score * 0.3)
            
            results.append((candidate_id, score))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def rerank_pipeline(
        self,
        query_text: str,
        faiss_results: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Full reranking pipeline: Rerank top N từ FAISS results.
        
        Args:
            query_text: Query text
            faiss_results: Results từ FAISS search (top 100-1000)
            top_k: Final number of results to return
            
        Returns:
            Reranked list of candidate dicts
        """
        # Take top_k_rerank candidates for reranking
        candidates_to_rerank = faiss_results[:self.top_k_rerank]
        
        # Rerank
        if self.use_cross_encoder:
            reranked = self.rerank_with_cross_encoder(query_text, candidates_to_rerank, top_k)
        else:
            reranked = self.rerank_weighted(candidates_to_rerank, top_k)
        
        # Map back to full candidate dicts
        reranked_dict = {cand_id: score for cand_id, score in reranked}
        
        final_results = []
        for candidate in faiss_results:
            candidate_id = candidate.get('id') or candidate.get('job_id') or candidate.get('candidate_id')
            if candidate_id in reranked_dict:
                candidate['similarity_score'] = reranked_dict[candidate_id]
                candidate['reranked'] = True
                final_results.append(candidate)
        
        # Sort by reranked score
        final_results.sort(key=lambda x: x.get('similarity_score', 0.0), reverse=True)
        
        return final_results[:top_k]
