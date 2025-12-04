"""Hybrid Search Service: Kết hợp semantic + keyword matching."""
from typing import List, Dict, Tuple, Optional
import logging
import re
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class HybridSearchService:
    """
    Hybrid Search Service kết hợp:
    1. Semantic search (embedding-based)
    2. Keyword matching (exact + fuzzy)
    3. Boost scores dựa trên keyword matches
    """
    
    def __init__(self, keyword_boost: float = 0.15):
        """
        Initialize hybrid search service.
        
        Args:
            keyword_boost: Boost factor cho keyword matches (default: 0.15 = 15% boost)
        """
        self.keyword_boost = keyword_boost
        logger.info(f"HybridSearchService initialized with keyword_boost={keyword_boost}")
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords từ text (loại bỏ stopwords, normalize).
        
        Args:
            text: Input text
            
        Returns:
            List of keywords
        """
        if not text:
            return []
        
        # Normalize: lowercase, remove special chars
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Vietnamese stopwords (common words to ignore)
        stopwords = {
            'và', 'của', 'cho', 'với', 'từ', 'trong', 'là', 'được', 'có', 'một',
            'các', 'này', 'đó', 'khi', 'nếu', 'như', 'theo', 'về', 'sẽ', 'đã',
            'cần', 'phải', 'nên', 'để', 'bằng', 'vì', 'do', 'nên', 'mà', 'hoặc'
        }
        
        # Split và filter
        words = text.split()
        keywords = [w for w in words if len(w) > 2 and w not in stopwords]
        
        return keywords
    
    def calculate_keyword_score(
        self,
        query_keywords: List[str],
        target_keywords: List[str]
    ) -> float:
        """
        Tính keyword matching score.
        
        Args:
            query_keywords: Keywords từ query
            target_keywords: Keywords từ target text
            
        Returns:
            Keyword score (0-1)
        """
        if not query_keywords or not target_keywords:
            return 0.0
        
        # Exact matches
        query_set = set(query_keywords)
        target_set = set(target_keywords)
        exact_matches = len(query_set & target_set)
        
        # Fuzzy matches (substring)
        fuzzy_matches = 0
        for q_word in query_keywords:
            for t_word in target_keywords:
                if q_word in t_word or t_word in q_word:
                    fuzzy_matches += 0.5
        
        # Normalize
        total_keywords = len(query_keywords)
        if total_keywords == 0:
            return 0.0
        
        exact_score = exact_matches / total_keywords
        fuzzy_score = min(fuzzy_matches / total_keywords, 1.0)
        
        # Weighted: 70% exact, 30% fuzzy
        keyword_score = exact_score * 0.7 + fuzzy_score * 0.3
        
        return min(keyword_score, 1.0)
    
    def hybrid_score(
        self,
        semantic_score: float,
        keyword_matches: int,
        total_keywords: int
    ) -> float:
        """
        Tính hybrid score kết hợp semantic + keyword.
        
        Args:
            semantic_score: Semantic similarity score (0-1)
            keyword_matches: Số keyword matches
            total_keywords: Tổng số keywords trong query
            
        Returns:
            Hybrid score (boosted)
        """
        if total_keywords == 0:
            return semantic_score
        
        keyword_score = keyword_matches / total_keywords
        boost = 1 + (keyword_score * self.keyword_boost)
        
        # Cap boost để không quá cao
        boost = min(boost, 1.5)  # Max 50% boost
        
        hybrid_score = semantic_score * boost
        
        # Ensure score stays in [0, 1] range
        return min(hybrid_score, 1.0)
    
    def search_with_keywords(
        self,
        query_text: str,
        query_embedding: List[float],
        candidates: List[Dict],
        candidate_embeddings: Dict[str, List[float]],
        candidate_texts: Dict[str, str],
        top_k: int = 10
    ) -> List[Tuple[str, float, Dict]]:
        """
        Hybrid search: semantic + keyword matching.
        
        Args:
            query_text: Query text (for keyword extraction)
            query_embedding: Query embedding (for semantic search)
            candidates: List of candidate dicts with 'id' field
            candidate_embeddings: Dict of candidate_id -> embedding
            candidate_texts: Dict of candidate_id -> text (for keyword matching)
            top_k: Number of top results
            
        Returns:
            List of (candidate_id, hybrid_score, metadata) tuples
        """
        # Extract keywords from query
        query_keywords = self.extract_keywords(query_text)
        logger.debug(f"Extracted {len(query_keywords)} keywords from query: {query_keywords[:10]}")
        
        # Calculate semantic similarities
        semantic_scores = {}
        query_emb_array = np.array(query_embedding, dtype=np.float32)
        
        for candidate in candidates:
            candidate_id = candidate.get('id') or candidate.get('job_id') or candidate.get('candidate_id')
            if not candidate_id:
                continue
            
            cand_emb = candidate_embeddings.get(candidate_id)
            if cand_emb:
                cand_emb_array = np.array(cand_emb, dtype=np.float32)
                semantic_sim = cosine_similarity([query_emb_array], [cand_emb_array])[0][0]
                semantic_scores[candidate_id] = semantic_sim
        
        # Calculate keyword scores and hybrid scores
        results = []
        for candidate in candidates:
            candidate_id = candidate.get('id') or candidate.get('job_id') or candidate.get('candidate_id')
            if not candidate_id:
                continue
            
            semantic_score = semantic_scores.get(candidate_id, 0.0)
            
            # Keyword matching
            candidate_text = candidate_texts.get(candidate_id, '')
            if candidate_text:
                target_keywords = self.extract_keywords(candidate_text)
                keyword_score = self.calculate_keyword_score(query_keywords, target_keywords)
                keyword_matches = len(set(query_keywords) & set(target_keywords))
            else:
                keyword_score = 0.0
                keyword_matches = 0
            
            # Hybrid score
            hybrid_score = self.hybrid_score(
                semantic_score,
                keyword_matches,
                len(query_keywords) if query_keywords else 1
            )
            
            metadata = {
                'semantic_score': semantic_score,
                'keyword_score': keyword_score,
                'keyword_matches': keyword_matches,
                'hybrid_score': hybrid_score
            }
            
            results.append((candidate_id, hybrid_score, metadata))
        
        # Sort by hybrid score
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k
        return results[:top_k]

