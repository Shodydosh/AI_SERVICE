"""Title Matching Validator - Kiểm tra và tăng cường mối quan hệ giữa desired job và recommended job."""
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class TitleMatchingValidator:
    """
    Validator để kiểm tra và tăng cường title matching giữa candidate desired job và JD title.
    
    Mục đích:
    - Đảm bảo các JD được đề xuất có title tương tự với desired job của candidate
    - Boost score cho các JD có title matching cao
    - Filter out các JD có title matching quá thấp
    """
    
    def __init__(
        self,
        min_title_similarity: float = 0.4,
        boost_threshold: float = 0.6,
        boost_factor: float = 1.2
    ):
        """
        Initialize title matching validator.
        
        Args:
            min_title_similarity: Minimum title similarity threshold (0-1)
                                 JD có similarity < threshold sẽ bị loại bỏ
            boost_threshold: Title similarity threshold để boost score (0-1)
                            JD có similarity >= threshold sẽ được boost
            boost_factor: Factor để boost score (ví dụ: 1.2 = tăng 20%)
        """
        self.min_title_similarity = min_title_similarity
        self.boost_threshold = boost_threshold
        self.boost_factor = boost_factor
    
    def normalize_title(self, title: str) -> str:
        """
        Normalize title text: lowercase, remove extra spaces.
        
        Args:
            title: Job title text
            
        Returns:
            Normalized title
        """
        if not title:
            return ""
        return " ".join(title.strip().lower().split())
    
    def calculate_title_similarity(
        self,
        candidate_title_emb: List[float],
        jd_title_emb: List[float]
    ) -> float:
        """
        Tính title similarity giữa candidate desired job và JD title.
        
        Args:
            candidate_title_emb: Candidate desired job title embedding
            jd_title_emb: JD title embedding
            
        Returns:
            Title similarity score (0-1)
        """
        if not candidate_title_emb or not jd_title_emb:
            return 0.0
        
        try:
            # Convert to numpy arrays
            cand_emb = np.array(candidate_title_emb, dtype=np.float32)
            jd_emb = np.array(jd_title_emb, dtype=np.float32)
            
            # Check if embeddings are valid (not zero vectors)
            cand_norm = np.linalg.norm(cand_emb)
            jd_norm = np.linalg.norm(jd_emb)
            
            if cand_norm < 1e-10 or jd_norm < 1e-10:
                logger.warning("Zero vector embedding detected in title similarity calculation")
                return 0.0
            
            # Calculate cosine similarity
            similarity = cosine_similarity([cand_emb], [jd_emb])[0][0]
            
            # Ensure similarity is in [0, 1] range
            similarity = max(0.0, min(1.0, similarity))
            
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating title similarity: {e}")
            return 0.0
    
    def validate_and_filter(
        self,
        job_matches: List[Dict],
        candidate_title_emb: Optional[List[float]] = None,
        job_title_embeddings: Optional[Dict[str, List[float]]] = None
    ) -> List[Dict]:
        """
        Validate và filter job matches dựa trên title similarity.
        
        Args:
            job_matches: List of job matches với structure:
                {
                    "job_id": str,
                    "title": str,
                    "similarity_score": float,
                    "field_similarities": {
                        "title": float (optional)
                    }
                }
            candidate_title_emb: Candidate desired job title embedding
            job_title_embeddings: Dict mapping job_id -> title embedding
            
        Returns:
            Filtered và validated job matches
        """
        if not job_matches:
            return []
        
        validated_matches = []
        
        for match in job_matches:
            job_id = match.get("job_id")
            title_sim = match.get("field_similarities", {}).get("title")
            
            # Nếu đã có title similarity trong field_similarities, sử dụng nó
            if title_sim is not None:
                if title_sim >= self.min_title_similarity:
                    validated_matches.append(match)
                else:
                    logger.debug(f"Job {job_id} filtered out: title similarity {title_sim:.4f} < {self.min_title_similarity}")
                continue
            
            # Nếu không có title similarity, tính toán nếu có embeddings
            if candidate_title_emb and job_title_embeddings and job_id in job_title_embeddings:
                jd_title_emb = job_title_embeddings[job_id]
                title_sim = self.calculate_title_similarity(candidate_title_emb, jd_title_emb)
                
                # Update field_similarities
                if "field_similarities" not in match:
                    match["field_similarities"] = {}
                match["field_similarities"]["title"] = title_sim
                
                if title_sim >= self.min_title_similarity:
                    validated_matches.append(match)
                else:
                    logger.debug(f"Job {job_id} filtered out: title similarity {title_sim:.4f} < {self.min_title_similarity}")
            else:
                # Nếu không có embeddings, giữ nguyên match (không filter)
                logger.warning(f"Job {job_id}: No title embeddings available, keeping match without title validation")
                validated_matches.append(match)
        
        logger.info(f"Title validation: {len(validated_matches)}/{len(job_matches)} jobs passed title similarity threshold ({self.min_title_similarity})")
        
        return validated_matches
    
    def boost_title_matches(
        self,
        job_matches: List[Dict],
        candidate_title_emb: Optional[List[float]] = None,
        job_title_embeddings: Optional[Dict[str, List[float]]] = None
    ) -> List[Dict]:
        """
        Boost score cho các job matches có title similarity cao.
        
        Args:
            job_matches: List of job matches
            candidate_title_emb: Candidate desired job title embedding
            job_title_embeddings: Dict mapping job_id -> title embedding
            
        Returns:
            Job matches với boosted scores
        """
        if not job_matches:
            return []
        
        boosted_matches = []
        boosted_count = 0
        
        for match in job_matches:
            job_id = match.get("job_id")
            original_score = match.get("similarity_score", 0.0)
            title_sim = match.get("field_similarities", {}).get("title")
            
            # Nếu đã có title similarity, sử dụng nó
            if title_sim is None and candidate_title_emb and job_title_embeddings and job_id in job_title_embeddings:
                jd_title_emb = job_title_embeddings[job_id]
                title_sim = self.calculate_title_similarity(candidate_title_emb, jd_title_emb)
                
                # Update field_similarities
                if "field_similarities" not in match:
                    match["field_similarities"] = {}
                match["field_similarities"]["title"] = title_sim
            
            # Boost score nếu title similarity >= threshold
            if title_sim and title_sim >= self.boost_threshold:
                boosted_score = original_score * self.boost_factor
                match["similarity_score"] = round(boosted_score, 4)
                match["title_boosted"] = True
                match["title_boost_factor"] = self.boost_factor
                boosted_count += 1
                logger.debug(f"Job {job_id}: Boosted score from {original_score:.4f} to {boosted_score:.4f} (title similarity: {title_sim:.4f})")
            else:
                match["title_boosted"] = False
            
            boosted_matches.append(match)
        
        if boosted_count > 0:
            logger.info(f"Title boost: {boosted_count}/{len(job_matches)} jobs received title matching boost (similarity >= {self.boost_threshold})")
        
        # Re-sort by boosted score
        boosted_matches.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return boosted_matches
    
    def validate_and_boost(
        self,
        job_matches: List[Dict],
        candidate_title_emb: Optional[List[float]] = None,
        job_title_embeddings: Optional[Dict[str, List[float]]] = None
    ) -> List[Dict]:
        """
        Kết hợp validate và boost: filter theo min threshold, sau đó boost các matches cao.
        
        Args:
            job_matches: List of job matches
            candidate_title_emb: Candidate desired job title embedding
            job_title_embeddings: Dict mapping job_id -> title embedding
            
        Returns:
            Validated và boosted job matches
        """
        # Step 1: Filter by minimum title similarity
        validated = self.validate_and_filter(
            job_matches,
            candidate_title_emb,
            job_title_embeddings
        )
        
        # Step 2: Boost high title similarity matches
        boosted = self.boost_title_matches(
            validated,
            candidate_title_emb,
            job_title_embeddings
        )
        
        return boosted
    
    def get_title_matching_stats(
        self,
        job_matches: List[Dict]
    ) -> Dict:
        """
        Lấy thống kê về title matching trong job matches.
        
        Args:
            job_matches: List of job matches
            
        Returns:
            Dictionary với statistics:
            {
                "total_matches": int,
                "with_title_similarity": int,
                "avg_title_similarity": float,
                "min_title_similarity": float,
                "max_title_similarity": float,
                "above_threshold": int,
                "boosted_count": int
            }
        """
        title_similarities = []
        boosted_count = 0
        
        for match in job_matches:
            title_sim = match.get("field_similarities", {}).get("title")
            if title_sim is not None:
                title_similarities.append(title_sim)
            
            if match.get("title_boosted", False):
                boosted_count += 1
        
        stats = {
            "total_matches": len(job_matches),
            "with_title_similarity": len(title_similarities),
            "avg_title_similarity": float(np.mean(title_similarities)) if title_similarities else 0.0,
            "min_title_similarity": float(np.min(title_similarities)) if title_similarities else 0.0,
            "max_title_similarity": float(np.max(title_similarities)) if title_similarities else 0.0,
            "above_threshold": sum(1 for s in title_similarities if s >= self.min_title_similarity) if title_similarities else 0,
            "boosted_count": boosted_count
        }
        
        return stats

