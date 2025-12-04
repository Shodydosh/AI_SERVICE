"""Async Matching Service: Chuyển I/O operations sang async."""
from typing import List, Dict, Optional
import logging
import asyncio
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class AsyncMatchingService:
    """
    Async Matching Service để xử lý matching operations bất đồng bộ.
    
    Lợi ích:
    - Xử lý nhiều candidates đồng thời
    - Non-blocking I/O operations
    - Better resource utilization
    """
    
    def __init__(self, matching_service):
        """
        Initialize async matching service.
        
        Args:
            matching_service: Synchronous matching service instance
        """
        self.matching_service = matching_service
        logger.info("AsyncMatchingService initialized")
    
    async def match_candidate_async(
        self,
        candidate_id: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Match candidate asynchronously.
        
        Args:
            candidate_id: Candidate ID
            top_k: Number of top matches
            
        Returns:
            List of job matches
        """
        # Run in thread pool để không block event loop
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            self.matching_service.find_jobs_for_candidate,
            candidate_id,
            top_k
        )
        return results
    
    async def match_candidate_text_async(
        self,
        title: Optional[str] = None,
        skills: Optional[str] = None,
        experience: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Match candidate from text asynchronously.
        
        Args:
            title: Desired job title
            skills: Skills
            experience: Experience
            top_k: Number of top matches
            
        Returns:
            List of job matches
        """
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            self.matching_service.find_jobs_for_candidate_text,
            title,
            skills,
            experience,
            top_k
        )
        return results
    
    async def match_multiple_candidates_async(
        self,
        candidate_ids: List[str],
        top_k: int = 10
    ) -> Dict[str, List[Dict]]:
        """
        Match multiple candidates concurrently.
        
        Args:
            candidate_ids: List of candidate IDs
            top_k: Number of top matches per candidate
            
        Returns:
            Dict of candidate_id -> list of matches
        """
        # Create tasks for all candidates
        tasks = [
            self.match_candidate_async(cand_id, top_k)
            for cand_id in candidate_ids
        ]
        
        # Execute concurrently
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        results_dict = {}
        for cand_id, result in zip(candidate_ids, results_list):
            if isinstance(result, Exception):
                logger.error(f"Error matching candidate {cand_id}: {result}")
                results_dict[cand_id] = []
            else:
                results_dict[cand_id] = result
        
        return results_dict
    
    async def fetch_experience_matches_async(
        self,
        candidate_experience_emb: List[float],
        top_k: int = 1000
    ) -> List[Dict]:
        """Fetch experience matches asynchronously."""
        loop = asyncio.get_event_loop()
        # Assuming matching_service has _filter_by_experience_requirement method
        results = await loop.run_in_executor(
            None,
            self.matching_service._filter_by_experience_requirement,
            candidate_experience_emb,
            None,  # job_ids
            top_k
        )
        return results
    
    async def fetch_skills_matches_async(
        self,
        candidate_skills_emb: List[float],
        job_ids: List[str],
        top_k: int = 100
    ) -> List[Dict]:
        """Fetch skills matches asynchronously."""
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            self.matching_service._filter_by_skills,
            candidate_skills_emb,
            job_ids,
            top_k
        )
        return results
    
    async def fetch_title_matches_async(
        self,
        candidate_title_emb: List[float],
        job_ids: Optional[List[str]],
        top_k: int = 10
    ) -> List[Dict]:
        """Fetch title matches asynchronously."""
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            self.matching_service._filter_by_title,
            candidate_title_emb,
            job_ids,
            top_k
        )
        return results

