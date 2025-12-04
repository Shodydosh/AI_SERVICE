"""Multi-Criteria Optimization Service: Pareto optimization với nhiều objectives."""
from typing import List, Dict, Optional, Callable
import logging
import numpy as np

logger = logging.getLogger(__name__)


class MultiCriteriaOptimizationService:
    """
    Multi-Criteria Optimization Service sử dụng Pareto optimization.
    
    Objectives:
    - Maximize skills_match
    - Maximize experience_fit
    - Minimize salary_gap
    - Maximize culture_fit
    """
    
    def __init__(
        self,
        objectives: Optional[List[Callable]] = None
    ):
        """
        Initialize multi-criteria optimization service.
        
        Args:
            objectives: List of objective functions (optional)
        """
        self.objectives = objectives or self._default_objectives()
        logger.info(f"MultiCriteriaOptimizationService initialized with {len(self.objectives)} objectives")
    
    def _default_objectives(self) -> List[Callable]:
        """Default objective functions."""
        def maximize_skills_match(candidate: Dict, job: Dict, match: Dict) -> float:
            """Maximize skills matching."""
            field_sims = match.get('field_similarities', {})
            return field_sims.get('skills', 0.0) or 0.0
        
        def maximize_experience_fit(candidate: Dict, job: Dict, match: Dict) -> float:
            """Maximize experience fit."""
            field_sims = match.get('field_similarities', {})
            return field_sims.get('experience', 0.0) or 0.0
        
        def minimize_salary_gap(candidate: Dict, job: Dict, match: Dict) -> float:
            """Minimize salary gap (inverted: higher is better)."""
            cand_salary = candidate.get('expected_salary', 0)
            job_salary_min = job.get('salary_min', 0)
            job_salary_max = job.get('salary_max', 0)
            
            if not cand_salary or not job_salary_min:
                return 0.5  # Neutral
            
            if job_salary_max:
                if job_salary_min <= cand_salary <= job_salary_max:
                    return 1.0  # Perfect match
                else:
                    gap = min(abs(cand_salary - job_salary_min), abs(cand_salary - job_salary_max))
                    return max(0.0, 1.0 - gap / max(cand_salary, job_salary_max))
            else:
                gap = abs(cand_salary - job_salary_min)
                return max(0.0, 1.0 - gap / max(cand_salary, job_salary_min))
        
        def maximize_culture_fit(candidate: Dict, job: Dict, match: Dict) -> float:
            """Maximize culture fit (placeholder - can be extended)."""
            # Simple: check location match
            cand_location = candidate.get('location') or ''
            job_location = job.get('location') or ''
            
            if cand_location and job_location:
                cand_location_lower = str(cand_location).lower()
                job_location_lower = str(job_location).lower()
                
                if cand_location_lower == job_location_lower:
                    return 1.0
                elif cand_location_lower in job_location_lower or job_location_lower in cand_location_lower:
                    return 0.7
                else:
                    return 0.3
            
            return 0.5  # Neutral
        
        return [
            maximize_skills_match,
            maximize_experience_fit,
            minimize_salary_gap,
            maximize_culture_fit
        ]
    
    def calculate_objective_scores(
        self,
        candidate: Dict,
        matches: List[Dict],
        job_data_dict: Dict[str, Dict]
    ) -> List[Dict]:
        """
        Calculate objective scores cho mỗi match.
        
        Args:
            candidate: Candidate data
            matches: List of match dicts
            job_data_dict: Dict of job_id -> job data
            
        Returns:
            List of matches với objective scores
        """
        scored_matches = []
        
        for match in matches:
            job_id = match.get('job_id')
            if not job_id or job_id not in job_data_dict:
                continue
            
            job_data = job_data_dict[job_id]
            
            # Calculate all objective scores
            objective_scores = []
            for obj_func in self.objectives:
                score = obj_func(candidate, job_data, match)
                objective_scores.append(score)
            
            # Add to match
            match['objective_scores'] = objective_scores
            match['objective_sum'] = sum(objective_scores)
            match['objective_mean'] = np.mean(objective_scores)
            
            scored_matches.append(match)
        
        return scored_matches
    
    def pareto_optimize(
        self,
        matches: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Pareto optimization: Find non-dominated solutions.
        
        Args:
            matches: List of matches với objective_scores
            top_k: Number of results to return
            
        Returns:
            Pareto-optimal results
        """
        if not matches or not matches[0].get('objective_scores'):
            return matches[:top_k]
        
        # Find Pareto front
        pareto_front = []
        
        for i, match1 in enumerate(matches):
            is_dominated = False
            
            for j, match2 in enumerate(matches):
                if i == j:
                    continue
                
                scores1 = match1['objective_scores']
                scores2 = match2['objective_scores']
                
                # Check if match2 dominates match1
                # match2 dominates match1 if all objectives are >= and at least one is >
                if all(s2 >= s1 for s1, s2 in zip(scores1, scores2)) and \
                   any(s2 > s1 for s1, s2 in zip(scores1, scores2)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(match1)
        
        # If Pareto front is too large, select top K by weighted sum
        if len(pareto_front) > top_k:
            # Sort by weighted sum of objectives
            pareto_front.sort(key=lambda x: x.get('objective_sum', 0.0), reverse=True)
            pareto_front = pareto_front[:top_k]
        
        logger.info(f"Pareto optimization: {len(pareto_front)} non-dominated solutions from {len(matches)} candidates")
        
        return pareto_front
    
    def optimize_multi_criteria(
        self,
        candidate: Dict,
        matches: List[Dict],
        job_data_dict: Dict[str, Dict],
        top_k: int = 10,
        use_pareto: bool = True
    ) -> List[Dict]:
        """
        Full multi-criteria optimization pipeline.
        
        Args:
            candidate: Candidate data
            matches: List of matches
            job_data_dict: Dict of job_id -> job data
            top_k: Number of results
            use_pareto: Whether to use Pareto optimization
            
        Returns:
            Optimized results
        """
        # Calculate objective scores
        scored_matches = self.calculate_objective_scores(candidate, matches, job_data_dict)
        
        if use_pareto:
            # Pareto optimization
            optimized = self.pareto_optimize(scored_matches, top_k)
        else:
            # Simple: sort by weighted sum
            scored_matches.sort(key=lambda x: x.get('objective_sum', 0.0), reverse=True)
            optimized = scored_matches[:top_k]
        
        return optimized

