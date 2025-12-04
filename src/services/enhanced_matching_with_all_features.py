"""Enhanced Matching Service với TẤT CẢ features mới."""
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
import logging
import time

from src.services.enhanced_multi_filter_matching_service import EnhancedMultiFilterMatchingService
from src.services.explainability_service import ExplainabilityService
from src.services.diversity_fairness_service import DiversityFairnessService
from src.services.multi_criteria_optimization_service import MultiCriteriaOptimizationService
from src.services.metrics_dashboard_service import MetricsDashboardService
from src.services.ab_testing_service import ABTestingService

logger = logging.getLogger(__name__)


class EnhancedMatchingWithAllFeatures:
    """
    Enhanced Matching Service với TẤT CẢ features:
    
    1. Hybrid Search
    2. Reranking
    3. Dynamic Filtering
    4. Contextual Embeddings
    5. Negative Signals
    6. Caching
    7. Explainability
    8. Diversity & Fairness
    9. Multi-Criteria Optimization
    10. Metrics Dashboard
    11. A/B Testing
    """
    
    def __init__(
        self,
        db: Session,
        use_explainability: bool = True,
        use_diversity_fairness: bool = True,
        use_multi_criteria: bool = True,
        use_metrics: bool = True,
        use_ab_testing: bool = True,
        **enhanced_service_kwargs
    ):
        """
        Initialize enhanced matching service với all features.
        
        Args:
            db: Database session
            use_explainability: Enable explainability
            use_diversity_fairness: Enable diversity & fairness
            use_multi_criteria: Enable multi-criteria optimization
            use_metrics: Enable metrics tracking
            use_ab_testing: Enable A/B testing
            **enhanced_service_kwargs: Arguments for EnhancedMultiFilterMatchingService
        """
        # Core matching service
        self.matching_service = EnhancedMultiFilterMatchingService(
            db=db,
            **enhanced_service_kwargs
        )
        
        # Additional services
        self.use_explainability = use_explainability
        if use_explainability:
            self.explainability = ExplainabilityService()
        else:
            self.explainability = None
        
        self.use_diversity_fairness = use_diversity_fairness
        if use_diversity_fairness:
            self.diversity_fairness = DiversityFairnessService()
        else:
            self.diversity_fairness = None
        
        self.use_multi_criteria = use_multi_criteria
        if use_multi_criteria:
            self.multi_criteria = MultiCriteriaOptimizationService()
        else:
            self.multi_criteria = None
        
        self.use_metrics = use_metrics
        if use_metrics:
            self.metrics = MetricsDashboardService()
        else:
            self.metrics = None
        
        self.use_ab_testing = use_ab_testing
        if use_ab_testing:
            self.ab_testing = ABTestingService()
        else:
            self.ab_testing = None
        
        self.db = db
        
        logger.info("EnhancedMatchingWithAllFeatures initialized")
        logger.info(f"  - Explainability: {use_explainability}")
        logger.info(f"  - Diversity & Fairness: {use_diversity_fairness}")
        logger.info(f"  - Multi-Criteria: {use_multi_criteria}")
        logger.info(f"  - Metrics: {use_metrics}")
        logger.info(f"  - A/B Testing: {use_ab_testing}")
    
    def find_jobs_for_candidate(
        self,
        candidate_id: str,
        top_k: int = 10,
        explain: bool = True,
        ensure_diversity: bool = True,
        use_pareto: bool = True
    ) -> Dict:
        """
        Find jobs for candidate với tất cả features.
        
        Args:
            candidate_id: Candidate ID
            top_k: Number of results
            explain: Whether to include explanations
            ensure_diversity: Whether to ensure diverse results
            use_pareto: Whether to use Pareto optimization
            
        Returns:
            Dict với results và metadata
        """
        start_time = time.time()
        
        # Track latency
        if self.metrics:
            self.metrics.track_latency('matching_start', 0.0)
        
        # Get candidate data
        from src.database.multi_field_repository import MultiFieldEmbeddingRepository
        repo = MultiFieldEmbeddingRepository(self.db)
        candidate = repo.get_candidate_multi_embedding(candidate_id)
        
        if not candidate:
            logger.error(f"Candidate {candidate_id} not found")
            return {'results': [], 'metadata': {}}
        
        candidate_data = {
            'candidate_id': candidate.candidate_id,
            'title': candidate.title,
            'skills': candidate.skills,
            'experience': candidate.experience,
            'expected_salary': getattr(candidate, 'expected_salary', None),
            'location': getattr(candidate, 'location', None),
            'industry': getattr(candidate, 'industry', None),
            'experience_years': getattr(candidate, 'experience_years', None)
        }
        
        # Core matching
        results = self.matching_service.find_jobs_for_candidate(candidate_id, top_k=top_k * 3)  # Get more for diversity and multi-criteria
        
        # Get job data for additional processing
        job_data_dict = {}
        job_embeddings = {}
        for result in results:
            job_id = result['job_id']
            job = repo.get_job_multi_embedding(job_id)
            if job:
                job_data_dict[job_id] = {
                    'job_id': job.job_id,
                    'title': job.title,
                    'skills': job.skills,
                    'requirements': job.requirement,
                    'salary_min': getattr(job, 'salary_min', None),
                    'salary_max': getattr(job, 'salary_max', None),
                    'location': job.location,
                    'industry': getattr(job, 'industry', None)
                }
                if job.title_embedding:
                    job_embeddings[job_id] = job.title_embedding
        
        # Multi-criteria optimization
        if self.use_multi_criteria and self.multi_criteria:
            logger.info("Applying multi-criteria optimization...")
            results = self.multi_criteria.optimize_multi_criteria(
                candidate=candidate_data,
                matches=results,
                job_data_dict=job_data_dict,
                top_k=top_k * 2,
                use_pareto=use_pareto
            )
        
        # Diversity & Fairness
        if self.use_diversity_fairness and self.diversity_fairness and ensure_diversity:
            logger.info("Applying diversity & fairness filtering...")
            results, diversity_metrics = self.diversity_fairness.apply_diversity_fairness(
                results=results,
                embeddings=job_embeddings,
                top_k=top_k
            )
        else:
            diversity_metrics = {}
            results = results[:top_k]
        
        # Explainability
        explanations = []
        if self.use_explainability and self.explainability and explain:
            logger.info("Generating explanations...")
            for result in results:
                job_id = result['job_id']
                if job_id in job_data_dict:
                    explanation = self.explainability.explain_match(
                        candidate=candidate_data,
                        job=job_data_dict[job_id],
                        score=result.get('similarity_score', 0.0),
                        field_similarities=result.get('field_similarities', {})
                    )
                    explanation['job_id'] = job_id
                    explanations.append(explanation)
        
        # Track metrics
        latency_ms = (time.time() - start_time) * 1000
        if self.metrics:
            self.metrics.track_latency('matching_full', latency_ms)
        
        # Build response
        response = {
            'results': results,
            'metadata': {
                'candidate_id': candidate_id,
                'total_results': len(results),
                'latency_ms': latency_ms,
                'diversity_metrics': diversity_metrics,
                'explanations': explanations if explanations else None
            }
        }
        
        return response
    
    def get_metrics_dashboard(self) -> Dict:
        """Get metrics dashboard data."""
        if self.metrics:
            return self.metrics.get_all_metrics()
        return {}
    
    def get_ab_test_metrics(self) -> Dict:
        """Get A/B testing metrics."""
        if self.ab_testing:
            return self.ab_testing.get_all_experiment_metrics()
        return {}

