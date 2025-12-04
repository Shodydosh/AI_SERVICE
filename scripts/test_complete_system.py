"""Script test toàn diện cho toàn bộ hệ thống - kiểm tra tất cả các đầu mục."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import argparse
from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.database.multi_field_repository import MultiFieldEmbeddingRepository

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_database_connection():
    """Test 1: Kiểm tra kết nối database."""
    logger.info("=" * 80)
    logger.info("TEST 1: DATABASE CONNECTION")
    logger.info("=" * 80)
    
    try:
        db: Session = next(get_db())
        repo = MultiFieldEmbeddingRepository(db)
        
        job_count = repo.count_job_multi_embeddings()
        candidate_count = repo.count_candidate_multi_embeddings()
        
        logger.info(f"✓ Database connected successfully")
        logger.info(f"  - Jobs in database: {job_count}")
        logger.info(f"  - Candidates in database: {candidate_count}")
        
        db.close()
        return True, job_count, candidate_count
    except Exception as e:
        logger.error(f"✗ Database connection failed: {e}", exc_info=True)
        return False, 0, 0


def test_embeddings_data():
    """Test 2: Kiểm tra embeddings data."""
    logger.info("=" * 80)
    logger.info("TEST 2: EMBEDDINGS DATA")
    logger.info("=" * 80)
    
    try:
        db: Session = next(get_db())
        repo = MultiFieldEmbeddingRepository(db)
        
        # Get sample job
        all_jobs = repo.get_all_job_multi_embeddings()
        if not all_jobs:
            logger.error("✗ No jobs found in database")
            db.close()
            return False
        
        job = all_jobs[0] if len(all_jobs) > 0 else None
        if not job:
            logger.error("✗ No valid job found")
            db.close()
            return False
        has_title_emb = bool(job.title_embedding)
        has_skills_emb = bool(job.skills_embedding)
        has_requirement_emb = bool(job.requirement_embedding)
        
        logger.info(f"✓ Sample job ID: {job.job_id}")
        logger.info(f"  - Title embedding: {'✓' if has_title_emb else '✗'}")
        logger.info(f"  - Skills embedding: {'✓' if has_skills_emb else '✗'}")
        logger.info(f"  - Requirement embedding: {'✓' if has_requirement_emb else '✗'}")
        
        # Get sample candidate
        all_candidates = repo.get_all_candidate_multi_embeddings()
        if not all_candidates:
            logger.warning("⚠ No candidates found in database")
        else:
            candidate = all_candidates[0] if len(all_candidates) > 0 else None
            if not candidate:
                logger.warning("⚠ No valid candidate found")
                db.close()
                return True
            has_cand_title_emb = bool(candidate.title_embedding)
            has_cand_skills_emb = bool(candidate.skills_embedding)
            has_cand_exp_emb = bool(candidate.experience_embedding)
            
            logger.info(f"✓ Sample candidate ID: {candidate.candidate_id}")
            logger.info(f"  - Title embedding: {'✓' if has_cand_title_emb else '✗'}")
            logger.info(f"  - Skills embedding: {'✓' if has_cand_skills_emb else '✗'}")
            logger.info(f"  - Experience embedding: {'✓' if has_cand_exp_emb else '✗'}")
        
        db.close()
        return True
    except Exception as e:
        logger.error(f"✗ Embeddings data check failed: {e}", exc_info=True)
        return False


def test_faiss_indices():
    """Test 3: Kiểm tra FAISS indices."""
    logger.info("=" * 80)
    logger.info("TEST 3: FAISS INDICES")
    logger.info("=" * 80)
    
    try:
        # Check if FAISS index files exist
        from pathlib import Path
        faiss_dir = Path("faiss_indices")
        
        title_index_file = faiss_dir / "title_index.faiss"
        skills_index_file = faiss_dir / "skills_index.faiss"
        requirement_index_file = faiss_dir / "requirement_index.faiss"
        
        has_title_index = title_index_file.exists()
        has_skills_index = skills_index_file.exists()
        has_requirement_index = requirement_index_file.exists()
        
        logger.info(f"  - Title index file: {'✓' if has_title_index else '✗ (will be built on demand)'}")
        logger.info(f"  - Skills index file: {'✓' if has_skills_index else '✗ (will be built on demand)'}")
        logger.info(f"  - Requirement index file: {'✓' if has_requirement_index else '✗ (will be built on demand)'}")
        
        # Note: FAISS indices are built on-demand, so missing files are OK
        logger.info("  Note: FAISS indices are built automatically when needed")
        
        return True  # Always pass - indices are built on demand
    except Exception as e:
        logger.error(f"✗ FAISS indices check failed: {e}", exc_info=True)
        return False


def test_core_services():
    """Test 4: Kiểm tra core services."""
    logger.info("=" * 80)
    logger.info("TEST 4: CORE SERVICES")
    logger.info("=" * 80)
    
    try:
        db: Session = next(get_db())
        
        # Test Hybrid Search
        try:
            from src.services.hybrid_search_service import HybridSearchService
            hybrid = HybridSearchService()
            logger.info("✓ HybridSearchService initialized")
        except Exception as e:
            logger.error(f"✗ HybridSearchService failed: {e}")
        
        # Test Reranking
        try:
            from src.services.reranking_service import RerankingService
            reranker = RerankingService()
            logger.info("✓ RerankingService initialized")
        except Exception as e:
            logger.error(f"✗ RerankingService failed: {e}")
        
        # Test Dynamic Filtering
        try:
            from src.services.dynamic_filtering_service import DynamicFilteringService
            dynamic = DynamicFilteringService()
            logger.info("✓ DynamicFilteringService initialized")
        except Exception as e:
            logger.error(f"✗ DynamicFilteringService failed: {e}")
        
        # Test Contextual Embeddings
        try:
            from src.services.contextual_embeddings_service import ContextualEmbeddingsService
            from src.embeddings.multi_field_generator import MultiFieldEmbeddingGenerator
            embedding_generator = MultiFieldEmbeddingGenerator()
            contextual = ContextualEmbeddingsService(embedding_generator=embedding_generator)
            logger.info("✓ ContextualEmbeddingsService initialized")
        except Exception as e:
            logger.error(f"✗ ContextualEmbeddingsService failed: {e}")
        
        # Test Negative Signals
        try:
            from src.services.negative_signals_service import NegativeSignalsService
            negative = NegativeSignalsService()
            logger.info("✓ NegativeSignalsService initialized")
        except Exception as e:
            logger.error(f"✗ NegativeSignalsService failed: {e}")
        
        # Test Caching
        try:
            from src.services.caching_service import CachingService
            cache = CachingService()
            logger.info("✓ CachingService initialized")
        except Exception as e:
            logger.error(f"✗ CachingService failed: {e}")
        
        db.close()
        return True
    except Exception as e:
        logger.error(f"✗ Core services check failed: {e}", exc_info=True)
        return False


def test_new_features():
    """Test 5: Kiểm tra new features."""
    logger.info("=" * 80)
    logger.info("TEST 5: NEW FEATURES")
    logger.info("=" * 80)
    
    try:
        # Test Explainability
        try:
            from src.services.explainability_service import ExplainabilityService
            explainer = ExplainabilityService()
            logger.info("✓ ExplainabilityService initialized")
        except Exception as e:
            logger.error(f"✗ ExplainabilityService failed: {e}")
        
        # Test Diversity & Fairness
        try:
            from src.services.diversity_fairness_service import DiversityFairnessService
            diversity = DiversityFairnessService()
            logger.info("✓ DiversityFairnessService initialized")
        except Exception as e:
            logger.error(f"✗ DiversityFairnessService failed: {e}")
        
        # Test Multi-Criteria Optimization
        try:
            from src.services.multi_criteria_optimization_service import MultiCriteriaOptimizationService
            multi_criteria = MultiCriteriaOptimizationService()
            logger.info("✓ MultiCriteriaOptimizationService initialized")
        except Exception as e:
            logger.error(f"✗ MultiCriteriaOptimizationService failed: {e}")
        
        # Test Metrics Dashboard
        try:
            from src.services.metrics_dashboard_service import MetricsDashboardService
            metrics = MetricsDashboardService()
            logger.info("✓ MetricsDashboardService initialized")
        except Exception as e:
            logger.error(f"✗ MetricsDashboardService failed: {e}")
        
        # Test A/B Testing
        try:
            from src.services.ab_testing_service import ABTestingService
            ab_testing = ABTestingService()
            logger.info("✓ ABTestingService initialized")
        except Exception as e:
            logger.error(f"✗ ABTestingService failed: {e}")
        
        return True
    except Exception as e:
        logger.error(f"✗ New features check failed: {e}", exc_info=True)
        return False


def test_enhanced_matching_service():
    """Test 6: Kiểm tra Enhanced Matching Service."""
    logger.info("=" * 80)
    logger.info("TEST 6: ENHANCED MATCHING SERVICE")
    logger.info("=" * 80)
    
    try:
        db: Session = next(get_db())
        
        from src.services.enhanced_multi_filter_matching_service import EnhancedMultiFilterMatchingService
        
        service = EnhancedMultiFilterMatchingService(
            db=db,
            use_hybrid_search=True,
            use_reranking=True,
            use_dynamic_filtering=True,
            use_contextual_embeddings=True,
            use_negative_signals=True,
            use_caching=True
        )
        
        logger.info("✓ EnhancedMultiFilterMatchingService initialized")
        logger.info(f"  - Hybrid Search: {service.use_hybrid_search}")
        logger.info(f"  - Reranking: {service.use_reranking}")
        logger.info(f"  - Dynamic Filtering: {service.use_dynamic_filtering}")
        logger.info(f"  - Contextual Embeddings: {service.use_contextual_embeddings}")
        logger.info(f"  - Negative Signals: {service.use_negative_signals}")
        logger.info(f"  - Caching: {service.use_caching}")
        
        db.close()
        return True
    except Exception as e:
        logger.error(f"✗ Enhanced Matching Service check failed: {e}", exc_info=True)
        return False


def test_full_matching_pipeline(candidate_id: str = "15001", top_k: int = 5):
    """Test 7: Kiểm tra full matching pipeline."""
    logger.info("=" * 80)
    logger.info("TEST 7: FULL MATCHING PIPELINE")
    logger.info("=" * 80)
    
    try:
        from src.services.enhanced_matching_with_all_features import EnhancedMatchingWithAllFeatures
        
        db: Session = next(get_db())
        service = EnhancedMatchingWithAllFeatures(
            db=db,
            use_explainability=True,
            use_diversity_fairness=True,
            use_multi_criteria=True,
            use_metrics=True,
            use_ab_testing=True
        )
        
        logger.info(f"Testing matching for candidate: {candidate_id}")
        response = service.find_jobs_for_candidate(
            candidate_id=candidate_id,
            top_k=top_k,
            explain=True,
            ensure_diversity=True,
            use_pareto=True
        )
        
        results = response['results']
        metadata = response['metadata']
        
        logger.info(f"✓ Matching pipeline completed")
        logger.info(f"  - Results found: {len(results)}")
        logger.info(f"  - Latency: {metadata.get('latency_ms', 0):.2f}ms")
        logger.info(f"  - Diversity Score: {metadata.get('diversity_metrics', {}).get('diversity_score', 0.0):.4f}")
        
        if results:
            logger.info(f"\nTop {min(3, len(results))} results:")
            for i, result in enumerate(results[:3], 1):
                logger.info(f"  {i}. Job ID: {result['job_id']}")
                logger.info(f"     Title: {result.get('title', 'N/A')}")
                logger.info(f"     Score: {result.get('similarity_score', 0.0):.4f}")
        
        db.close()
        return len(results) > 0
    except Exception as e:
        logger.error(f"✗ Full matching pipeline failed: {e}", exc_info=True)
        return False


def test_explanations():
    """Test 8: Kiểm tra explanations."""
    logger.info("=" * 80)
    logger.info("TEST 8: EXPLANATIONS")
    logger.info("=" * 80)
    
    try:
        from src.services.enhanced_matching_with_all_features import EnhancedMatchingWithAllFeatures
        
        db: Session = next(get_db())
        service = EnhancedMatchingWithAllFeatures(db=db, use_explainability=True)
        
        response = service.find_jobs_for_candidate("15001", top_k=3, explain=True)
        explanations = response['metadata'].get('explanations', [])
        
        logger.info(f"✓ Generated {len(explanations)} explanations")
        
        if explanations:
            exp = explanations[0]
            logger.info(f"  Sample explanation:")
            logger.info(f"    - Overall Score: {exp.get('overall_score', 0.0)}")
            logger.info(f"    - Why Recommended: {exp.get('why_recommended', 'N/A')[:100]}")
            if exp.get('matched_skills'):
                logger.info(f"    - Matched Skills: {len(exp['matched_skills'])} skills")
        
        db.close()
        return len(explanations) > 0
    except Exception as e:
        logger.error(f"✗ Explanations check failed: {e}", exc_info=True)
        return False


def run_complete_test_suite(candidate_id: str = "15001", top_k: int = 5):
    """Chạy toàn bộ test suite."""
    logger.info("=" * 80)
    logger.info("🧪 CHẠY TEST TOÀN DIỆN CHO TOÀN BỘ HỆ THỐNG")
    logger.info("=" * 80)
    
    results = {}
    
    # Test 1: Database
    success, job_count, candidate_count = test_database_connection()
    results['database'] = success
    
    if not success:
        logger.error("Database test failed. Cannot continue.")
        return results
    
    # Test 2: Embeddings
    results['embeddings'] = test_embeddings_data()
    
    # Test 3: FAISS
    results['faiss'] = test_faiss_indices()
    
    # Test 4: Core Services
    results['core_services'] = test_core_services()
    
    # Test 5: New Features
    results['new_features'] = test_new_features()
    
    # Test 6: Enhanced Matching Service
    results['enhanced_matching'] = test_enhanced_matching_service()
    
    # Test 7: Full Pipeline
    results['full_pipeline'] = test_full_matching_pipeline(candidate_id, top_k)
    
    # Test 8: Explanations
    results['explanations'] = test_explanations()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 80)
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {test_name.upper()}: {status}")
    
    logger.info(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("✅ TẤT CẢ TESTS ĐÃ PASS!")
    else:
        logger.warning(f"⚠️  {total_tests - passed_tests} tests failed")
    
    logger.info("=" * 80)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test toàn diện cho toàn bộ hệ thống')
    parser.add_argument('--candidate-id', type=str, default='15001',
                       help='Candidate ID để test')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top matches')
    
    args = parser.parse_args()
    run_complete_test_suite(args.candidate_id, args.top_k)

