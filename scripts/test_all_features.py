"""Test script cho tất cả features mới."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.services.enhanced_matching_with_all_features import EnhancedMatchingWithAllFeatures

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_all_features(candidate_id: str = "15001", top_k: int = 10):
    """Test tất cả features."""
    logger.info("=" * 80)
    logger.info("TESTING ALL FEATURES")
    logger.info("=" * 80)
    
    db: Session = next(get_db())
    try:
        # Initialize service với tất cả features
        service = EnhancedMatchingWithAllFeatures(
            db=db,
            # Core features
            use_hybrid_search=True,
            use_reranking=True,
            use_dynamic_filtering=True,
            use_contextual_embeddings=True,
            use_negative_signals=True,
            use_caching=True,
            # New features
            use_explainability=True,
            use_diversity_fairness=True,
            use_multi_criteria=True,
            use_metrics=True,
            use_ab_testing=True
        )
        
        # Test matching
        logger.info(f"\n🔍 Testing matching for candidate: {candidate_id}")
        response = service.find_jobs_for_candidate(
            candidate_id=candidate_id,
            top_k=top_k,
            explain=True,
            ensure_diversity=True,
            use_pareto=True
        )
        
        results = response['results']
        metadata = response['metadata']
        
        # Display results
        logger.info(f"\n📊 Results: {len(results)} jobs found")
        logger.info(f"   Latency: {metadata.get('latency_ms', 0):.2f}ms")
        logger.info(f"   Diversity Score: {metadata.get('diversity_metrics', {}).get('diversity_score', 0.0):.4f}")
        logger.info("=" * 80)
        
        # Display top 5 với explanations
        for i, result in enumerate(results[:5], 1):
            logger.info(f"\n{i}. Job ID: {result['job_id']}")
            logger.info(f"   Title: {result.get('title', 'N/A')}")
            logger.info(f"   Similarity Score: {result.get('similarity_score', 0.0):.4f}")
            
            # Field similarities
            field_sims = result.get('field_similarities', {})
            if field_sims:
                logger.info(f"   Field Similarities:")
                if field_sims.get('title'):
                    logger.info(f"     - Title: {field_sims['title']:.4f}")
                if field_sims.get('skills'):
                    logger.info(f"     - Skills: {field_sims['skills']:.4f}")
                if field_sims.get('experience'):
                    logger.info(f"     - Experience: {field_sims['experience']:.4f}")
            
            # Objective scores (multi-criteria)
            if 'objective_scores' in result:
                obj_scores = result['objective_scores']
                logger.info(f"   Objective Scores: {[round(s, 3) for s in obj_scores]}")
            
            # Explanation
            explanations = metadata.get('explanations', [])
            if explanations and i <= len(explanations):
                exp = explanations[i-1]
                logger.info(f"   Explanation:")
                logger.info(f"     - Why: {exp.get('why_recommended', 'N/A')}")
                if exp.get('matched_skills'):
                    logger.info(f"     - Matched Skills: {exp['matched_skills'][:5]}")
                if exp.get('missing_skills'):
                    logger.info(f"     - Missing Skills: {exp['missing_skills'][:3]}")
        
        # Metrics dashboard
        logger.info("\n" + "=" * 80)
        logger.info("📈 METRICS DASHBOARD")
        logger.info("=" * 80)
        
        dashboard = service.get_metrics_dashboard()
        if dashboard:
            latency = dashboard.get('latency', {})
            if latency:
                logger.info(f"Latency (p95): {latency.get('p95', 0):.2f}ms")
                logger.info(f"Latency (p99): {latency.get('p99', 0):.2f}ms")
            
            engagement = dashboard.get('engagement', {})
            if engagement:
                logger.info(f"CTR: {engagement.get('ctr', 0)*100:.2f}%")
                logger.info(f"Application Rate: {engagement.get('application_rate', 0)*100:.2f}%")
        
        # A/B Testing metrics
        logger.info("\n" + "=" * 80)
        logger.info("🧪 A/B TESTING METRICS")
        logger.info("=" * 80)
        
        ab_metrics = service.get_ab_test_metrics()
        if ab_metrics:
            for exp_name, metrics in ab_metrics.items():
                logger.info(f"\nExperiment: {exp_name}")
                logger.info(f"  Control Calls: {metrics.get('control_calls', 0)}")
                logger.info(f"  Variant Calls: {metrics.get('variant_calls', 0)}")
                logger.info(f"  Improvement: {metrics.get('improvement', 0)*100:.2f}%")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ TEST COMPLETED!")
        logger.info("=" * 80)
        
        return response
        
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
        return None
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test All Features')
    parser.add_argument('--candidate-id', type=str, default='15001',
                       help='Candidate ID to test')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of top matches')
    
    args = parser.parse_args()
    test_all_features(args.candidate_id, args.top_k)

