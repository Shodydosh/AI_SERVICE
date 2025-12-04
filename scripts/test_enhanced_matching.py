"""Test script cho Enhanced Matching System."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.services.enhanced_multi_filter_matching_service import EnhancedMultiFilterMatchingService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_enhanced_matching(candidate_id: str = "15001", top_k: int = 10):
    """Test enhanced matching service."""
    logger.info("=" * 80)
    logger.info("TESTING ENHANCED MATCHING SYSTEM")
    logger.info("=" * 80)
    
    db: Session = next(get_db())
    try:
        # Initialize enhanced service với tất cả features
        service = EnhancedMultiFilterMatchingService(
            db=db,
            use_faiss=True,
            use_hybrid_search=True,
            use_reranking=True,
            use_dynamic_filtering=True,
            use_contextual_embeddings=True,
            use_negative_signals=True,
            use_caching=True
        )
        
        # Test matching
        logger.info(f"\n🔍 Testing matching for candidate: {candidate_id}")
        results = service.find_jobs_for_candidate(candidate_id, top_k=top_k)
        
        # Display results
        logger.info(f"\n📊 Results: {len(results)} jobs found")
        logger.info("=" * 80)
        
        for i, result in enumerate(results, 1):
            logger.info(f"\n{i}. Job ID: {result['job_id']}")
            logger.info(f"   Title: {result.get('title', 'N/A')}")
            logger.info(f"   Company: {result.get('company', 'N/A')}")
            logger.info(f"   Similarity Score: {result.get('similarity_score', 0.0):.4f}")
            
            field_sims = result.get('field_similarities', {})
            if field_sims:
                logger.info(f"   Field Similarities:")
                if field_sims.get('title'):
                    logger.info(f"     - Title: {field_sims['title']:.4f}")
                if field_sims.get('skills'):
                    logger.info(f"     - Skills: {field_sims['skills']:.4f}")
                if field_sims.get('experience'):
                    logger.info(f"     - Experience: {field_sims['experience']:.4f}")
            
            # Negative signals
            if 'negative_signals' in result:
                neg_sigs = result['negative_signals']
                logger.info(f"   Negative Signals:")
                logger.info(f"     - Total Penalty: {neg_sigs.get('total_penalty', 0.0):.4f}")
                if neg_sigs.get('salary_penalty', 0) > 0:
                    logger.info(f"     - Salary Penalty: {neg_sigs['salary_penalty']:.4f}")
                if neg_sigs.get('location_penalty', 0) > 0:
                    logger.info(f"     - Location Penalty: {neg_sigs['location_penalty']:.4f}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ TEST COMPLETED!")
        logger.info("=" * 80)
        
        return results
        
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
        return []
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Enhanced Matching System')
    parser.add_argument('--candidate-id', type=str, default='15001',
                       help='Candidate ID to test')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of top matches to return')
    
    args = parser.parse_args()
    test_enhanced_matching(args.candidate_id, args.top_k)

