"""Script test đơn giản cho hệ thống matching."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from sqlalchemy.orm import Session
from src.database.connection import get_db

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_database():
    """Test 1: Database connection."""
    logger.info("=" * 80)
    logger.info("TEST 1: DATABASE CONNECTION")
    logger.info("=" * 80)
    
    try:
        db: Session = next(get_db())
        from src.database.multi_field_repository import MultiFieldEmbeddingRepository
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


def test_embeddings():
    """Test 2: Embeddings Data."""
    logger.info("=" * 80)
    logger.info("TEST 2: EMBEDDINGS DATA")
    logger.info("=" * 80)
    
    try:
        db: Session = next(get_db())
        from src.database.multi_field_repository import MultiFieldEmbeddingRepository
        repo = MultiFieldEmbeddingRepository(db)
        
        # Get sample job
        all_jobs = repo.get_all_job_multi_embeddings()
        if not all_jobs:
            logger.warning("⚠ No jobs found in database")
            db.close()
            return False
        
        job = all_jobs[0]
        has_title_emb = bool(job.title_embedding)
        has_skills_emb = bool(job.skills_embedding)
        has_requirement_emb = bool(job.requirement_embedding)
        
        logger.info(f"✓ Sample job ID: {job.job_id}")
        logger.info(f"  - Title embedding: {'✓' if has_title_emb else '✗'}")
        logger.info(f"  - Skills embedding: {'✓' if has_skills_emb else '✗'}")
        logger.info(f"  - Requirement embedding: {'✓' if has_requirement_emb else '✗'}")
        
        # Get sample candidate
        all_candidates = repo.get_all_candidate_multi_embeddings()
        if all_candidates:
            candidate = all_candidates[0]
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
        logger.error(f"✗ Embeddings test failed: {e}", exc_info=True)
        return False


def test_rule_matcher():
    """Test 3: Rule Matcher."""
    logger.info("=" * 80)
    logger.info("TEST 3: RULE MATCHER")
    logger.info("=" * 80)
    
    try:
        from src.utils.rule_matcher import RuleMatcher
        
        matcher = RuleMatcher()
        logger.info("✓ RuleMatcher initialized")
        
        # Test với sample data
        result = matcher.evaluate_match(
            candidate_title="Python Developer",
            candidate_skills=["Python", "FastAPI", "PostgreSQL"],
            job_title="Senior Python Developer",
            job_requirements="Python, FastAPI, PostgreSQL required",
            job_description="We are looking for an experienced Python developer..."
        )
        
        logger.info(f"✓ Test match completed")
        logger.info(f"  - Final Status: {result.get('final_status', 'N/A')}")
        logger.info(f"  - Title Score: {result.get('final_title_score', 0.0):.2f}")
        logger.info(f"  - Skill Score: {result.get('skill_score', 0.0):.2f}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Rule Matcher test failed: {e}", exc_info=True)
        return False


def run_simple_test_suite(candidate_id: str = "15001", top_k: int = 5):
    """Chạy test suite đơn giản."""
    logger.info("=" * 80)
    logger.info("🧪 CHẠY TEST ĐƠN GIẢN CHO HỆ THỐNG")
    logger.info("=" * 80)
    
    results = {}
    
    # Test 1: Database
    success, job_count, candidate_count = test_database()
    results['database'] = success
    
    if not success:
        logger.error("Database test failed. Cannot continue.")
        return results
    
    # Test 2: Embeddings
    results['embeddings'] = test_embeddings()
    
    # Test 3: Rule Matcher
    results['rule_matcher'] = test_rule_matcher()
    
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Test đơn giản cho hệ thống')
    parser.add_argument('--candidate-id', type=str, default='15001',
                       help='Candidate ID để test')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top matches')
    
    args = parser.parse_args()
    run_simple_test_suite(args.candidate_id, args.top_k)

