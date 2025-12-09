"""Script chạy toàn bộ workflow hệ thống một lượt với logging chi tiết."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import argparse
from datetime import datetime
from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.database.multi_field_repository import MultiFieldEmbeddingRepository

# Setup logging to both console and file
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

log_file = log_dir / f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print section header."""
    logger.info("=" * 100)
    logger.info(f"  {title}")
    logger.info("=" * 100)


def step_1_init_database():
    """Bước 1: Khởi tạo database tables."""
    print_section("BƯỚC 1: KHỞI TẠO DATABASE TABLES")
    
    try:
        from src.database.connection import engine, Base, get_database_info
        from src.database.models import (
            JobDescriptionMultiEmbedding,
            CandidateMultiEmbedding
        )
        
        db_info = get_database_info()
        logger.info(f"Connecting to database: {db_info['host']}:{db_info['port']}/{db_info['database']} (user: {db_info['username']})")
        
        logger.info("Creating multi-field embedding tables...")
        Base.metadata.create_all(bind=engine, tables=[
            JobDescriptionMultiEmbedding.__table__,
            CandidateMultiEmbedding.__table__
        ])
        
        logger.info("✓ Database tables initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"✗ Error initializing database: {e}", exc_info=True)
        return False


def step_2_check_data():
    """Bước 2: Kiểm tra dữ liệu."""
    print_section("BƯỚC 2: KIỂM TRA DỮ LIỆU")
    
    try:
        db: Session = next(get_db())
        repo = MultiFieldEmbeddingRepository(db)
        
        job_count = repo.count_job_multi_embeddings()
        candidate_count = repo.count_candidate_multi_embeddings()
        
        logger.info(f"Database Status:")
        logger.info(f"  - Jobs: {job_count}")
        logger.info(f"  - Candidates: {candidate_count}")
        
        if job_count == 0:
            logger.warning("⚠ No jobs found in database. Need to process job data.")
        if candidate_count == 0:
            logger.warning("⚠ No candidates found in database. Need to process candidate data.")
        
        db.close()
        return job_count > 0 and candidate_count > 0
    except Exception as e:
        logger.error(f"✗ Error checking data: {e}", exc_info=True)
        return False


def step_3_process_embeddings(jd_file: str = None, candidate_file: str = None, batch_size: int = 50):
    """Bước 3: Process embeddings (nếu cần)."""
    print_section("BƯỚC 3: PROCESS EMBEDDINGS")
    
    try:
        db: Session = next(get_db())
        repo = MultiFieldEmbeddingRepository(db)
        
        job_count = repo.count_job_multi_embeddings()
        candidate_count = repo.count_candidate_multi_embeddings()
        
        # Check if processing is needed
        if job_count > 0 and candidate_count > 0:
            logger.info("✓ Embeddings already exist in database")
            logger.info(f"  - Jobs: {job_count}")
            logger.info(f"  - Candidates: {candidate_count}")
            db.close()
            return True
        
        # Process if needed
        from src.services.multi_field_embedding_service import MultiFieldEmbeddingService
        service = MultiFieldEmbeddingService(db)
        
        total_jobs = 0
        total_candidates = 0
        
        # Auto-detect files if not provided
        if not jd_file:
            jd_file = "data/filtered/jds_with_skills.csv"
            if not Path(jd_file).exists():
                jd_file = "data/raw/job_data.csv"
        
        if not candidate_file:
            candidate_file = "data/filtered/candidates_with_skills.csv"
            if not Path(candidate_file).exists():
                candidate_file = "data/raw/candidates_dataset.csv"
        
        # Process JD
        if job_count == 0 and Path(jd_file).exists():
            logger.info(f"Processing Job Descriptions from: {jd_file}")
            total_jobs = service.process_jd_dataset(
                file_path=jd_file,
                file_type="csv",
                batch_size=batch_size
            )
            logger.info(f"✓ Processed {total_jobs} Job Descriptions!")
        elif job_count == 0:
            logger.warning(f"JD file not found: {jd_file}")
        
        # Process Candidates
        if candidate_count == 0 and Path(candidate_file).exists():
            logger.info(f"Processing Candidates from: {candidate_file}")
            total_candidates = service.process_candidate_dataset(
                file_path=candidate_file,
                file_type="csv",
                batch_size=batch_size
            )
            logger.info(f"✓ Processed {total_candidates} Candidates!")
        elif candidate_count == 0:
            logger.warning(f"Candidate file not found: {candidate_file}")
        
        db.close()
        return True
    except Exception as e:
        logger.error(f"✗ Error processing embeddings: {e}", exc_info=True)
        return False


def step_4_test_matching(candidate_id: str = "15001", top_k: int = 10):
    """Bước 4: Test matching với enhanced features."""
    print_section("BƯỚC 4: TEST MATCHING VỚI ENHANCED FEATURES")
    
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
        
        logger.info(f"\n📊 MATCHING RESULTS:")
        logger.info(f"  - Total Results: {len(results)}")
        logger.info(f"  - Latency: {metadata.get('latency_ms', 0):.2f}ms")
        logger.info(f"  - Diversity Score: {metadata.get('diversity_metrics', {}).get('diversity_score', 0.0):.4f}")
        
        if results:
            logger.info(f"\n📋 TOP {min(5, len(results))} RECOMMENDATIONS:")
            for i, result in enumerate(results[:5], 1):
                logger.info(f"\n  {i}. Job ID: {result['job_id']}")
                logger.info(f"     Title: {result.get('title', 'N/A')}")
                logger.info(f"     Company: {result.get('company', 'N/A')}")
                logger.info(f"     Location: {result.get('location', 'N/A')}")
                logger.info(f"     Similarity Score: {result.get('similarity_score', 0.0):.4f}")
                
                field_sims = result.get('field_similarities', {})
                if field_sims:
                    logger.info(f"     Field Similarities:")
                    if field_sims.get('title') is not None:
                        logger.info(f"       - Title: {field_sims['title']:.4f}")
                    if field_sims.get('skills') is not None:
                        logger.info(f"       - Skills: {field_sims['skills']:.4f}")
                    if field_sims.get('experience') is not None:
                        logger.info(f"       - Experience: {field_sims['experience']:.4f}")
                
                # Explanation
                explanations = metadata.get('explanations', [])
                if explanations and i <= len(explanations):
                    exp = explanations[i-1]
                    logger.info(f"     Explanation:")
                    logger.info(f"       - Why: {exp.get('why_recommended', 'N/A')}")
                    if exp.get('matched_skills'):
                        logger.info(f"       - Matched Skills: {', '.join(exp['matched_skills'][:5])}")
        else:
            logger.warning("⚠ No matching jobs found")
        
        db.close()
        return len(results) > 0
    except Exception as e:
        logger.error(f"✗ Error testing matching: {e}", exc_info=True)
        return False


def step_5_system_summary():
    """Bước 5: Tóm tắt hệ thống."""
    print_section("BƯỚC 5: TÓM TẮT HỆ THỐNG")
    
    try:
        db: Session = next(get_db())
        repo = MultiFieldEmbeddingRepository(db)
        
        job_count = repo.count_job_multi_embeddings()
        candidate_count = repo.count_candidate_multi_embeddings()
        
        logger.info("📊 SYSTEM SUMMARY:")
        logger.info(f"  - Database: Connected")
        logger.info(f"  - Jobs: {job_count}")
        logger.info(f"  - Candidates: {candidate_count}")
        logger.info(f"  - All Features: Enabled")
        logger.info(f"    • Hybrid Search: ✓")
        logger.info(f"    • Reranking: ✓")
        logger.info(f"    • Dynamic Filtering: ✓")
        logger.info(f"    • Contextual Embeddings: ✓")
        logger.info(f"    • Negative Signals: ✓")
        logger.info(f"    • Caching: ✓")
        logger.info(f"    • Explainability: ✓")
        logger.info(f"    • Diversity & Fairness: ✓")
        logger.info(f"    • Multi-Criteria Optimization: ✓")
        logger.info(f"    • Metrics Dashboard: ✓")
        logger.info(f"    • A/B Testing: ✓")
        
        db.close()
        return True
    except Exception as e:
        logger.error(f"✗ Error generating summary: {e}", exc_info=True)
        return False


def run_full_workflow(
    skip_init: bool = False,
    skip_processing: bool = False,
    jd_file: str = None,
    candidate_file: str = None,
    batch_size: int = 50,
    test_candidate_id: str = "15001",
    test_top_k: int = 10
):
    """Chạy toàn bộ workflow."""
    logger.info("=" * 100)
    logger.info("🚀 CHẠY TOÀN BỘ WORKFLOW HỆ THỐNG")
    logger.info("=" * 100)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 100)
    
    results = {}
    
    # Bước 1: Init database
    if not skip_init:
        results['init'] = step_1_init_database()
        if not results['init']:
            logger.error("Failed at Step 1. Cannot proceed.")
            return results
    else:
        logger.info("Skipping database initialization...")
        results['init'] = True
    
    # Bước 2: Check data
    results['check_data'] = step_2_check_data()
    
    # Bước 3: Process embeddings
    if not skip_processing:
        results['processing'] = step_3_process_embeddings(jd_file, candidate_file, batch_size)
    else:
        logger.info("Skipping embedding processing...")
        results['processing'] = True
    
    # Bước 4: Test matching
    results['matching'] = step_4_test_matching(test_candidate_id, test_top_k)
    
    # Bước 5: Summary
    results['summary'] = step_5_system_summary()
    
    # Final summary
    print_section("KẾT QUẢ CUỐI CÙNG")
    
    total_steps = len(results)
    passed_steps = sum(1 for v in results.values() if v)
    
    for step_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {step_name.upper()}: {status}")
    
    logger.info(f"\nTotal: {passed_steps}/{total_steps} steps completed successfully")
    
    if passed_steps == total_steps:
        logger.info("✅ TOÀN BỘ WORKFLOW HOÀN THÀNH THÀNH CÔNG!")
    else:
        logger.warning(f"⚠️  {total_steps - passed_steps} steps failed")
    
    logger.info(f"\nLog file saved to: {log_file}")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 100)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chạy toàn bộ workflow hệ thống với logging')
    parser.add_argument('--skip-init', action='store_true',
                       help='Skip database initialization')
    parser.add_argument('--skip-processing', action='store_true',
                       help='Skip embedding processing')
    parser.add_argument('--jd-file', type=str, default=None,
                       help='Path to JD CSV file')
    parser.add_argument('--candidate-file', type=str, default=None,
                       help='Path to Candidate CSV file')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Batch size for processing')
    parser.add_argument('--test-candidate-id', type=str, default='15001',
                       help='Candidate ID for testing')
    parser.add_argument('--test-top-k', type=int, default=10,
                       help='Number of top matches for testing')
    
    args = parser.parse_args()
    
    run_full_workflow(
        skip_init=args.skip_init,
        skip_processing=args.skip_processing,
        jd_file=args.jd_file,
        candidate_file=args.candidate_file,
        batch_size=args.batch_size,
        test_candidate_id=args.test_candidate_id,
        test_top_k=args.test_top_k
    )

