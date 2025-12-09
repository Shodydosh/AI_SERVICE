"""Script chạy toàn bộ hệ thống với enhanced features."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import argparse
from pathlib import Path
from sqlalchemy.orm import Session
from src.database.connection import get_db

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def init_database():
    """Bước 1: Khởi tạo database tables."""
    logger.info("=" * 80)
    logger.info("BƯỚC 1: KHỞI TẠO DATABASE TABLES")
    logger.info("=" * 80)
    
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


def process_embeddings(jd_file: str = None, candidate_file: str = None, batch_size: int = 50):
    """Bước 2: Process embeddings cho JD và Candidates."""
    logger.info("=" * 80)
    logger.info("BƯỚC 2: PROCESS EMBEDDINGS")
    logger.info("=" * 80)
    
    try:
        from src.database.connection import get_db
        from src.services.multi_field_embedding_service import MultiFieldEmbeddingService
        
        # Process JD
        if jd_file and Path(jd_file).exists():
            logger.info(f"Processing Job Descriptions from: {jd_file}")
            db: Session = next(get_db())
            try:
                service = MultiFieldEmbeddingService(db)
                total_jobs = service.process_jd_dataset(
                    file_path=jd_file,
                    file_type="csv",
                    batch_size=batch_size
                )
                logger.info(f"✓ Processed {total_jobs} Job Descriptions!")
            finally:
                db.close()
        elif jd_file:
            logger.warning(f"JD file not found: {jd_file}")
        
        # Process Candidates
        if candidate_file and Path(candidate_file).exists():
            logger.info(f"Processing Candidates from: {candidate_file}")
            db: Session = next(get_db())
            try:
                service = MultiFieldEmbeddingService(db)
                total_candidates = service.process_candidate_dataset(
                    file_path=candidate_file,
                    file_type="csv",
                    batch_size=batch_size
                )
                logger.info(f"✓ Processed {total_candidates} Candidates!")
            finally:
                db.close()
        elif candidate_file:
            logger.warning(f"Candidate file not found: {candidate_file}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Error processing embeddings: {e}", exc_info=True)
        return False


def test_enhanced_matching(candidate_id: str = "15001", top_k: int = 10):
    """Bước 3: Test enhanced matching với tất cả features."""
    logger.info("=" * 80)
    logger.info("BƯỚC 3: TEST ENHANCED MATCHING (ALL FEATURES)")
    logger.info("=" * 80)
    
    try:
        from src.services.enhanced_matching_with_all_features import EnhancedMatchingWithAllFeatures
        
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
            
            # Display results
            logger.info(f"\n📊 Results: {len(results)} jobs found")
            logger.info(f"   Latency: {metadata.get('latency_ms', 0):.2f}ms")
            logger.info(f"   Diversity Score: {metadata.get('diversity_metrics', {}).get('diversity_score', 0.0):.4f}")
            
            # Display top 5 với explanations
            logger.info("\n" + "=" * 80)
            logger.info("TOP 5 RECOMMENDATIONS:")
            logger.info("=" * 80)
            
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
                
                # Explanation
                explanations = metadata.get('explanations', [])
                if explanations and i <= len(explanations):
                    exp = explanations[i-1]
                    logger.info(f"   Explanation:")
                    logger.info(f"     - Why: {exp.get('why_recommended', 'N/A')}")
                    if exp.get('matched_skills'):
                        logger.info(f"     - Matched Skills: {exp['matched_skills'][:5]}")
            
            logger.info("\n" + "=" * 80)
            logger.info("✓ Enhanced matching test completed successfully!")
            logger.info("=" * 80)
            
            return True
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"✗ Error testing matching: {e}", exc_info=True)
        return False


def run_complete_system(
    skip_init: bool = False,
    skip_processing: bool = False,
    jd_file: str = None,
    candidate_file: str = None,
    batch_size: int = 50,
    test_candidate_id: str = "15001",
    test_top_k: int = 10
):
    """
    Chạy toàn bộ hệ thống.
    
    Args:
        skip_init: Skip database initialization
        skip_processing: Skip embedding processing
        jd_file: Path to JD CSV file
        candidate_file: Path to Candidate CSV file
        batch_size: Batch size for processing
        test_candidate_id: Candidate ID for testing
        test_top_k: Number of top matches for testing
    """
    logger.info("=" * 80)
    logger.info("🚀 CHẠY TOÀN BỘ HỆ THỐNG VỚI ENHANCED FEATURES")
    logger.info("=" * 80)
    
    success = True
    
    # Bước 1: Init database
    if not skip_init:
        if not init_database():
            logger.error("Failed to initialize database. Exiting.")
            return False
    else:
        logger.info("Skipping database initialization...")
    
    # Bước 2: Process embeddings
    if not skip_processing:
        # Auto-detect files if not provided
        if not jd_file:
            jd_file = "data/filtered/jds_with_skills.csv"
            if not Path(jd_file).exists():
                jd_file = "data/raw/job_data.csv"
        
        if not candidate_file:
            candidate_file = "data/filtered/candidates_with_skills.csv"
            if not Path(candidate_file).exists():
                candidate_file = "data/raw/candidates_dataset.csv"
        
        if not process_embeddings(jd_file, candidate_file, batch_size):
            logger.warning("Embedding processing had errors, but continuing...")
    else:
        logger.info("Skipping embedding processing...")
    
    # Bước 3: Test enhanced matching
    if not test_enhanced_matching(test_candidate_id, test_top_k):
        logger.error("Enhanced matching test failed.")
        success = False
    
    # Summary
    logger.info("\n" + "=" * 80)
    if success:
        logger.info("✅ TOÀN BỘ HỆ THỐNG CHẠY THÀNH CÔNG!")
    else:
        logger.info("⚠️  HỆ THỐNG CHẠY VỚI MỘT SỐ LỖI")
    logger.info("=" * 80)
    
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chạy toàn bộ hệ thống với enhanced features')
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
    
    run_complete_system(
        skip_init=args.skip_init,
        skip_processing=args.skip_processing,
        jd_file=args.jd_file,
        candidate_file=args.candidate_file,
        batch_size=args.batch_size,
        test_candidate_id=args.test_candidate_id,
        test_top_k=args.test_top_k
    )

