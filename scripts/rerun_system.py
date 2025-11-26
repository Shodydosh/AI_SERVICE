"""Script to re-run embedding generation and rebuild system."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.services.embedding_service import EmbeddingService
from src.vector_search.faiss_manager import FAISSIndexManager
from src.services.precompute_service import PrecomputeService
from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def rerun_embeddings(jd_file: str = None, candidate_file: str = None):
    """Re-run embedding generation."""
    logger.info("=" * 80)
    logger.info("STEP 1: Regenerating Embeddings")
    logger.info("=" * 80)
    
    db: Session = SessionLocal()
    try:
        service = EmbeddingService(db)
        
        if jd_file:
            logger.info(f"Processing JD dataset: {jd_file}")
            count = service.process_jd_dataset(jd_file, 'csv')
            logger.info(f"✓ Generated embeddings for {count} job descriptions")
        
        if candidate_file:
            logger.info(f"Processing candidate dataset: {candidate_file}")
            count = service.process_candidate_dataset(candidate_file, 'csv')
            logger.info(f"✓ Generated embeddings for {count} candidates")
        
        logger.info("")
        return True
    except Exception as e:
        logger.error(f"✗ Error regenerating embeddings: {e}")
        return False
    finally:
        db.close()


def rebuild_faiss_indices():
    """Rebuild FAISS indices."""
    logger.info("=" * 80)
    logger.info("STEP 2: Rebuilding FAISS Indices")
    logger.info("=" * 80)
    
    db: Session = SessionLocal()
    try:
        faiss_manager = FAISSIndexManager(
            dimension=settings.EMBEDDING_DIMENSION,
            index_type="HNSW",
            index_params={
                "ef_search": 64,
                "ef_construction": 200,
                "M": 32
            },
            normalize=True
        )
        
        # Build JD index
        logger.info("Building JD index...")
        faiss_manager.build_index_from_db(db, dataset_type='jd')
        faiss_manager.save_index('indices/jd_index.faiss', dataset_type='jd')
        logger.info("✓ JD FAISS index built and saved")
        
        # Build candidate index
        logger.info("Building candidate index...")
        faiss_manager.build_index_from_db(db, dataset_type='candidate')
        faiss_manager.save_index('indices/candidate_index.faiss', dataset_type='candidate')
        logger.info("✓ Candidate FAISS index built and saved")
        
        logger.info("")
        return True
    except Exception as e:
        logger.error(f"✗ Error rebuilding FAISS indices: {e}")
        return False
    finally:
        db.close()


def precompute_recommendations():
    """Pre-compute recommendations for all candidates."""
    logger.info("=" * 80)
    logger.info("STEP 3: Pre-computing Recommendations")
    logger.info("=" * 80)
    
    db: Session = SessionLocal()
    try:
        precompute_service = PrecomputeService(db)
        results = precompute_service.precompute_all_candidates(top_k=10)
        
        logger.info("")
        logger.info(f"✓ Pre-computation complete:")
        logger.info(f"  - Total candidates: {results['total_candidates']}")
        logger.info(f"  - Processed: {results['processed']}")
        logger.info(f"  - Failed: {results['failed']}")
        logger.info(f"  - Total recommendations: {results['total_recommendations']}")
        logger.info("")
        
        return results['processed'] > 0
    except Exception as e:
        logger.error(f"✗ Error pre-computing recommendations: {e}")
        return False
    finally:
        db.close()


def main():
    """Main function to re-run system."""
    parser = argparse.ArgumentParser(
        description="Re-run embedding generation and rebuild system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Re-run everything
  python scripts/rerun_system.py --jd-file data/processed/jd_processed.csv --candidate-file data/processed/candidate_processed.csv
  
  # Re-run only embeddings
  python scripts/rerun_system.py --jd-file data/processed/jd_processed.csv --skip-faiss --skip-precompute
  
  # Rebuild FAISS only
  python scripts/rerun_system.py --skip-embeddings --skip-precompute
        """
    )
    
    parser.add_argument(
        "--jd-file",
        type=str,
        help="Path to JD processed dataset"
    )
    
    parser.add_argument(
        "--candidate-file",
        type=str,
        help="Path to candidate processed dataset"
    )
    
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding regeneration"
    )
    
    parser.add_argument(
        "--skip-faiss",
        action="store_true",
        help="Skip FAISS index rebuilding"
    )
    
    parser.add_argument(
        "--skip-precompute",
        action="store_true",
        help="Skip pre-computation"
    )
    
    args = parser.parse_args()
    
    # Check if at least one operation is enabled
    if args.skip_embeddings and args.skip_faiss and args.skip_precompute:
        parser.error("At least one operation must be enabled")
    
    success = True
    
    # Step 1: Regenerate embeddings
    if not args.skip_embeddings:
        if not args.jd_file and not args.candidate_file:
            logger.warning("⚠ No dataset files provided, skipping embedding regeneration")
        else:
            success = rerun_embeddings(
                jd_file=args.jd_file,
                candidate_file=args.candidate_file
            ) and success
    
    # Step 2: Rebuild FAISS indices
    if not args.skip_faiss:
        success = rebuild_faiss_indices() and success
    
    # Step 3: Pre-compute recommendations
    if not args.skip_precompute:
        success = precompute_recommendations() and success
    
    logger.info("=" * 80)
    if success:
        logger.info("✓ SYSTEM REBUILD COMPLETE")
    else:
        logger.error("✗ SYSTEM REBUILD FAILED")
    logger.info("=" * 80)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


