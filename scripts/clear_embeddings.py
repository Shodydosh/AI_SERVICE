"""Script to clear all embeddings from database for regeneration."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.connection import SessionLocal
from src.database.models import JobDescriptionEmbedding, CandidateEmbedding
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clear_embeddings(clear_jd: bool = True, clear_candidates: bool = True):
    """Clear all embeddings from database.
    
    Args:
        clear_jd: Whether to clear job description embeddings
        clear_candidates: Whether to clear candidate embeddings
    """
    db = SessionLocal()
    try:
        logger.info("=" * 80)
        logger.info("CLEARING EMBEDDINGS FROM DATABASE")
        logger.info("=" * 80)
        
        if clear_jd:
            jd_count = db.query(JobDescriptionEmbedding).count()
            logger.info(f"Found {jd_count} job description embeddings")
            if jd_count > 0:
                db.query(JobDescriptionEmbedding).delete()
                logger.info(f"✓ Deleted {jd_count} job description embeddings")
        
        if clear_candidates:
            candidate_count = db.query(CandidateEmbedding).count()
            logger.info(f"Found {candidate_count} candidate embeddings")
            if candidate_count > 0:
                db.query(CandidateEmbedding).delete()
                logger.info(f"✓ Deleted {candidate_count} candidate embeddings")
        
        db.commit()
        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ All embeddings cleared successfully!")
        logger.info("=" * 80)
        logger.info("You can now regenerate embeddings with:")
        logger.info("  python scripts/generate_embeddings_from_processed.py --jd-file data/processed/job_data.csv --candidate-file data/processed/candidates_dataset.csv")
        logger.info("")
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error clearing embeddings: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clear embeddings from database")
    parser.add_argument("--jd-only", action="store_true", help="Clear only job description embeddings")
    parser.add_argument("--candidates-only", action="store_true", help="Clear only candidate embeddings")
    
    args = parser.parse_args()
    
    clear_jd = not args.candidates_only
    clear_candidates = not args.jd_only
    
    clear_embeddings(clear_jd=clear_jd, clear_candidates=clear_candidates)

