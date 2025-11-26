"""Script to clean 'NaN' string values from candidate_embeddings table."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.connection import SessionLocal
from src.database.models import CandidateEmbedding
from sqlalchemy import update
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_nan_values():
    """Clean 'NaN' string values from database, converting them to NULL."""
    db = SessionLocal()
    try:
        logger.info("=" * 80)
        logger.info("CLEANING 'NaN' STRING VALUES FROM DATABASE")
        logger.info("=" * 80)
        
        # Count records with 'NaN' strings
        fields_to_check = ['name', 'email', 'skills', 'experience', 'education', 'summary', 'resume_text']
        nan_counts = {}
        
        for field in fields_to_check:
            count = db.query(CandidateEmbedding).filter(
                getattr(CandidateEmbedding, field) == 'NaN'
            ).count()
            nan_counts[field] = count
            logger.info(f"Found {count} records with '{field}' = 'NaN'")
        
        total_to_fix = sum(nan_counts.values())
        logger.info(f"\nTotal 'NaN' values to clean: {total_to_fix}")
        
        if total_to_fix == 0:
            logger.info("No 'NaN' values found. Database is clean!")
            return
        
        # Clean each field
        logger.info("\nCleaning 'NaN' values...")
        for field in fields_to_check:
            if nan_counts[field] > 0:
                db.execute(
                    update(CandidateEmbedding)
                    .where(getattr(CandidateEmbedding, field) == 'NaN')
                    .values(**{field: None})
                )
                logger.info(f"✓ Cleaned {nan_counts[field]} '{field}' values")
        
        # Also clean 'nan' (lowercase)
        for field in fields_to_check:
            count = db.query(CandidateEmbedding).filter(
                getattr(CandidateEmbedding, field) == 'nan'
            ).count()
            if count > 0:
                db.execute(
                    update(CandidateEmbedding)
                    .where(getattr(CandidateEmbedding, field) == 'nan')
                    .values(**{field: None})
                )
                logger.info(f"✓ Cleaned {count} '{field}' values (lowercase 'nan')")
        
        db.commit()
        logger.info("\n✓ Database cleaned successfully!")
        logger.info("=" * 80)
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error cleaning database: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    clean_nan_values()

