"""
Migration script to create Two-Tower schema.
Creates new tables without deleting old ones.
"""
import logging
from sqlalchemy.orm import Session
from src.database.connection import get_db, Base, engine
from src.database.models import (
    JobDescriptionTwoTower,
    CandidateTwoTower,
    ReindexTracking,
    Base
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_schema():
    """
    Create Two-Tower tables.
    """
    logger.info("=" * 80)
    logger.info("TWO-TOWER SCHEMA MIGRATION")
    logger.info("=" * 80)
    
    try:
        # Create tables
        logger.info("Creating Two-Tower tables...")
        Base.metadata.create_all(engine, tables=[
            JobDescriptionTwoTower.__table__,
            CandidateTwoTower.__table__,
            ReindexTracking.__table__
        ])
        
        logger.info("✓ Created job_description_two_tower table")
        logger.info("✓ Created candidate_two_tower table")
        logger.info("✓ Created reindex_tracking table")
        
        logger.info("")
        logger.info("Migration completed successfully!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Run batch_reindex_two_tower.py to build FAISS indices")
        logger.info("2. Start API server with Two-Tower routes enabled")
        
    except Exception as e:
        logger.error(f"Error during migration: {e}", exc_info=True)
        raise


def main():
    migrate_schema()


if __name__ == "__main__":
    main()

