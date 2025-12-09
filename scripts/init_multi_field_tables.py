"""Script to initialize multi-field embedding tables only."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.connection import engine, Base, get_database_info
# Import models to register them with Base.metadata
from src.database.models import (
    JobDescriptionMultiEmbedding,
    CandidateMultiEmbedding
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def init_multi_field_tables():
    """Create multi-field embedding tables."""
    db_info = get_database_info()
    logger.info("=" * 80)
    logger.info("INITIALIZING MULTI-FIELD EMBEDDING TABLES")
    logger.info("=" * 80)
    logger.info(f"Connecting to database: {db_info['host']}:{db_info['port']}/{db_info['database']} (user: {db_info['username']})")
    logger.info("")
    
    try:
        logger.info("Creating tables:")
        logger.info("  - job_description_multi_embeddings")
        logger.info("  - candidate_multi_embeddings")
        logger.info("")
        
        # Check if using PostgreSQL
        from sqlalchemy import inspect
        inspector = inspect(engine)
        dialect_name = engine.dialect.name
        
        if dialect_name == 'postgresql':
            logger.info(f"✓ Detected PostgreSQL database")
            
            # Check if tables already exist
            existing_tables = inspector.get_table_names()
            if 'job_description_multi_embeddings' in existing_tables:
                logger.info("  ⚠ Table 'job_description_multi_embeddings' already exists")
            if 'candidate_multi_embeddings' in existing_tables:
                logger.info("  ⚠ Table 'candidate_multi_embeddings' already exists")
        
        Base.metadata.create_all(bind=engine, tables=[
            JobDescriptionMultiEmbedding.__table__,
            CandidateMultiEmbedding.__table__
        ])
        
        logger.info("✓ Multi-field embedding tables created/verified successfully!")
        logger.info("")
        logger.info("You can now run:")
        logger.info("  python scripts/process_multi_field_embeddings.py --process-all")
        return True
    except Exception as e:
        logger.error(f"✗ Error creating tables: {e}", exc_info=True)
        logger.error("")
        logger.error("Please check:")
        logger.error("1. PostgreSQL database is running")
        logger.error("2. Database connection settings in config/settings.py")
        logger.error("3. Database user has CREATE TABLE permissions")
        return False


if __name__ == "__main__":
    success = init_multi_field_tables()
    sys.exit(0 if success else 1)

