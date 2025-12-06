"""Migration script to add embedding_timestamp and content_hash columns."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from src.database.connection import SessionLocal, engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_migration():
    """Add embedding_timestamp and content_hash columns to embedding tables."""
    db = SessionLocal()
    
    try:
        logger.info("Starting migration: Adding embedding_timestamp and content_hash columns")
        
        # Add columns to candidate_multi_embeddings
        logger.info("Adding columns to candidate_multi_embeddings...")
        db.execute(text("""
            ALTER TABLE candidate_multi_embeddings
            ADD COLUMN IF NOT EXISTS embedding_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW();
        """))
        
        db.execute(text("""
            ALTER TABLE candidate_multi_embeddings
            ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64);
        """))
        
        # Add index
        db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_candidate_multi_embedding_timestamp
            ON candidate_multi_embeddings(embedding_timestamp);
        """))
        
        # Update existing records to set embedding_timestamp = created_at
        db.execute(text("""
            UPDATE candidate_multi_embeddings
            SET embedding_timestamp = created_at
            WHERE embedding_timestamp IS NULL;
        """))
        
        # Add columns to job_description_multi_embeddings
        logger.info("Adding columns to job_description_multi_embeddings...")
        db.execute(text("""
            ALTER TABLE job_description_multi_embeddings
            ADD COLUMN IF NOT EXISTS embedding_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW();
        """))
        
        db.execute(text("""
            ALTER TABLE job_description_multi_embeddings
            ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64);
        """))
        
        # Add index
        db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_job_multi_embedding_timestamp
            ON job_description_multi_embeddings(embedding_timestamp);
        """))
        
        # Update existing records
        db.execute(text("""
            UPDATE job_description_multi_embeddings
            SET embedding_timestamp = created_at
            WHERE embedding_timestamp IS NULL;
        """))
        
        db.commit()
        logger.info("✓ Migration completed successfully")
        
    except Exception as e:
        db.rollback()
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        db.close()


if __name__ == '__main__':
    run_migration()

