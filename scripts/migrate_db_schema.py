"""Migration script to update database schema to match models."""
import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.connection import engine
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_schema():
    """
    Update database schema to ensure TEXT fields are properly set.
    This converts any VARCHAR(200) fields that should be TEXT.
    """
    logger.info("=" * 80)
    logger.info("MIGRATING DATABASE SCHEMA")
    logger.info("=" * 80)
    
    with engine.connect() as conn:
        try:
            # Check current schema
            logger.info("Checking current schema...")
            
            # Get column types for candidate_embeddings table
            check_query = text("""
                SELECT column_name, data_type, character_maximum_length
                FROM information_schema.columns
                WHERE table_name = 'candidate_embeddings'
                AND column_name IN ('name', 'email', 'skills', 'experience', 'education', 'summary', 'resume_text')
                ORDER BY column_name;
            """)
            
            result = conn.execute(check_query)
            columns = result.fetchall()
            
            logger.info("\nCurrent column types:")
            for col in columns:
                col_name, data_type, max_length = col
                logger.info(f"  {col_name}: {data_type}({max_length or 'N/A'})")
            
            # Update schema: Convert VARCHAR to TEXT for fields that should be TEXT
            # Note: name and email should remain VARCHAR(200) per model
            migrations = []
            
            # Check if skills, experience, education, summary, resume_text need to be TEXT
            text_fields = ['skills', 'experience', 'education', 'summary', 'resume_text']
            for field in text_fields:
                field_info = next((c for c in columns if c[0] == field), None)
                if field_info:
                    col_name, data_type, max_length = field_info
                    if data_type == 'character varying' and max_length is not None:
                        logger.info(f"\nConverting {field} from VARCHAR({max_length}) to TEXT...")
                        alter_query = text(f"""
                            ALTER TABLE candidate_embeddings 
                            ALTER COLUMN {field} TYPE TEXT;
                        """)
                        conn.execute(alter_query)
                        migrations.append(field)
                        logger.info(f"✓ Converted {field} to TEXT")
            
            if migrations:
                conn.commit()
                logger.info(f"\n✓ Successfully migrated {len(migrations)} columns")
            else:
                logger.info("\n✓ Schema is already up to date")
            
            # Verify final schema
            logger.info("\nFinal column types:")
            result = conn.execute(check_query)
            columns = result.fetchall()
            for col in columns:
                col_name, data_type, max_length = col
                logger.info(f"  {col_name}: {data_type}({max_length or 'N/A'})")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"✗ Error migrating schema: {e}")
            raise
    
    logger.info("=" * 80)

if __name__ == "__main__":
    migrate_schema()

