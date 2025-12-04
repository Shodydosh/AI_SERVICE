"""Database Optimization Script: Indexing, partitioning, materialized views."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from sqlalchemy import text
from src.database.connection import get_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_indexes(db):
    """Create indexes for query optimization."""
    logger.info("Creating indexes...")
    
    indexes = [
        # Composite index for industry + location
        """
        CREATE INDEX IF NOT EXISTS idx_jd_industry_location 
        ON job_descriptions(industry, location);
        """,
        
        # GIN index for skills (PostgreSQL)
        """
        CREATE INDEX IF NOT EXISTS idx_candidate_skills_gin 
        ON candidates USING GIN(to_tsvector('simple', skills));
        """,
        
        # Index for title search
        """
        CREATE INDEX IF NOT EXISTS idx_jd_title_trgm 
        ON job_descriptions USING GIN(title gin_trgm_ops);
        """,
        
        # Index for candidate title
        """
        CREATE INDEX IF NOT EXISTS idx_candidate_title_trgm 
        ON candidates USING GIN(title gin_trgm_ops);
        """,
        
        # Index for embedding lookups
        """
        CREATE INDEX IF NOT EXISTS idx_jd_multi_field_embeddings_job_id 
        ON job_multi_field_embeddings(job_id);
        """,
        
        """
        CREATE INDEX IF NOT EXISTS idx_candidate_multi_field_embeddings_candidate_id 
        ON candidate_multi_field_embeddings(candidate_id);
        """
    ]
    
    for index_sql in indexes:
        try:
            db.execute(text(index_sql))
            db.commit()
            logger.info(f"✓ Created index")
        except Exception as e:
            logger.warning(f"Could not create index: {e}")
            db.rollback()


def create_materialized_views(db):
    """Create materialized views for pre-computed scores."""
    logger.info("Creating materialized views...")
    
    views = [
        # Materialized view for top job matches per candidate
        """
        CREATE MATERIALIZED VIEW IF NOT EXISTS mv_top_job_matches AS
        SELECT 
            c.candidate_id,
            j.job_id,
            j.title as job_title,
            j.company,
            j.location,
            -- Pre-computed similarity scores (example)
            0.5 as similarity_score
        FROM candidates c
        CROSS JOIN job_descriptions j
        LIMIT 1000000;  -- Limit for performance
        """,
        
        # Index on materialized view
        """
        CREATE INDEX IF NOT EXISTS idx_mv_top_job_matches_candidate_id 
        ON mv_top_job_matches(candidate_id);
        """
    ]
    
    for view_sql in views:
        try:
            db.execute(text(view_sql))
            db.commit()
            logger.info(f"✓ Created materialized view")
        except Exception as e:
            logger.warning(f"Could not create materialized view: {e}")
            db.rollback()


def analyze_tables(db):
    """Run ANALYZE on tables for query planner optimization."""
    logger.info("Analyzing tables...")
    
    tables = [
        'job_descriptions',
        'candidates',
        'job_multi_field_embeddings',
        'candidate_multi_field_embeddings'
    ]
    
    for table in tables:
        try:
            db.execute(text(f"ANALYZE {table};"))
            db.commit()
            logger.info(f"✓ Analyzed {table}")
        except Exception as e:
            logger.warning(f"Could not analyze {table}: {e}")
            db.rollback()


def main():
    """Main function."""
    logger.info("=" * 80)
    logger.info("DATABASE OPTIMIZATION")
    logger.info("=" * 80)
    
    db = next(get_db())
    try:
        # Create indexes
        create_indexes(db)
        
        # Create materialized views
        # create_materialized_views(db)  # Comment out if not needed
        
        # Analyze tables
        analyze_tables(db)
        
        logger.info("=" * 80)
        logger.info("✅ Database optimization completed!")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    main()

