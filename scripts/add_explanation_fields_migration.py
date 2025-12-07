"""Migration script to add explanation fields to processed_candidate_recommendations table."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from src.database.connection import SessionLocal, engine


def add_explanation_fields():
    """Add explanation fields to processed_candidate_recommendations table."""
    db = SessionLocal()
    try:
        print("Adding explanation fields to processed_candidate_recommendations table...")
        
        # Check if columns already exist
        check_query = text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'processed_candidate_recommendations' 
            AND column_name IN ('rule_scores', 'embedding_scores', 'explanation_text', 'comprehensive_explanation', 'confidence_score')
        """)
        existing_columns = [row[0] for row in db.execute(check_query).fetchall()]
        
        # Add columns if they don't exist
        if 'rule_scores' not in existing_columns:
            db.execute(text("""
                ALTER TABLE processed_candidate_recommendations 
                ADD COLUMN rule_scores TEXT
            """))
            print("  ✓ Added rule_scores column")
        else:
            print("  - rule_scores column already exists")
        
        if 'embedding_scores' not in existing_columns:
            db.execute(text("""
                ALTER TABLE processed_candidate_recommendations 
                ADD COLUMN embedding_scores TEXT
            """))
            print("  ✓ Added embedding_scores column")
        else:
            print("  - embedding_scores column already exists")
        
        if 'explanation_text' not in existing_columns:
            db.execute(text("""
                ALTER TABLE processed_candidate_recommendations 
                ADD COLUMN explanation_text TEXT
            """))
            print("  ✓ Added explanation_text column")
        else:
            print("  - explanation_text column already exists")
        
        if 'comprehensive_explanation' not in existing_columns:
            db.execute(text("""
                ALTER TABLE processed_candidate_recommendations 
                ADD COLUMN comprehensive_explanation TEXT
            """))
            print("  ✓ Added comprehensive_explanation column")
        else:
            print("  - comprehensive_explanation column already exists")
        
        if 'confidence_score' not in existing_columns:
            db.execute(text("""
                ALTER TABLE processed_candidate_recommendations 
                ADD COLUMN confidence_score FLOAT
            """))
            print("  ✓ Added confidence_score column")
        else:
            print("  - confidence_score column already exists")
        
        # Add index on confidence_score if it doesn't exist
        try:
            db.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_processed_candidate_confidence 
                ON processed_candidate_recommendations(confidence_score)
            """))
            print("  ✓ Added index on confidence_score")
        except Exception as e:
            print(f"  - Index on confidence_score may already exist: {e}")
        
        db.commit()
        print("\n✓ Migration completed successfully!")
        
    except Exception as e:
        db.rollback()
        print(f"\n✗ Migration failed: {e}")
        raise
    finally:
        db.close()


if __name__ == '__main__':
    add_explanation_fields()


