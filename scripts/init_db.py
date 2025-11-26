"""Script to initialize the database."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.connection import engine, Base, get_database_info
# Import models to register them with Base.metadata
from src.database.models import (
    JobDescriptionEmbedding, 
    CandidateEmbedding,
    ProcessedCandidateRecommendation
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_database():
    """Create all database tables."""
    db_info = get_database_info()
    logger.info(f"Connecting to database: {db_info['host']}:{db_info['port']}/{db_info['database']} (user: {db_info['username']})")
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully!")


if __name__ == "__main__":
    init_database()

