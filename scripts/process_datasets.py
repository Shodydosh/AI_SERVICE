"""Script to process JD and candidate datasets."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.services.embedding_service import EmbeddingService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_datasets(jd_file: str = None, candidate_file: str = None, file_type: str = "csv"):
    """Process JD and/or candidate datasets."""
    db: Session = SessionLocal()
    try:
        service = EmbeddingService(db)
        
        if jd_file:
            logger.info(f"Processing JD dataset: {jd_file}")
            count = service.process_jd_dataset(jd_file, file_type)
            logger.info(f"Processed {count} job descriptions")
        
        if candidate_file:
            logger.info(f"Processing candidate dataset: {candidate_file}")
            count = service.process_candidate_dataset(candidate_file, file_type)
            logger.info(f"Processed {count} candidates")
    
    except Exception as e:
        logger.error(f"Error processing datasets: {e}")
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JD and candidate datasets")
    parser.add_argument("--jd-file", type=str, help="Path to JD dataset file")
    parser.add_argument("--candidate-file", type=str, help="Path to candidate dataset file")
    parser.add_argument("--file-type", type=str, default="csv", choices=["csv", "json"], help="File type")
    
    args = parser.parse_args()
    
    if not args.jd_file and not args.candidate_file:
        parser.error("At least one of --jd-file or --candidate-file must be provided")
    
    process_datasets(args.jd_file, args.candidate_file, args.file_type)

