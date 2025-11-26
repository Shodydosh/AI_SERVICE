"""Match a candidate to top 50 jobs."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.services.matching_service import MatchingService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def match_candidate(
    candidate_id: str = None,
    candidate_file: str = None,
    candidate_index: int = 0,
    candidate_text: str = None,
    top_k: int = 50
):
    """
    Match a candidate to top jobs.
    
    Args:
        candidate_id: Candidate ID from database
        candidate_file: Path to processed candidate file
        candidate_index: Index in candidate file (if using file)
        candidate_text: Candidate information as text
        top_k: Number of top matches to return
    """
    logger.info("=" * 80)
    logger.info("CANDIDATE TO JOB MATCHING")
    logger.info("=" * 80)
    
    db: Session = SessionLocal()
    try:
        matching_service = MatchingService(db, use_faiss=True)
        
        if candidate_id:
            logger.info(f"Matching candidate ID: {candidate_id}")
            matches = matching_service.find_jobs_for_candidate(
                candidate_id=candidate_id,
                top_k=top_k
            )
        elif candidate_file:
            logger.info(f"Matching candidate from file: {candidate_file} (index: {candidate_index})")
            matches = matching_service.find_jobs_for_candidate_from_file(
                candidate_file=candidate_file,
                candidate_index=candidate_index,
                top_k=top_k
            )
        elif candidate_text:
            logger.info("Matching candidate from text")
            matches = matching_service.find_jobs_for_candidate_text(
                candidate_text=candidate_text,
                top_k=top_k
            )
        else:
            logger.error("Must provide candidate_id, candidate_file, or candidate_text")
            return False
        
        # Display results
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"TOP {len(matches)} JOB MATCHES")
        logger.info("=" * 80)
        
        from tqdm import tqdm
        for i, match in enumerate(tqdm(matches, desc="Displaying results", unit="match"), 1):
            logger.info("")
            logger.info(f"{i}. {match['title']} (Score: {match['similarity_score']:.4f})")
            logger.info(f"   Job ID: {match['job_id']}")
            if match.get('company'):
                logger.info(f"   Company: {match['company']}")
            if match.get('location'):
                logger.info(f"   Location: {match['location']}")
            if match.get('description'):
                logger.info(f"   Description: {match['description'][:200]}...")
        
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Found {len(matches)} matching jobs")
        logger.info("=" * 80)
        
        return True
    
    except Exception as e:
        logger.error(f"Error matching candidate: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(
        description="Match candidate to top 50 jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Match by candidate ID (from database)
  python scripts/match_candidate_to_jobs.py --candidate-id candidate_001
  
  # Match candidate from processed file (first candidate)
  python scripts/match_candidate_to_jobs.py --candidate-file data/processed/candidates_dataset.csv
  
  # Match specific candidate from file (by index)
  python scripts/match_candidate_to_jobs.py --candidate-file data/processed/candidates_dataset.csv --candidate-index 5
  
  # Match from text
  python scripts/match_candidate_to_jobs.py --candidate-text "Software engineer with Python and ML experience"
  
  # Get top 100 matches
  python scripts/match_candidate_to_jobs.py --candidate-id candidate_001 --top-k 100
        """
    )
    
    parser.add_argument("--candidate-id", type=str, help="Candidate ID from database")
    parser.add_argument("--candidate-file", type=str, help="Path to processed candidate file")
    parser.add_argument("--candidate-index", type=int, default=0, help="Index in candidate file (0-based)")
    parser.add_argument("--candidate-text", type=str, help="Candidate information as text")
    parser.add_argument("--top-k", type=int, default=50, help="Number of top matches (default: 50)")
    
    args = parser.parse_args()
    
    if not args.candidate_id and not args.candidate_file and not args.candidate_text:
        parser.error("Must provide one of: --candidate-id, --candidate-file, or --candidate-text")
    
    success = match_candidate(
        candidate_id=args.candidate_id,
        candidate_file=args.candidate_file,
        candidate_index=args.candidate_index,
        candidate_text=args.candidate_text,
        top_k=args.top_k
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

