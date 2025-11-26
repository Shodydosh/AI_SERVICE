"""Script to generate processed candidate recommendations using multi-filtering and save to database."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.services.field_mapping_matching_service import FieldMappingMatchingService
from src.database.repository import EmbeddingRepository
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_and_save_recommendations(
    candidate_file: str,
    jd_file: str,
    top_k: int = 10,
    limit_candidates: int = None,
    replace_existing: bool = True
):
    """
    Generate recommendations for all candidates using multi-filtering
    and save to processed_candidate_recommendations table.
    
    Args:
        candidate_file: Path to candidate CSV file
        jd_file: Path to JD CSV file
        top_k: Number of top jobs per candidate (default: 10)
        limit_candidates: Optional limit on number of candidates to process
        replace_existing: If True, replace existing recommendations
    """
    logger.info("=" * 80)
    logger.info("GENERATING PROCESSED CANDIDATE RECOMMENDATIONS")
    logger.info("=" * 80)
    logger.info(f"Candidate file: {candidate_file}")
    logger.info(f"JD file: {jd_file}")
    logger.info(f"Top K: {top_k}")
    logger.info(f"Limit candidates: {limit_candidates or 'All'}")
    logger.info("")
    
    db: Session = SessionLocal()
    try:
        # Initialize services
        matching_service = FieldMappingMatchingService(db)
        repository = EmbeddingRepository(db)
        
        # Generate recommendations for all candidates
        logger.info("Generating recommendations using multi-filtering approach...")
        all_recommendations = matching_service.find_top_jobs_for_all_candidates(
            candidate_file=candidate_file,
            jd_file=jd_file,
            top_k=top_k,
            limit_candidates=limit_candidates
        )
        
        logger.info("")
        logger.info(f"Generated recommendations for {len(all_recommendations)} candidates")
        
        # Save to database
        logger.info("Saving recommendations to database...")
        total_saved = repository.save_processed_recommendations_batch(
            all_recommendations=all_recommendations,
            replace_existing=replace_existing
        )
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("PROCESSED RECOMMENDATIONS - Completed Successfully")
        logger.info("=" * 80)
        logger.info(f"Total recommendations saved: {total_saved}")
        logger.info(f"Average recommendations per candidate: {total_saved / len(all_recommendations):.1f}")
        logger.info("")
        logger.info("Recommendations are now stored in 'processed_candidate_recommendations' table")
        logger.info("You can query them without re-running embeddings!")
        
        return total_saved
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}", exc_info=True)
        raise
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate processed candidate recommendations using multi-filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate recommendations for all candidates
  python scripts/generate_processed_recommendations.py \\
      --candidate-file data/processed/candidate_processed.csv \\
      --jd-file data/processed/jd_processed.csv
  
  # Generate for first 100 candidates only
  python scripts/generate_processed_recommendations.py \\
      --candidate-file data/processed/candidate_processed.csv \\
      --jd-file data/processed/jd_processed.csv \\
      --limit-candidates 100
  
  # Generate top 20 jobs per candidate
  python scripts/generate_processed_recommendations.py \\
      --candidate-file data/processed/candidate_processed.csv \\
      --jd-file data/processed/jd_processed.csv \\
      --top-k 20
        """
    )
    
    parser.add_argument(
        "--candidate-file",
        type=str,
        required=True,
        help="Path to candidate CSV file"
    )
    
    parser.add_argument(
        "--jd-file",
        type=str,
        required=True,
        help="Path to JD CSV file"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top jobs per candidate (default: 10)"
    )
    
    parser.add_argument(
        "--limit-candidates",
        type=int,
        default=None,
        help="Limit number of candidates to process (for testing)"
    )
    
    parser.add_argument(
        "--no-replace",
        action="store_true",
        help="Don't replace existing recommendations (append only)"
    )
    
    args = parser.parse_args()
    
    # Run workflow
    try:
        generate_and_save_recommendations(
            candidate_file=args.candidate_file,
            jd_file=args.jd_file,
            top_k=args.top_k,
            limit_candidates=args.limit_candidates,
            replace_existing=not args.no_replace
        )
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

