"""Complete workflow script: Embedding -> Recommendations -> Save to Database."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from src.database.connection import SessionLocal, engine, Base
from src.database.models import ProcessedCandidateRecommendation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_full_workflow(
    candidate_file: str,
    jd_file: str,
    skip_embeddings: bool = False,
    skip_recommendations: bool = False,
    top_k: int = 10,
    limit_candidates: int = None
):
    """
    Run complete workflow:
    1. Initialize database (create tables if needed)
    2. Generate embeddings (if not skipped)
    3. Generate recommendations (if not skipped)
    
    Args:
        candidate_file: Path to candidate CSV file
        jd_file: Path to JD CSV file
        skip_embeddings: Skip embedding generation step
        skip_recommendations: Skip recommendation generation step
        top_k: Number of top jobs per candidate
        limit_candidates: Limit number of candidates to process
    """
    logger.info("=" * 80)
    logger.info("FULL WORKFLOW: Embedding -> Recommendations -> Database")
    logger.info("=" * 80)
    logger.info("")
    
    # Step 1: Initialize database
    logger.info("STEP 1: Initializing Database")
    logger.info("-" * 80)
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✓ Database tables created/verified")
    except Exception as e:
        logger.error(f"✗ Error initializing database: {e}")
        return False
    logger.info("")
    
    # Step 2: Generate embeddings
    if not skip_embeddings:
        logger.info("STEP 2: Generating Field-by-Field Embeddings")
        logger.info("-" * 80)
        try:
            import subprocess
            import sys
            
            # Run embedding script
            logger.info("Generating JD embeddings...")
            result = subprocess.run(
                [sys.executable, "scripts/generate_field_mapping_embeddings.py",
                 "--jd-file", jd_file,
                 "--file-type", "csv",
                 "--batch-size", "100"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                logger.error(f"Error generating JD embeddings: {result.stderr}")
                return False
            logger.info("✓ JD embeddings generated")
            logger.info("")
            
            logger.info("Generating candidate embeddings...")
            result = subprocess.run(
                [sys.executable, "scripts/generate_field_mapping_embeddings.py",
                 "--candidate-file", candidate_file,
                 "--file-type", "csv",
                 "--batch-size", "100"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                logger.error(f"Error generating candidate embeddings: {result.stderr}")
                return False
            logger.info("✓ Candidate embeddings generated")
            logger.info("")
        except Exception as e:
            logger.error(f"✗ Error generating embeddings: {e}", exc_info=True)
            return False
    else:
        logger.info("STEP 2: Skipping Embedding Generation")
        logger.info("-" * 80)
        logger.info("(Using existing embeddings in database)")
        logger.info("")
    
    # Step 3: Generate recommendations
    if not skip_recommendations:
        logger.info("STEP 3: Generating Processed Recommendations")
        logger.info("-" * 80)
        try:
            import subprocess
            import sys
            
            cmd = [
                sys.executable, "scripts/generate_processed_recommendations.py",
                "--candidate-file", candidate_file,
                "--jd-file", jd_file,
                "--top-k", str(top_k)
            ]
            if limit_candidates:
                cmd.extend(["--limit-candidates", str(limit_candidates)])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Error generating recommendations: {result.stderr}")
                return False
            logger.info("✓ Recommendations generated and saved")
        except Exception as e:
            logger.error(f"✗ Error generating recommendations: {e}", exc_info=True)
            return False
    else:
        logger.info("STEP 3: Skipping Recommendation Generation")
        logger.info("-" * 80)
        logger.info("(Using existing recommendations in database)")
        logger.info("")
    
    logger.info("=" * 80)
    logger.info("FULL WORKFLOW - Completed Successfully")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Query processed recommendations from 'processed_candidate_recommendations' table")
    logger.info("  2. Use MatchingService with use_processed=True to get cached results")
    logger.info("")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run complete workflow: Embedding -> Recommendations -> Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full workflow
  python scripts/run_full_workflow.py \\
      --candidate-file data/processed/candidate_processed.csv \\
      --jd-file data/processed/jd_processed.csv
  
  # Skip embedding generation (use existing)
  python scripts/run_full_workflow.py \\
      --candidate-file data/processed/candidate_processed.csv \\
      --jd-file data/processed/jd_processed.csv \\
      --skip-embeddings
  
  # Skip recommendation generation (use existing)
  python scripts/run_full_workflow.py \\
      --candidate-file data/processed/candidate_processed.csv \\
      --jd-file data/processed/jd_processed.csv \\
      --skip-recommendations
  
  # Test with limited candidates
  python scripts/run_full_workflow.py \\
      --candidate-file data/processed/candidate_processed.csv \\
      --jd-file data/processed/jd_processed.csv \\
      --limit-candidates 100
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
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation step"
    )
    
    parser.add_argument(
        "--skip-recommendations",
        action="store_true",
        help="Skip recommendation generation step"
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
    
    args = parser.parse_args()
    
    try:
        success = run_full_workflow(
            candidate_file=args.candidate_file,
            jd_file=args.jd_file,
            skip_embeddings=args.skip_embeddings,
            skip_recommendations=args.skip_recommendations,
            top_k=args.top_k,
            limit_candidates=args.limit_candidates
        )
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

