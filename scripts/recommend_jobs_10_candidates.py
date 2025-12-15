"""Recommend top 10 jobs for 10 candidates using Two-Tower Matching Service."""
import sys
import logging
from pathlib import Path
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.connection import SessionLocal
from src.database.new_models import CV
from src.services.two_tower_matching_service import TwoTowerMatchingService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def format_candidate_info(candidate: CV) -> str:
    """Format candidate information for display."""
    info_parts = []
    if candidate.id:
        info_parts.append(f"ID: {candidate.id}")
    if candidate.fullName:
        info_parts.append(f"Name: {candidate.fullName}")
    if candidate.title:
        info_parts.append(f"Title: {candidate.title}")
    if candidate.email:
        info_parts.append(f"Email: {candidate.email}")
    return " | ".join(info_parts) if info_parts else f"ID: {candidate.id}"


def print_candidate_recommendations(candidate: CV, recommendations: List[Dict]):
    """Print job recommendations for a candidate in a formatted table."""
    print("\n" + "=" * 100)
    print(f"CANDIDATE: {format_candidate_info(candidate)}")
    print("=" * 100)
    
    if not recommendations:
        print("No job recommendations found.")
        return
    
    # Print header
    print(f"\nTop {len(recommendations)} Job Recommendations:")
    print("-" * 100)
    print(f"{'Rank':<6} {'Job ID':<20} {'Title':<40} {'Company':<20} {'Score':<10}")
    print("-" * 100)
    
    # Print each recommendation
    for idx, job in enumerate(recommendations, 1):
        job_id = job.get('job_id', 'N/A')[:18]  # Truncate if too long
        title = (job.get('title') or 'N/A')[:38]  # Truncate if too long
        company = (job.get('company') or 'N/A')[:18]  # Truncate if too long
        score = job.get('score', 0.0)
        
        print(f"{idx:<6} {job_id:<20} {title:<40} {company:<20} {score:<10.4f}")
    
    # Print detailed field scores for top 3
    if len(recommendations) > 0:
        print("\nDetailed Scores for Top 3 Jobs:")
        print("-" * 100)
        for idx, job in enumerate(recommendations[:3], 1):
            print(f"\nRank {idx}: {job.get('title', 'N/A')}")
            print(f"  Job ID: {job.get('job_id', 'N/A')}")
            field_scores = job.get('field_scores', {})
            print(f"  Total Score: {job.get('score', 0.0):.4f}")
            print(f"    - Title Similarity: {field_scores.get('title', 0.0):.4f}")
            print(f"    - Skills Similarity: {field_scores.get('skills', 0.0):.4f}")
            print(f"    - Experience Similarity: {field_scores.get('experience', 0.0):.4f}")
            if job.get('location'):
                print(f"  Location: {job.get('location')}")


def main():
    """Main function to recommend jobs for 10 candidates."""
    db = SessionLocal()
    
    try:
        # Get 10 candidates from database that have all embeddings
        logger.info("Fetching 10 candidates with embeddings from database...")
        candidates = db.query(CV).filter(
            CV.title_embedding.isnot(None),
            CV.skills_embedding.isnot(None),
            CV.experience_embedding.isnot(None)
        ).limit(10).all()
        
        if not candidates:
            logger.warning("No candidates with embeddings found in database.")
            print("No candidates with embeddings found in database.")
            print("Note: Only candidates with title_embedding, skills_embedding, and experience_embedding are processed.")
            return
        
        logger.info(f"Found {len(candidates)} candidates with embeddings. Processing recommendations...")
        print(f"\n{'='*100}")
        print(f"JOB RECOMMENDATIONS FOR {len(candidates)} CANDIDATES (WITH EMBEDDINGS)")
        print(f"Note: Only candidates and jobs with complete embeddings are processed.")
        print(f"{'='*100}")
        
        # Initialize matching service once (it will be reused for all candidates)
        logger.info("Initializing TwoTowerMatchingService...")
        matching_service = TwoTowerMatchingService(db)
        logger.info("TwoTowerMatchingService initialized successfully.")
        
        # Process each candidate
        successful_count = 0
        failed_count = 0
        
        for idx, candidate in enumerate(candidates, 1):
            try:
                logger.info(f"Processing candidate {idx}/{len(candidates)}: {candidate.id}")
                
                # Get top 10 job recommendations
                recommendations = matching_service.find_jobs_for_candidate(
                    candidate_id=candidate.id,
                    top_k=10
                )
                
                # Print recommendations
                print_candidate_recommendations(candidate, recommendations)
                
                successful_count += 1
                logger.info(f"Successfully processed candidate {candidate.id}: {len(recommendations)} recommendations")
                
            except Exception as e:
                failed_count += 1
                logger.error(f"Error processing candidate {candidate.id}: {e}", exc_info=True)
                print(f"\n{'='*100}")
                print(f"ERROR processing candidate: {format_candidate_info(candidate)}")
                print(f"Error: {str(e)}")
                print(f"{'='*100}")
                continue
        
        # Print summary
        print("\n" + "=" * 100)
        print("SUMMARY")
        print("=" * 100)
        print(f"Total candidates processed: {len(candidates)}")
        print(f"Successful: {successful_count}")
        print(f"Failed: {failed_count}")
        print("=" * 100)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nFatal error: {str(e)}")
        raise
    
    finally:
        db.close()
        logger.info("Database connection closed.")


if __name__ == "__main__":
    main()

