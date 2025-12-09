"""Script to test multi-filter matching service."""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.services.multi_filter_matching_service import MultiFilterMatchingService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_matching_by_candidate_id(candidate_id: str, top_k: int = 10):
    """Test matching for a candidate by ID."""
    logger.info("=" * 80)
    logger.info(f"TESTING MULTI-FILTER MATCHING for Candidate ID: {candidate_id}")
    logger.info("=" * 80)
    
    db: Session = next(get_db())
    try:
        # Get candidate information for comparison
        from src.database.multi_field_repository import MultiFieldEmbeddingRepository
        repo = MultiFieldEmbeddingRepository(db)
        candidate = repo.get_candidate_multi_embedding(candidate_id)
        
        if candidate:
            logger.info("\n" + "=" * 80)
            logger.info("📄 THÔNG TIN ỨNG VIÊN (CV)")
            logger.info("=" * 80)
            logger.info(f"Candidate ID: {candidate.candidate_id}")
            logger.info(f"Tên: {candidate.name or 'N/A'}")
            logger.info(f"Email: {candidate.email or 'N/A'}")
            logger.info(f"Vị trí mong muốn/Title: {candidate.title or 'N/A'}")
            logger.info(f"Kỹ năng (Skills): {candidate.skills[:500] + '...' if candidate.skills and len(candidate.skills) > 500 else (candidate.skills or 'N/A')}")
            logger.info(f"Kinh nghiệm (Experience): {candidate.experience[:500] + '...' if candidate.experience and len(candidate.experience) > 500 else (candidate.experience or 'N/A')}")
            logger.info("=" * 80)
        
        service = MultiFilterMatchingService(db, use_faiss=False)  # Use DB search for testing
        results = service.find_jobs_for_candidate(candidate_id, top_k=top_k)
        
        # Helper function to format similarity values (handle None)
        def format_similarity(value):
            if value is None:
                return "N/A (skipped)"
            return f"{value:.4f}"
        
        # Helper function to truncate text
        def truncate_text(text, max_length=500):
            if not text:
                return "N/A"
            if len(text) > max_length:
                return text[:max_length] + "..."
            return text
        
        logger.info(f"\nFound {len(results)} matching jobs:\n")
        for i, result in enumerate(results, 1):
            # Get full job details from database
            job = repo.get_job_multi_embedding(result['job_id'])
            
            logger.info("=" * 80)
            logger.info(f"🏢 MATCH #{i}: Job ID: {result['job_id']}")
            logger.info("=" * 80)
            logger.info(f"📊 Similarity Score: {result['similarity_score']:.4f}")
            logger.info(f"   - Experience/Requirement: {format_similarity(result['field_similarities']['experience'])}")
            logger.info(f"   - Skills: {format_similarity(result['field_similarities']['skills'])}")
            logger.info(f"   - Title: {format_similarity(result['field_similarities']['title'])}")
            logger.info("")
            logger.info(f"📋 THÔNG TIN CÔNG VIỆC:")
            logger.info(f"   Tiêu đề (Title): {job.title if job else result.get('title', 'N/A')}")
            logger.info(f"   Công ty (Company): {job.company if job else result.get('company', 'N/A')}")
            logger.info(f"   Địa điểm (Location): {job.location if job else result.get('location', 'N/A')}")
            logger.info(f"   Kỹ năng yêu cầu (Required Skills): {truncate_text(job.skills, 500) if job else 'N/A'}")
            logger.info(f"   Yêu cầu công việc (Requirements): {truncate_text(job.requirement, 500) if job else 'N/A'}")
            logger.info("")
            
            if candidate:
                logger.info(f"🔄 ĐỐI CHIẾU:")
                logger.info(f"   CV Title: {candidate.title or 'N/A'}")
                logger.info(f"   JD Title: {job.title if job else result.get('title', 'N/A')}")
                logger.info(f"   → Title Similarity: {format_similarity(result['field_similarities']['title'])}")
                logger.info("")
                logger.info(f"   CV Skills: {truncate_text(candidate.skills, 200) if candidate.skills else 'N/A'}")
                logger.info(f"   JD Skills: {truncate_text(job.skills, 200) if job else 'N/A'}")
                logger.info(f"   → Skills Similarity: {format_similarity(result['field_similarities']['skills'])}")
                logger.info("")
                logger.info(f"   CV Experience: {truncate_text(candidate.experience, 200) if candidate.experience else 'N/A'}")
                logger.info(f"   JD Requirements: {truncate_text(job.requirement, 200) if job else 'N/A'}")
                logger.info(f"   → Experience/Requirement Similarity: {format_similarity(result['field_similarities']['experience'])}")
            logger.info("")
        
        return results
    except Exception as e:
        logger.error(f"✗ Error testing matching: {e}", exc_info=True)
        raise
    finally:
        db.close()


def test_matching_by_text(title: str = None, skills: str = None, experience: str = None, top_k: int = 10):
    """Test matching for a candidate by text."""
    logger.info("=" * 80)
    logger.info("TESTING MULTI-FILTER MATCHING for New Candidate (from text)")
    logger.info("=" * 80)
    
    db: Session = next(get_db())
    try:
        service = MultiFilterMatchingService(db, use_faiss=False)  # Use DB search for testing
        results = service.find_jobs_for_candidate_text(
            title=title,
            skills=skills,
            experience=experience,
            top_k=top_k
        )
        
        logger.info(f"\nFound {len(results)} matching jobs:\n")
        
        # Helper function to format similarity values (handle None)
        def format_similarity(value):
            if value is None:
                return "N/A (skipped)"
            return f"{value:.4f}"
        
        for i, result in enumerate(results, 1):
            logger.info(f"{i}. Job ID: {result['job_id']}")
            logger.info(f"   Title: {result['title']}")
            logger.info(f"   Company: {result.get('company', 'N/A')}")
            logger.info(f"   Location: {result.get('location', 'N/A')}")
            logger.info(f"   Similarity Score: {result['similarity_score']:.4f}")
            logger.info(f"   Field Similarities:")
            logger.info(f"     - Experience: {format_similarity(result['field_similarities']['experience'])}")
            logger.info(f"     - Skills: {format_similarity(result['field_similarities']['skills'])}")
            logger.info(f"     - Title: {format_similarity(result['field_similarities']['title'])}")
            logger.info("")
        
        return results
    except Exception as e:
        logger.error(f"✗ Error testing matching: {e}", exc_info=True)
        raise
    finally:
        db.close()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test multi-filter matching service')
    parser.add_argument('--candidate-id', type=str, help='Candidate ID to test matching')
    parser.add_argument('--title', type=str, help='Candidate desired job title')
    parser.add_argument('--skills', type=str, help='Candidate skills')
    parser.add_argument('--experience', type=str, help='Candidate experience')
    parser.add_argument('--top-k', type=int, default=10, help='Number of top matches to return (default: 10)')
    
    args = parser.parse_args()
    
    if args.candidate_id:
        test_matching_by_candidate_id(args.candidate_id, top_k=args.top_k)
    elif args.title or args.skills or args.experience:
        test_matching_by_text(
            title=args.title,
            skills=args.skills,
            experience=args.experience,
            top_k=args.top_k
        )
    else:
        parser.print_help()
        logger.error("Please specify --candidate-id or provide candidate text fields (--title, --skills, --experience)")


if __name__ == "__main__":
    main()

