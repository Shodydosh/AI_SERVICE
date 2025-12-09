"""Complete workflow script: Initialize DB -> Process Data -> Test Matching."""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import traceback
from sqlalchemy.orm import Session
from src.database.connection import get_db, engine, Base, get_database_info
from src.database.models import (
    JobDescriptionMultiEmbedding,
    CandidateMultiEmbedding
)
from src.services.multi_field_embedding_service import MultiFieldEmbeddingService
from src.services.multi_filter_matching_service import MultiFilterMatchingService

# Configure logging to show all output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def step_1_init_database():
    """Step 1: Initialize database tables."""
    print_header("STEP 1: INITIALIZING DATABASE TABLES")
    
    try:
        db_info = get_database_info()
        logger.info(f"Connecting to database: {db_info['host']}:{db_info['port']}/{db_info['database']} (user: {db_info['username']})")
        
        logger.info("Creating multi-field embedding tables...")
        Base.metadata.create_all(bind=engine, tables=[
            JobDescriptionMultiEmbedding.__table__,
            CandidateMultiEmbedding.__table__
        ])
        
        logger.info("✓ Tables created/verified successfully!")
        return True
    except Exception as e:
        logger.error(f"✗ Error creating tables: {e}")
        traceback.print_exc()
        return False


def step_2_process_datasets(jd_file: str = None, candidate_file: str = None, batch_size: int = 100):
    """Step 2: Process datasets."""
    print_header("STEP 2: PROCESSING DATASETS")
    
    project_root = Path(__file__).parent.parent
    
    # Set default file paths
    if not jd_file:
        jd_file = str(project_root / "data" / "raw" / "job_data.csv")
    if not candidate_file:
        candidate_file = str(project_root / "data" / "raw" / "candidates_dataset.csv")
    
    total_jobs = 0
    total_candidates = 0
    
    # Process job descriptions
    if os.path.exists(jd_file):
        logger.info(f"Processing job descriptions from: {jd_file}")
        db: Session = next(get_db())
        try:
            service = MultiFieldEmbeddingService(db)
            total_jobs = service.process_jd_dataset(
                file_path=jd_file,
                file_type="csv",
                batch_size=batch_size
            )
            logger.info(f"✓ Successfully processed {total_jobs} job descriptions")
        except Exception as e:
            logger.error(f"✗ Error processing job descriptions: {e}")
            traceback.print_exc()
            return False, 0, 0
        finally:
            db.close()
    else:
        logger.warning(f"JD file not found: {jd_file}")
    
    # Process candidates
    if os.path.exists(candidate_file):
        logger.info(f"Processing candidates from: {candidate_file}")
        db: Session = next(get_db())
        try:
            service = MultiFieldEmbeddingService(db)
            total_candidates = service.process_candidate_dataset(
                file_path=candidate_file,
                file_type="csv",
                batch_size=batch_size
            )
            logger.info(f"✓ Successfully processed {total_candidates} candidates")
        except Exception as e:
            logger.error(f"✗ Error processing candidates: {e}")
            traceback.print_exc()
            return False, total_jobs, 0
        finally:
            db.close()
    else:
        logger.warning(f"Candidate file not found: {candidate_file}")
    
    return True, total_jobs, total_candidates


def step_3_test_matching(candidate_id: str = "15001", top_k: int = 10):
    """Step 3: Test matching."""
    print_header(f"STEP 3: TESTING MATCHING (Candidate ID: {candidate_id})")
    
    db: Session = next(get_db())
    try:
        # Diagnostic check before testing
        from src.database.multi_field_repository import MultiFieldEmbeddingRepository
        repo = MultiFieldEmbeddingRepository(db)
        
        job_count = repo.count_job_multi_embeddings()
        candidate_count = repo.count_candidate_multi_embeddings()
        
        logger.info(f"Database status:")
        logger.info(f"  - Jobs in database: {job_count}")
        logger.info(f"  - Candidates in database: {candidate_count}")
        
        if job_count == 0:
            logger.error("✗ No jobs found in database! Please run Step 2 to process datasets first.")
            return False
        
        if candidate_count == 0:
            logger.error("✗ No candidates found in database! Please run Step 2 to process datasets first.")
            return False
        
        # Check if candidate exists
        candidate = repo.get_candidate_multi_embedding(candidate_id)
        if not candidate:
            logger.error(f"✗ Candidate {candidate_id} not found in database!")
            logger.info(f"Available candidate IDs (first 10):")
            all_candidates = repo.get_all_candidate_multi_embeddings()
            for i, c in enumerate(all_candidates[:10], 1):
                logger.info(f"  {i}. {c.candidate_id}")
            return False
        
        logger.info(f"✓ Candidate {candidate_id} found in database")
        
        # Helper function to truncate text and handle None/empty (defined early so we can use it)
        def truncate_text(text, max_length=500):
            if text is None:
                return "N/A (None)"
            if isinstance(text, str):
                text = text.strip()
                if not text:
                    return "N/A (Empty)"
            text_str = str(text)
            if len(text_str) > max_length:
                return text_str[:max_length] + "..."
            return text_str
        
        # Get actual candidate values for debugging
        candidate_skills_value = getattr(candidate, 'skills', None)
        candidate_experience_value = getattr(candidate, 'experience', None)
        candidate_title_value = getattr(candidate, 'title', None)
        
        # Print candidate (CV) information for comparison
        logger.info("\n" + "=" * 80)
        logger.info("📄 THÔNG TIN ỨNG VIÊN (CV)")
        logger.info("=" * 80)
        logger.info(f"Candidate ID: {candidate.candidate_id}")
        logger.info(f"Tên: {candidate.name or 'N/A'}")
        logger.info(f"Email: {candidate.email or 'N/A'}")
        logger.info(f"Vị trí mong muốn/Title: {candidate_title_value or 'N/A'}")
        logger.info(f"Kỹ năng (Skills): {truncate_text(candidate_skills_value, 500)}")
        logger.info(f"Kinh nghiệm (Experience): {truncate_text(candidate_experience_value, 500)}")
        
        # Debug candidate fields
        logger.info("\n🔍 DEBUG - Candidate field values:")
        logger.info(f"   - title: type={type(candidate_title_value).__name__}, value={repr(candidate_title_value) if candidate_title_value else 'None/Empty'}")
        logger.info(f"   - skills: type={type(candidate_skills_value).__name__}, value={repr(candidate_skills_value) if candidate_skills_value else 'None/Empty'}")
        logger.info(f"   - experience: type={type(candidate_experience_value).__name__}, value={repr(candidate_experience_value) if candidate_experience_value else 'None/Empty'}")
        
        logger.info("=" * 80)
        
        service = MultiFilterMatchingService(db, use_faiss=False)
        results = service.find_jobs_for_candidate(candidate_id, top_k=top_k)
        
        if not results:
            logger.warning("No matching jobs found!")
            logger.info("This could mean:")
            logger.info("1. Matching criteria too strict")
            logger.info("2. Jobs don't match candidate profile")
            return False
        
        logger.info(f"\n✓ Found {len(results)} matching jobs:\n")
        
        # Helper function to format similarity values (handle None)
        def format_similarity(value):
            if value is None:
                return "N/A (skipped)"
            return f"{value:.4f}"
        
        for i, result in enumerate(results[:5], 1):  # Show top 5
            # Get full job details from database
            job = repo.get_job_multi_embedding(result['job_id'])
            
            if not job:
                logger.warning(f"⚠️ Job {result['job_id']} not found in database!")
                continue
            
            # Get actual values
            job_skills_value = getattr(job, 'skills', None)
            job_requirement_value = getattr(job, 'requirement', None)
            
            # Log actual values for debugging
            logger.info(f"🔍 DEBUG - Job {result['job_id']} field values:")
            logger.info(f"   - skills: type={type(job_skills_value).__name__}, value={repr(job_skills_value) if job_skills_value else 'None/Empty'}")
            logger.info(f"   - requirement: type={type(job_requirement_value).__name__}, value={repr(job_requirement_value) if job_requirement_value else 'None/Empty'}")
            
            logger.info("=" * 80)
            logger.info(f"🏢 MATCH #{i}: Job ID: {result['job_id']}")
            logger.info("=" * 80)
            logger.info(f"📊 Similarity Score: {result['similarity_score']:.4f}")
            logger.info(f"   - Experience/Requirement: {format_similarity(result['field_similarities']['experience'])}")
            logger.info(f"   - Skills: {format_similarity(result['field_similarities']['skills'])}")
            logger.info(f"   - Title: {format_similarity(result['field_similarities']['title'])}")
            logger.info("")
            logger.info(f"📋 THÔNG TIN CÔNG VIỆC:")
            logger.info(f"   Tiêu đề (Title): {job.title or 'N/A'}")
            logger.info(f"   Công ty (Company): {job.company or 'N/A'}")
            logger.info(f"   Địa điểm (Location): {job.location or 'N/A'}")
            logger.info(f"   Kỹ năng yêu cầu (Required Skills): {truncate_text(job_skills_value, 500)}")
            logger.info(f"   Yêu cầu công việc (Requirements): {truncate_text(job_requirement_value, 500)}")
            logger.info("")
            logger.info(f"🔄 ĐỐI CHIẾU:")
            logger.info(f"   CV Title: {truncate_text(candidate_title_value, 200)}")
            logger.info(f"   JD Title: {truncate_text(job.title, 200)}")
            logger.info(f"   → Title Similarity: {format_similarity(result['field_similarities']['title'])}")
            logger.info("")
            logger.info(f"   CV Skills: {truncate_text(candidate_skills_value, 200)}")
            logger.info(f"   JD Skills: {truncate_text(job_skills_value, 200)}")
            logger.info(f"   → Skills Similarity: {format_similarity(result['field_similarities']['skills'])}")
            logger.info("")
            logger.info(f"   CV Experience: {truncate_text(candidate_experience_value, 200)}")
            logger.info(f"   JD Requirements: {truncate_text(job_requirement_value, 200)}")
            logger.info(f"   → Experience/Requirement Similarity: {format_similarity(result['field_similarities']['experience'])}")
            logger.info("")
        
        return True
    except Exception as e:
        logger.error(f"✗ Error testing matching: {e}")
        traceback.print_exc()
        return False
    finally:
        db.close()


def main():
    """Main workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run complete workflow for 3-field multi-filter matching')
    parser.add_argument('--jd-file', type=str, help='Path to job description CSV file')
    parser.add_argument('--candidate-file', type=str, help='Path to candidate CSV file')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing (default: 50)')
    parser.add_argument('--test-candidate-id', type=str, default="15001", help='Candidate ID to test matching (default: 15001)')
    parser.add_argument('--skip-processing', action='store_true', help='Skip dataset processing step')
    parser.add_argument('--skip-testing', action='store_true', help='Skip matching test step')
    
    args = parser.parse_args()
    
    print_header("COMPLETE WORKFLOW: 3-FIELD MULTI-FILTER MATCHING")
    
    # Step 1: Initialize database
    if not step_1_init_database():
        logger.error("Failed at Step 1. Cannot proceed.")
        sys.exit(1)
    
    # Step 2: Process datasets
    if not args.skip_processing:
        success, total_jobs, total_candidates = step_2_process_datasets(
            jd_file=args.jd_file,
            candidate_file=args.candidate_file,
            batch_size=args.batch_size
        )
        
        if not success:
            logger.error("Failed at Step 2. Cannot proceed.")
            sys.exit(1)
        
        if total_jobs == 0 and total_candidates == 0:
            logger.warning("No data processed. Check file paths.")
        else:
            logger.info(f"✓ Processed {total_jobs} jobs and {total_candidates} candidates")
            
            # Verify data in database
            db = next(get_db())
            try:
                from src.database.multi_field_repository import MultiFieldEmbeddingRepository
                repo = MultiFieldEmbeddingRepository(db)
                db_job_count = repo.count_job_multi_embeddings()
                db_candidate_count = repo.count_candidate_multi_embeddings()
                logger.info(f"✓ Verified in database: {db_job_count} jobs, {db_candidate_count} candidates")
            finally:
                db.close()
    
    # Step 3: Test matching
    if not args.skip_testing:
        if not step_3_test_matching(candidate_id=args.test_candidate_id):
            logger.error("Failed at Step 3.")
            sys.exit(1)
    
    print_header("WORKFLOW COMPLETED SUCCESSFULLY!")
    logger.info("All steps completed. System is ready to use!")


if __name__ == "__main__":
    main()

