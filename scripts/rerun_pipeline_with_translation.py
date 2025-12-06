"""Script to rerun the complete pipeline from scratch with Vietnamese translation."""
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.orm import Session
from src.database.connection import get_db, engine
from src.database.models import (
    JobDescriptionMultiEmbedding,
    CandidateMultiEmbedding
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clear_embeddings():
    """Clear all existing embeddings from database."""
    logger.info("=" * 80)
    logger.info("CLEARING EXISTING EMBEDDINGS")
    logger.info("=" * 80)
    
    db: Session = next(get_db())
    try:
        # Delete all job embeddings
        job_count = db.query(JobDescriptionMultiEmbedding).count()
        if job_count > 0:
            logger.info(f"Deleting {job_count} job embeddings...")
            db.query(JobDescriptionMultiEmbedding).delete()
        
        # Delete all candidate embeddings
        candidate_count = db.query(CandidateMultiEmbedding).count()
        if candidate_count > 0:
            logger.info(f"Deleting {candidate_count} candidate embeddings...")
            db.query(CandidateMultiEmbedding).delete()
        
        db.commit()
        logger.info("✓ All embeddings cleared successfully!")
        return True
    except Exception as e:
        logger.error(f"✗ Error clearing embeddings: {e}", exc_info=True)
        db.rollback()
        return False
    finally:
        db.close()


def process_embeddings(jd_file: str = None, candidate_file: str = None, batch_size: int = 100):
    """Process embeddings with Vietnamese translation."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("PROCESSING EMBEDDINGS WITH VIETNAMESE TRANSLATION")
    logger.info("=" * 80)
    
    # Auto-detect files if not provided
    if not jd_file:
        jd_file = "data/filtered/jds_with_skills.csv"
        if not Path(jd_file).exists():
            jd_file = "data/raw/job_data.csv"
    
    if not candidate_file:
        candidate_file = "data/filtered/candidates_with_skills.csv"
        if not Path(candidate_file).exists():
            candidate_file = "data/raw/candidates_dataset.csv"
    
    try:
        # Try to import MultiFieldEmbeddingService
        try:
            from src.services.multi_field_embedding_service import MultiFieldEmbeddingService
        except ImportError:
            logger.error("MultiFieldEmbeddingService not found. Trying alternative approach...")
            # Use direct processing with encoders
            from src.embeddings.job_tower_encoder import JobTowerEncoder
            from src.embeddings.candidate_tower_encoder import CandidateTowerEncoder
            from src.database.multi_field_repository import MultiFieldEmbeddingRepository
            from src.utils.three_field_extractor import ThreeFieldExtractor
            import pandas as pd
            
            db: Session = next(get_db())
            try:
                repo = MultiFieldEmbeddingRepository(db)
                extractor = ThreeFieldExtractor()
                job_encoder = JobTowerEncoder()
                candidate_encoder = CandidateTowerEncoder()
                
                # Process jobs
                if Path(jd_file).exists():
                    logger.info(f"Processing jobs from: {jd_file}")
                    df_jobs = pd.read_csv(jd_file)
                    total_jobs = 0
                    
                    for idx, row in df_jobs.iterrows():
                        try:
                            fields = extractor.extract_job_fields(row)
                            if not fields.get('title'):
                                continue
                            
                            # Encode (translation happens in preprocess functions)
                            embeddings = job_encoder.encode_job(
                                title=fields.get('title', ''),
                                skills=fields.get('skills', ''),
                                requirements=fields.get('requirement', '')
                            )
                            
                            # Save to database
                            job_id = str(row.get('job_id', f"job_{idx}"))
                            repo.upsert_job_multi_embedding(
                                job_id=job_id,
                                title=fields.get('title', ''),
                                skills=fields.get('skills', ''),
                                requirement=fields.get('requirement', ''),
                                title_embedding=embeddings['title_embedding'],
                                skills_embedding=embeddings['skills_embedding'],
                                requirement_embedding=embeddings['requirement_embedding']
                            )
                            
                            total_jobs += 1
                            if total_jobs % batch_size == 0:
                                db.commit()
                                logger.info(f"Processed {total_jobs} jobs...")
                        except Exception as e:
                            logger.warning(f"Error processing job {idx}: {e}")
                            continue
                    
                    db.commit()
                    logger.info(f"✓ Processed {total_jobs} jobs")
                
                # Process candidates
                if Path(candidate_file).exists():
                    logger.info(f"Processing candidates from: {candidate_file}")
                    df_candidates = pd.read_csv(candidate_file)
                    total_candidates = 0
                    
                    for idx, row in df_candidates.iterrows():
                        try:
                            fields = extractor.extract_candidate_fields(row)
                            if not fields.get('title'):
                                continue
                            
                            # Encode (translation happens in preprocess functions)
                            embeddings = candidate_encoder.encode_candidate(
                                title=fields.get('title', ''),
                                skills=fields.get('skills', ''),
                                experience=fields.get('experience', '')
                            )
                            
                            # Save to database
                            candidate_id = str(row.get('candidate_id', f"candidate_{idx}"))
                            repo.upsert_candidate_multi_embedding(
                                candidate_id=candidate_id,
                                title=fields.get('title', ''),
                                skills=fields.get('skills', ''),
                                experience=fields.get('experience', ''),
                                title_embedding=embeddings['title_embedding'],
                                skills_embedding=embeddings['skills_embedding'],
                                experience_embedding=embeddings['experience_embedding']
                            )
                            
                            total_candidates += 1
                            if total_candidates % batch_size == 0:
                                db.commit()
                                logger.info(f"Processed {total_candidates} candidates...")
                        except Exception as e:
                            logger.warning(f"Error processing candidate {idx}: {e}")
                            continue
                    
                    db.commit()
                    logger.info(f"✓ Processed {total_candidates} candidates")
                
                return True
            finally:
                db.close()
        
        # If service exists, use it
        db: Session = next(get_db())
        try:
            service = MultiFieldEmbeddingService(db)
            
            # Process jobs
            if Path(jd_file).exists():
                logger.info(f"Processing jobs from: {jd_file}")
                total_jobs = service.process_jd_dataset(
                    file_path=jd_file,
                    file_type="csv",
                    batch_size=batch_size
                )
                logger.info(f"✓ Processed {total_jobs} jobs")
            else:
                logger.warning(f"JD file not found: {jd_file}")
            
            # Process candidates
            if Path(candidate_file).exists():
                logger.info(f"Processing candidates from: {candidate_file}")
                total_candidates = service.process_candidate_dataset(
                    file_path=candidate_file,
                    file_type="csv",
                    batch_size=batch_size
                )
                logger.info(f"✓ Processed {total_candidates} candidates")
            else:
                logger.warning(f"Candidate file not found: {candidate_file}")
            
            return True
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"✗ Error processing embeddings: {e}", exc_info=True)
        return False


def generate_ground_truth():
    """Generate ground truth pairs with translation."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("GENERATING GROUND TRUTH PAIRS")
    logger.info("=" * 80)
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/generate_ground_truth_500_pairs.py"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("✓ Ground truth pairs generated successfully!")
            logger.info(result.stdout)
            return True
        else:
            logger.error(f"✗ Error generating ground truth: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"✗ Error generating ground truth: {e}", exc_info=True)
        return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Rerun complete pipeline with Vietnamese translation')
    parser.add_argument('--jd-file', type=str, help='Path to job description CSV file')
    parser.add_argument('--candidate-file', type=str, help='Path to candidate CSV file')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')
    parser.add_argument('--skip-clear', action='store_true', help='Skip clearing existing embeddings')
    parser.add_argument('--skip-ground-truth', action='store_true', help='Skip generating ground truth')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("RERUNNING COMPLETE PIPELINE WITH VIETNAMESE TRANSLATION")
    logger.info("=" * 80)
    logger.info("")
    logger.info("This will:")
    logger.info("1. Clear existing embeddings (if --skip-clear not set)")
    logger.info("2. Process datasets with Vietnamese→English translation")
    logger.info("3. Generate ground truth pairs (if --skip-ground-truth not set)")
    logger.info("")
    
    # Step 1: Clear embeddings
    if not args.skip_clear:
        if not clear_embeddings():
            logger.error("Failed to clear embeddings. Exiting.")
            return
    else:
        logger.info("Skipping clearing embeddings (--skip-clear set)")
    
    # Step 2: Process embeddings
    if not process_embeddings(
        jd_file=args.jd_file,
        candidate_file=args.candidate_file,
        batch_size=args.batch_size
    ):
        logger.error("Failed to process embeddings. Exiting.")
        return
    
    # Step 3: Generate ground truth
    if not args.skip_ground_truth:
        if not generate_ground_truth():
            logger.warning("Failed to generate ground truth pairs, but embeddings are processed.")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Train two tower model: python scripts/train_two_tower.py")
    logger.info("2. Evaluate model: python scripts/evaluate_two_tower.py")


if __name__ == "__main__":
    main()


