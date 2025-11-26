"""Test embedding generation and matching on a small sample before full processing."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.services.embedding_service import EmbeddingService
from src.services.matching_service import MatchingService
from src.embeddings.model_selector import EmbeddingModelSelector
from config.settings import settings
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_file(input_file: str, output_file: str, sample_size: int = 2):
    """Create a sample file with first N records."""
    logger.info(f"Creating sample file: {output_file} from {input_file}")
    df = pd.read_csv(input_file, nrows=sample_size)
    df.to_csv(output_file, index=False)
    logger.info(f"✓ Created sample with {len(df)} records")
    return output_file


def test_embeddings(
    jd_file: str = None,
    candidate_file: str = None,
    sample_size: int = 2,
    test_matching: bool = True
):
    """Test embedding generation and matching on a small sample.
    
    Args:
        jd_file: Path to JD dataset
        candidate_file: Path to candidate dataset
        sample_size: Number of records to test (default: 2)
        test_matching: Whether to test matching after embedding
    """
    logger.info("=" * 80)
    logger.info("TESTING EMBEDDING GENERATION ON SAMPLE DATA")
    logger.info("=" * 80)
    
    # Show model info
    model_info = EmbeddingModelSelector().get_model_info(settings.EMBEDDING_MODEL)
    if model_info:
        logger.info(f"Using model: {model_info['name']} ({model_info['dimensions']} dimensions)")
    else:
        logger.info(f"Using model: {settings.EMBEDDING_MODEL} ({settings.EMBEDDING_DIMENSION} dimensions)")
    logger.info("")
    
    # Create sample files
    jd_sample = None
    candidate_sample = None
    
    if jd_file:
        jd_sample = f"data/processed/job_data_sample_{sample_size}.csv"
        create_sample_file(jd_file, jd_sample, sample_size)
        logger.info("")
    
    if candidate_file:
        candidate_sample = f"data/processed/candidates_sample_{sample_size}.csv"
        create_sample_file(candidate_file, candidate_sample, sample_size)
        logger.info("")
    
    db: Session = SessionLocal()
    try:
        # Clear existing test data
        logger.info("Clearing existing test embeddings...")
        from src.database.models import JobDescriptionEmbedding, CandidateEmbedding
        db.query(JobDescriptionEmbedding).delete()
        db.query(CandidateEmbedding).delete()
        db.commit()
        logger.info("✓ Cleared existing embeddings")
        logger.info("")
        
        service = EmbeddingService(db)
        
        # Process JD sample
        if jd_sample:
            logger.info("=" * 80)
            logger.info("PROCESSING JD SAMPLE")
            logger.info("=" * 80)
            
            # Show sample data and embedded text
            jd_df = pd.read_csv(jd_sample)
            logger.info(f"Sample JD records:")
            from src.data_processing.jd_processor import JDProcessor
            jd_processor = JDProcessor()
            jd_processor.load_from_csv(jd_sample)
            
            for idx, row in jd_df.iterrows():
                logger.info(f"  {idx + 1}. Job ID: {row.get('job_id', 'N/A')}")
                logger.info(f"     Title: {row.get('title', 'N/A')[:100]}")
                logger.info(f"     Description: {str(row.get('description', ''))[:150]}...")
                # Show embedded text
                combined_text = jd_processor.get_combined_text(pd.Series(row))
                logger.info(f"     Embedded text: {combined_text[:200]}...")
                logger.info("")
            
            count = service.process_jd_dataset(jd_sample, "csv")
            logger.info(f"✓ Generated embeddings for {count} job descriptions")
            logger.info("")
        
        # Process candidate sample
        if candidate_sample:
            logger.info("=" * 80)
            logger.info("PROCESSING CANDIDATE SAMPLE")
            logger.info("=" * 80)
            
            # Show sample data
            candidate_df = pd.read_csv(candidate_sample)
            logger.info(f"Sample candidate records:")
            for idx, row in candidate_df.iterrows():
                logger.info(f"  {idx + 1}. Candidate ID: {row.get('candidate_id', 'N/A')}")
                logger.info(f"     Name: {row.get('name', 'N/A')}")
                logger.info(f"     Skills: {str(row.get('skills', ''))[:150]}...")
                logger.info("")
            
            count = service.process_candidate_dataset(candidate_sample, "csv")
            logger.info(f"✓ Generated embeddings for {count} candidates")
            logger.info("")
        
        # Test matching if requested
        if test_matching and jd_sample and candidate_sample:
            logger.info("=" * 80)
            logger.info("TESTING MATCHING QUALITY")
            logger.info("=" * 80)
            
            matching_service = MatchingService(db, use_faiss=False)  # Use DB search for small sample
            
            # Get all candidates with valid IDs
            candidate_df = pd.read_csv(candidate_sample)
            from src.database.repository import EmbeddingRepository
            repository = EmbeddingRepository(db)
            all_candidates = repository.get_all_candidate_embeddings()
            
            # Test with 5 candidates: first 2 + 3 random (prefer good data, but use any if needed)
            import random
            candidate_df_full = pd.read_csv(candidate_file if candidate_file else candidate_sample, low_memory=False)
            
            # Score candidates based on available fields
            def score_candidate(row):
                score = 0
                if pd.notna(row.get('skills')) and str(row.get('skills', '')).strip().lower() not in ['nan', 'none', 'null', '']:
                    score += 3
                if pd.notna(row.get('experience')) and str(row.get('experience', '')).strip().lower() not in ['nan', 'none', 'null', '']:
                    score += 2
                if pd.notna(row.get('work_experience')) and str(row.get('work_experience', '')).strip().lower() not in ['nan', 'none', 'null', '']:
                    score += 2
                if pd.notna(row.get('summary')) and str(row.get('summary', '')).strip().lower() not in ['nan', 'none', 'null', '']:
                    score += 2
                if pd.notna(row.get('education')) and str(row.get('education', '')).strip().lower() not in ['nan', 'none', 'null', '']:
                    score += 1
                return score
            
            # Get first 2 candidates
            first_2_ids = [c.candidate_id for c in all_candidates[:2]]
            logger.info(f"First 2 candidates: {first_2_ids}")
            
            # Find candidates with good data (score >= 3) that are not in first 2
            good_candidates = []
            all_other_candidates = []
            for idx, row in candidate_df_full.iterrows():
                cv_id = str(row.get('cv_id', ''))
                if cv_id and cv_id not in first_2_ids:
                    score = score_candidate(row)
                    if score >= 3:
                        good_candidates.append((cv_id, score))
                    all_other_candidates.append(cv_id)
            
            # Select 3 random candidates: prefer good data, but use any if not enough
            if len(good_candidates) >= 3:
                random_candidates_with_data = random.sample(good_candidates, 3)
                random_ids = [cv_id for cv_id, score in random_candidates_with_data]
                logger.info(f"Selected 3 random candidates with good data (score >= 3): {random_ids}")
            elif len(good_candidates) > 0:
                # Use all good candidates + random from others
                random_ids = [cv_id for cv_id, _ in good_candidates]
                remaining_needed = 3 - len(random_ids)
                other_available = [c for c in all_other_candidates if c not in random_ids]
                if remaining_needed > 0 and len(other_available) >= remaining_needed:
                    random_ids.extend(random.sample(other_available, remaining_needed))
                logger.info(f"Selected {len(good_candidates)} good candidates + {3-len(good_candidates)} random: {random_ids}")
            else:
                # Fallback: use any available candidates
                random_ids = random.sample(all_other_candidates, min(3, len(all_other_candidates))) if len(all_other_candidates) >= 3 else all_other_candidates[:3]
                logger.info(f"Using random candidates (no good data found): {random_ids}")
            
            # Get all test candidate IDs
            test_candidate_ids = first_2_ids + random_ids[:3]
            test_candidates = [c for c in all_candidates if c.candidate_id in test_candidate_ids]
            
            # If we don't have enough in DB, add more from available
            if len(test_candidates) < 5 and len(all_candidates) > len(test_candidates):
                remaining = [c for c in all_candidates if c.candidate_id not in test_candidate_ids]
                test_candidates.extend(remaining[:5-len(test_candidates)])
            
            # Limit to 5 candidates total
            test_candidates = test_candidates[:5]
            logger.info(f"Testing with {len(test_candidates)} candidates: {[c.candidate_id for c in test_candidates]}")
            
            total_jds = len(repository.get_all_jd_embeddings())
            top_k = min(10, total_jds)  # Top 10 jobs
            
            for candidate_idx, candidate in enumerate(test_candidates, 1):
                candidate_id = candidate.candidate_id
                logger.info("")
                logger.info("=" * 80)
                logger.info(f"CANDIDATE {candidate_idx}: {candidate_id}")
                logger.info("=" * 80)
                logger.info(f"  Name: {candidate.name or 'N/A'}")
                logger.info(f"  Email: {candidate.email or 'N/A'}")
                logger.info(f"  Skills: {str(candidate.skills)[:150] if candidate.skills else 'N/A'}...")
                logger.info(f"  Experience: {str(candidate.experience)[:150] if candidate.experience else 'N/A'}...")
                logger.info("")
                
                # Show what text was embedded for this candidate
                from src.data_processing.candidate_processor import CandidateProcessor
                processor = CandidateProcessor()
                processor.load_from_csv(candidate_sample)
                candidate_row = candidate_df[candidate_df['cv_id'] == candidate_id] if 'cv_id' in candidate_df.columns else candidate_df.iloc[candidate_idx-1:candidate_idx]
                if len(candidate_row) > 0:
                    combined_text = processor.get_combined_text(candidate_row.iloc[0])
                    logger.info(f"Embedded text for candidate:")
                    logger.info(f"  {combined_text[:400]}...")
                    logger.info("")
                
                # Find top 10 jobs
                matches = matching_service.find_jobs_for_candidate(
                    candidate_id=candidate_id,
                    top_k=top_k,
                    use_faiss=False
                )
                
                logger.info(f"TOP {len(matches)} JOB MATCHES:")
                logger.info("-" * 80)
                for i, match in enumerate(matches, 1):
                    logger.info(f"  {i}. {match['title']} (Similarity: {match['similarity_score']:.4f} = {match['similarity_score']*100:.2f}%)")
                    logger.info(f"     Job ID: {match['job_id']}")
                    if match.get('company'):
                        logger.info(f"     Company: {match['company']}")
                    if match.get('location'):
                        logger.info(f"     Location: {match['location']}")
                    if match.get('description'):
                        logger.info(f"     Description: {match['description'][:150]}...")
                    if match.get('requirements'):
                        logger.info(f"     Requirements: {match['requirements'][:150]}...")
                    logger.info("")
        
        logger.info("=" * 80)
        logger.info("✓ TEST COMPLETE")
        logger.info("=" * 80)
        logger.info("Review the results above. If they look good, proceed with full embedding:")
        logger.info("  python scripts/generate_embeddings_from_processed.py --jd-file data/processed/job_data.csv --candidate-file data/processed/candidates_dataset.csv --faiss-index-type HNSW")
        logger.info("")
        
        return True
    
    except Exception as e:
        logger.error(f"Error testing embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(
        description="Test embedding generation on a small sample",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with 2 records from each dataset
  python scripts/test_embeddings_sample.py --jd-file data/processed/job_data.csv --candidate-file data/processed/candidates_dataset.csv
  
  # Test with 5 records
  python scripts/test_embeddings_sample.py --jd-file data/processed/job_data.csv --candidate-file data/processed/candidates_dataset.csv --sample-size 5
        """
    )
    
    parser.add_argument("--jd-file", type=str, help="Path to JD dataset")
    parser.add_argument("--candidate-file", type=str, help="Path to candidate dataset")
    parser.add_argument("--sample-size", type=int, default=2, help="Number of records to test (default: 2)")
    parser.add_argument("--no-matching-test", action="store_true", help="Skip matching quality test")
    
    args = parser.parse_args()
    
    if not args.jd_file and not args.candidate_file:
        parser.error("At least one of --jd-file or --candidate-file must be provided")
    
    success = test_embeddings(
        jd_file=args.jd_file,
        candidate_file=args.candidate_file,
        sample_size=args.sample_size,
        test_matching=not args.no_matching_test
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

