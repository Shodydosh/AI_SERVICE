"""Script to re-embed all data with improved field-by-field approach."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import time
from sqlalchemy.orm import Session
from sqlalchemy import text
from src.database.connection import SessionLocal, engine, Base
from src.embeddings.field_mapping_embedding import FieldMappingEmbeddingGenerator
from src.embeddings.improved_field_mapping_embedding import ImprovedFieldMappingEmbeddingGenerator
from src.embeddings.advanced_field_mapping_embedding import AdvancedFieldMappingEmbeddingGenerator
from src.data_processing.jd_processor import JDProcessor
from src.data_processing.candidate_processor import CandidateProcessor
from src.database.repository import EmbeddingRepository
from src.embeddings.model_selector import EmbeddingModelSelector
from config.settings import settings
from tqdm import tqdm
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clear_existing_embeddings(db: Session, clear_jd: bool = True, clear_candidates: bool = True, clear_recommendations: bool = True):
    """Clear existing embeddings from database."""
    logger.info("Clearing existing embeddings...")
    
    try:
        if clear_recommendations:
            logger.info("  Clearing processed recommendations...")
            db.execute(text("DELETE FROM processed_candidate_recommendations"))
            db.commit()
            logger.info("  ✓ Processed recommendations cleared")
        
        if clear_candidates:
            logger.info("  Clearing candidate embeddings...")
            db.execute(text("DELETE FROM candidate_embeddings"))
            db.commit()
            logger.info("  ✓ Candidate embeddings cleared")
        
        if clear_jd:
            logger.info("  Clearing JD embeddings...")
            db.execute(text("DELETE FROM job_description_embeddings"))
            db.commit()
            logger.info("  ✓ JD embeddings cleared")
        
        logger.info("✓ All existing embeddings cleared")
        return True
    except Exception as e:
        logger.error(f"Error clearing embeddings: {e}")
        db.rollback()
        return False


def reembed_jds(
    db: Session,
    jd_file: str,
    batch_size: int = 50,
    embedding_batch_size: int = 32
):
    """Re-embed all job descriptions with improved field-by-field approach."""
    logger.info("=" * 80)
    logger.info("RE-EMBEDDING JOB DESCRIPTIONS")
    logger.info("=" * 80)
    
    # Load JD data
    processor = JDProcessor()
    processor.load_from_csv(jd_file)
    
    if not processor.validate_data():
        raise ValueError("JD dataset validation failed")
    
    data = processor.data
    logger.info(f"Loaded {len(data)} job descriptions")
    
    # Initialize advanced embedding generator (best quality)
    embedding_generator = AdvancedFieldMappingEmbeddingGenerator(
        use_semantic_expansion=True,
        use_keyword_boost=True
    )
    repository = EmbeddingRepository(db)
    
    # Process in batches
    total_processed = 0
    embeddings_data = []
    start_time = time.time()
    
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Processing JDs"):
        try:
            # Extract fields
            jd_fields = {
                'title': str(row.get('title', '')).strip() if pd.notna(row.get('title')) else '',
                'requirements': str(row.get('requirements', '')).strip() if pd.notna(row.get('requirements')) else '',
                'description': str(row.get('description', '')).strip() if pd.notna(row.get('description')) else '',
                'company': str(row.get('company', '')).strip() if pd.notna(row.get('company')) else '',
                'location': str(row.get('location', '')).strip() if pd.notna(row.get('location')) else '',
            }
            
            # Generate field embeddings
            jd_embeddings = embedding_generator.generate_jd_field_embeddings(jd_fields)
            
            # Combine field embeddings using advanced method
            weights = {
                'title': 0.25,
                'requirements': 0.50,  # Increased for advanced method
                'description': 0.25
            }
            
            # Get content lengths for attention weighting
            content_lengths = {k: len(v) for k, v in jd_fields.items() if v}
            
            # Use advanced combination method
            if isinstance(embedding_generator, AdvancedFieldMappingEmbeddingGenerator):
                combined_embedding = embedding_generator.combine_field_embeddings_advanced(
                    jd_embeddings,
                    weights,
                    content_lengths
                )
            elif isinstance(embedding_generator, ImprovedFieldMappingEmbeddingGenerator):
                combined_embedding = embedding_generator.combine_field_embeddings(
                    jd_embeddings,
                    weights,
                    content_lengths
                )
            else:
                # Fallback to weighted average
                combined_embedding = None
                total_weight = 0.0
                for field, weight in weights.items():
                    if field in jd_embeddings:
                        field_emb = np.array(jd_embeddings[field])
                        if combined_embedding is None:
                            combined_embedding = field_emb * weight
                        else:
                            combined_embedding += field_emb * weight
                        total_weight += weight
                if combined_embedding is not None and total_weight > 0:
                    combined_embedding = combined_embedding / total_weight
                    norm = np.linalg.norm(combined_embedding)
                    if norm > 0:
                        combined_embedding = combined_embedding / norm
                    combined_embedding = combined_embedding.tolist()
                else:
                    combined_embedding = [0.0] * embedding_generator.get_embedding_dimension()
            
            embeddings_data.append({
                'job_id': str(row.get('job_id', f'jd_{idx}')),
                'title': jd_fields['title'],
                'description': jd_fields['description'],
                'requirements': jd_fields['requirements'],
                'company': jd_fields['company'],
                'location': jd_fields['location'],
                'embedding': combined_embedding if isinstance(combined_embedding, list) else combined_embedding.tolist()
            })
            
            # Save in batches
            if len(embeddings_data) >= batch_size:
                saved = repository.create_jd_embeddings_batch(embeddings_data, replace_existing=True)
                total_processed += saved
                embeddings_data = []
                
                # Log progress
                elapsed = time.time() - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                logger.info(f"Progress: {total_processed}/{len(data)} JDs ({rate:.1f} JDs/sec)")
        
        except Exception as e:
            logger.error(f"Error processing JD at index {idx}: {e}")
            continue
    
    # Save remaining
    if embeddings_data:
        saved = repository.create_jd_embeddings_batch(embeddings_data, replace_existing=True)
        total_processed += saved
    
    elapsed = time.time() - start_time
    logger.info("")
    logger.info(f"✓ Completed: {total_processed} job descriptions processed in {elapsed:.1f}s")
    logger.info(f"  Average rate: {total_processed/elapsed:.1f} JDs/sec")
    return total_processed


def reembed_candidates(
    db: Session,
    candidate_file: str,
    batch_size: int = 50,
    embedding_batch_size: int = 32
):
    """Re-embed all candidates with improved field-by-field approach."""
    logger.info("=" * 80)
    logger.info("RE-EMBEDDING CANDIDATES")
    logger.info("=" * 80)
    
    # Load candidate data
    processor = CandidateProcessor()
    processor.load_from_csv(candidate_file)
    
    data = processor.data
    logger.info(f"Loaded {len(data)} candidates")
    
    # Create candidate_id if missing (use index)
    if 'candidate_id' not in data.columns:
        logger.warning("candidate_id column not found, creating from index")
        data['candidate_id'] = data.index.astype(str).map(lambda x: f'candidate_{x}')
    
    # Validate required fields exist
    required_fields = ['skills', 'experience']
    missing_fields = [f for f in required_fields if f not in data.columns]
    if missing_fields:
        logger.warning(f"Some fields missing: {missing_fields}, will use empty strings")
    
    # Initialize advanced embedding generator (best quality)
    embedding_generator = AdvancedFieldMappingEmbeddingGenerator(
        use_semantic_expansion=True,
        use_keyword_boost=True
    )
    repository = EmbeddingRepository(db)
    
    # Process in batches
    total_processed = 0
    embeddings_data = []
    start_time = time.time()
    
    # Use FieldMappingMatchingService for better field extraction
    from src.services.field_mapping_matching_service import FieldMappingMatchingService
    temp_service = FieldMappingMatchingService(None)
    
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Processing candidates"):
        try:
            # Use improved field extraction with fallbacks
            candidate_fields = temp_service.extract_candidate_fields(row)
            
            # If no fields found, try to create minimal fields from available data
            if not candidate_fields:
                logger.warning(f"No candidate fields found for row {idx}, trying fallbacks")
                candidate_fields = {}
                if pd.notna(row.get('summary')):
                    candidate_fields['experience'] = str(row.get('summary', ''))[:500]
                elif pd.notna(row.get('resume_text')):
                    candidate_fields['experience'] = str(row.get('resume_text', ''))[:500]
            
            # Get candidate_id (create if missing)
            candidate_id = str(row.get('candidate_id', f'candidate_{idx}'))
            
            # Generate field embeddings
            candidate_embeddings = embedding_generator.generate_candidate_field_embeddings(candidate_fields)
            
            # Check if we have any embeddings
            if not candidate_embeddings:
                logger.warning(f"No embeddings generated for candidate {candidate_id}, skipping")
                continue
            
            # Combine field embeddings using advanced method
            weights = {
                'skills': 0.48,      # Optimized for advanced method
                'experience': 0.35,
                'desired_job': 0.17  # Optimized for advanced method
            }
            
            # Get content lengths for attention weighting
            content_lengths = {k: len(str(v)) for k, v in candidate_fields.items() if v}
            
            # Use advanced combination method
            if isinstance(embedding_generator, AdvancedFieldMappingEmbeddingGenerator):
                combined_embedding = embedding_generator.combine_field_embeddings_advanced(
                    candidate_embeddings,
                    weights,
                    content_lengths
                )
            elif isinstance(embedding_generator, ImprovedFieldMappingEmbeddingGenerator):
                combined_embedding = embedding_generator.combine_field_embeddings(
                    candidate_embeddings,
                    weights,
                    content_lengths
                )
            else:
                # Fallback to weighted average
                combined_embedding = None
                total_weight = 0.0
                for field, weight in weights.items():
                    if field in candidate_embeddings:
                        field_emb = np.array(candidate_embeddings[field])
                        if combined_embedding is None:
                            combined_embedding = field_emb * weight
                        else:
                            combined_embedding += field_emb * weight
                        total_weight += weight
                if combined_embedding is not None and total_weight > 0:
                    combined_embedding = combined_embedding / total_weight
                    norm = np.linalg.norm(combined_embedding)
                    if norm > 0:
                        combined_embedding = combined_embedding / norm
                    combined_embedding = combined_embedding.tolist()
                else:
                    logger.warning(f"Failed to combine embeddings for candidate {candidate_id}")
                    combined_embedding = [0.0] * embedding_generator.get_embedding_dimension()
            
            embeddings_data.append({
                'candidate_id': candidate_id,
                'name': str(row.get('name', '')).strip() if pd.notna(row.get('name')) else None,
                'email': str(row.get('email', '')).strip() if pd.notna(row.get('email')) else None,
                'skills': candidate_fields['skills'] if candidate_fields['skills'] else None,
                'experience': candidate_fields['experience'] if candidate_fields['experience'] else None,
                'education': str(row.get('education', '')).strip() if pd.notna(row.get('education')) else None,
                'summary': str(row.get('summary', '')).strip() if pd.notna(row.get('summary')) else None,
                'resume_text': str(row.get('resume_text', '')).strip() if pd.notna(row.get('resume_text')) else None,
                'embedding': combined_embedding if isinstance(combined_embedding, list) else combined_embedding.tolist()
            })
            
            # Save in batches
            if len(embeddings_data) >= batch_size:
                saved = repository.create_candidate_embeddings_batch(embeddings_data, replace_existing=True)
                total_processed += saved
                embeddings_data = []
                
                # Log progress
                elapsed = time.time() - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                logger.info(f"Progress: {total_processed}/{len(data)} candidates ({rate:.1f} candidates/sec)")
        
        except Exception as e:
            logger.error(f"Error processing candidate at index {idx}: {e}")
            continue
    
    # Save remaining
    if embeddings_data:
        saved = repository.create_candidate_embeddings_batch(embeddings_data, replace_existing=True)
        total_processed += saved
    
    elapsed = time.time() - start_time
    logger.info("")
    logger.info(f"✓ Completed: {total_processed} candidates processed in {elapsed:.1f}s")
    logger.info(f"  Average rate: {total_processed/elapsed:.1f} candidates/sec")
    return total_processed


def main():
    parser = argparse.ArgumentParser(
        description="Re-embed all data with improved field-by-field approach",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Re-embed everything
  python scripts/reembed_all_data.py \\
      --jd-file data/processed/jd_processed.csv \\
      --candidate-file data/processed/candidate_processed.csv \\
      --clear-all
  
  # Re-embed only candidates
  python scripts/reembed_all_data.py \\
      --candidate-file data/processed/candidate_processed.csv \\
      --clear-candidates
  
  # Re-embed with custom batch size
  python scripts/reembed_all_data.py \\
      --jd-file data/processed/jd_processed.csv \\
      --candidate-file data/processed/candidate_processed.csv \\
      --batch-size 100
        """
    )
    
    parser.add_argument(
        "--jd-file",
        type=str,
        help="Path to JD dataset file"
    )
    
    parser.add_argument(
        "--candidate-file",
        type=str,
        help="Path to candidate dataset file"
    )
    
    parser.add_argument(
        "--clear-all",
        action="store_true",
        help="Clear all existing embeddings and recommendations"
    )
    
    parser.add_argument(
        "--clear-jd",
        action="store_true",
        help="Clear existing JD embeddings"
    )
    
    parser.add_argument(
        "--clear-candidates",
        action="store_true",
        help="Clear existing candidate embeddings"
    )
    
    parser.add_argument(
        "--clear-recommendations",
        action="store_true",
        help="Clear existing processed recommendations"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for database operations (default: 50)"
    )
    
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32)"
    )
    
    args = parser.parse_args()
    
    if not args.jd_file and not args.candidate_file:
        parser.error("At least one of --jd-file or --candidate-file must be provided")
    
    # Show model info
    model_info = EmbeddingModelSelector().get_model_info(settings.EMBEDDING_MODEL)
    if model_info:
        logger.info(f"Using embedding model: {model_info['name']}")
        logger.info(f"  Dimensions: {model_info['dimensions']}")
        logger.info(f"  Performance: {model_info['performance']}")
    else:
        logger.info(f"Using embedding model: {settings.EMBEDDING_MODEL}")
        logger.info(f"  Dimensions: {settings.EMBEDDING_DIMENSION}")
    logger.info("")
    
    db: Session = SessionLocal()
    try:
        # Clear existing data if requested
        if args.clear_all:
            clear_existing_embeddings(db, clear_jd=True, clear_candidates=True, clear_recommendations=True)
        else:
            clear_existing_embeddings(
                db,
                clear_jd=args.clear_jd,
                clear_candidates=args.clear_candidates,
                clear_recommendations=args.clear_recommendations
            )
        logger.info("")
        
        # Re-embed JD dataset
        if args.jd_file:
            reembed_jds(
                db=db,
                jd_file=args.jd_file,
                batch_size=args.batch_size,
                embedding_batch_size=args.embedding_batch_size
            )
            logger.info("")
        
        # Re-embed candidate dataset
        if args.candidate_file:
            reembed_candidates(
                db=db,
                candidate_file=args.candidate_file,
                batch_size=args.batch_size,
                embedding_batch_size=args.embedding_batch_size
            )
            logger.info("")
        
        logger.info("=" * 80)
        logger.info("RE-EMBEDDING - Completed Successfully")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Generate recommendations: python scripts/generate_processed_recommendations.py")
        logger.info("  2. Or run full workflow: python scripts/run_full_workflow.py --skip-embeddings")
        logger.info("")
        
    except Exception as e:
        logger.error(f"Error in re-embedding: {e}", exc_info=True)
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()

