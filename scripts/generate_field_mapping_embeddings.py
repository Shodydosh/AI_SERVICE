"""Script to generate field-by-field embeddings and store in PostgreSQL."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import pandas as pd
from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.embeddings.field_mapping_embedding import FieldMappingEmbeddingGenerator
from src.data_processing.jd_processor import JDProcessor
from src.data_processing.candidate_processor import CandidateProcessor
from src.database.repository import EmbeddingRepository
from src.embeddings.model_selector import EmbeddingModelSelector
from config.settings import settings
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_field_embeddings_for_jds(
    db: Session,
    jd_file: str,
    file_type: str = "csv",
    batch_size: int = 100
):
    """
    Generate field-by-field embeddings for JDs and store in database.
    Note: This stores the combined embedding (same as before) but prepares for field mapping.
    """
    logger.info("=" * 80)
    logger.info("GENERATING FIELD EMBEDDINGS FOR JOB DESCRIPTIONS")
    logger.info("=" * 80)
    
    # Load JD data
    processor = JDProcessor()
    if file_type.lower() == "csv":
        processor.load_from_csv(jd_file)
    elif file_type.lower() == "json":
        processor.load_from_json(jd_file)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    if not processor.validate_data():
        raise ValueError("JD dataset validation failed")
    
    data = processor.data
    logger.info(f"Loaded {len(data)} job descriptions")
    
    # Initialize embedding generator
    embedding_generator = FieldMappingEmbeddingGenerator()
    repository = EmbeddingRepository(db)
    
    # Process in batches
    total_processed = 0
    embeddings_data = []
    
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
            
            # Generate embeddings for each field
            jd_embeddings = embedding_generator.generate_jd_field_embeddings(jd_fields)
            
            # For storage, we'll use a combined embedding approach
            # Combine field embeddings with weights
            import numpy as np
            combined_embedding = None
            total_weight = 0.0
            
            weights = {'title': 0.3, 'requirements': 0.4, 'description': 0.3}
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
                # Normalize
                norm = np.linalg.norm(combined_embedding)
                if norm > 0:
                    combined_embedding = combined_embedding / norm
                
                embeddings_data.append({
                    'job_id': str(row.get('job_id', f'jd_{idx}')),
                    'title': jd_fields['title'],
                    'description': jd_fields['description'],
                    'requirements': jd_fields['requirements'],
                    'company': jd_fields['company'],
                    'location': jd_fields['location'],
                    'embedding': combined_embedding.tolist()
                })
            
            # Save in batches
            if len(embeddings_data) >= batch_size:
                saved = repository.create_jd_embeddings_batch(embeddings_data, replace_existing=True)
                total_processed += saved
                embeddings_data = []
                logger.info(f"Saved batch: {total_processed} JDs processed")
        
        except Exception as e:
            logger.error(f"Error processing JD at index {idx}: {e}")
            continue
    
    # Save remaining
    if embeddings_data:
        saved = repository.create_jd_embeddings_batch(embeddings_data, replace_existing=True)
        total_processed += saved
    
    logger.info(f"✓ Completed: {total_processed} job descriptions processed")
    return total_processed


def generate_field_embeddings_for_candidates(
    db: Session,
    candidate_file: str,
    file_type: str = "csv",
    batch_size: int = 100
):
    """
    Generate field-by-field embeddings for candidates and store in database.
    """
    logger.info("=" * 80)
    logger.info("GENERATING FIELD EMBEDDINGS FOR CANDIDATES")
    logger.info("=" * 80)
    
    # Load candidate data
    processor = CandidateProcessor()
    if file_type.lower() == "csv":
        processor.load_from_csv(candidate_file)
    elif file_type.lower() == "json":
        processor.load_from_json(candidate_file)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    if not processor.validate_data():
        raise ValueError("Candidate dataset validation failed")
    
    data = processor.data
    logger.info(f"Loaded {len(data)} candidates")
    
    # Initialize embedding generator
    embedding_generator = FieldMappingEmbeddingGenerator()
    repository = EmbeddingRepository(db)
    
    # Process in batches
    total_processed = 0
    embeddings_data = []
    
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Processing candidates"):
        try:
            # Extract fields
            candidate_fields = {
                'skills': str(row.get('skills', '')).strip() if pd.notna(row.get('skills')) else '',
                'experience': str(row.get('experience', '')).strip() if pd.notna(row.get('experience')) else '',
                'desired_job': str(row.get('desired_job_translated', '')).strip() if pd.notna(row.get('desired_job_translated')) else '',
            }
            
            # Generate embeddings for each field
            candidate_embeddings = embedding_generator.generate_candidate_field_embeddings(candidate_fields)
            
            # For storage, combine field embeddings with weights
            import numpy as np
            combined_embedding = None
            total_weight = 0.0
            
            weights = {'skills': 0.4, 'experience': 0.35, 'desired_job': 0.25}
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
                # Normalize
                norm = np.linalg.norm(combined_embedding)
                if norm > 0:
                    combined_embedding = combined_embedding / norm
                
                embeddings_data.append({
                    'candidate_id': str(row.get('candidate_id', f'candidate_{idx}')),
                    'name': str(row.get('name', '')).strip() if pd.notna(row.get('name')) else None,
                    'email': str(row.get('email', '')).strip() if pd.notna(row.get('email')) else None,
                    'skills': candidate_fields['skills'] if candidate_fields['skills'] else None,
                    'experience': candidate_fields['experience'] if candidate_fields['experience'] else None,
                    'education': str(row.get('education', '')).strip() if pd.notna(row.get('education')) else None,
                    'summary': str(row.get('summary', '')).strip() if pd.notna(row.get('summary')) else None,
                    'resume_text': str(row.get('resume_text', '')).strip() if pd.notna(row.get('resume_text')) else None,
                    'embedding': combined_embedding.tolist()
                })
            
            # Save in batches
            if len(embeddings_data) >= batch_size:
                saved = repository.create_candidate_embeddings_batch(embeddings_data, replace_existing=True)
                total_processed += saved
                embeddings_data = []
                logger.info(f"Saved batch: {total_processed} candidates processed")
        
        except Exception as e:
            logger.error(f"Error processing candidate at index {idx}: {e}")
            continue
    
    # Save remaining
    if embeddings_data:
        saved = repository.create_candidate_embeddings_batch(embeddings_data, replace_existing=True)
        total_processed += saved
    
    logger.info(f"✓ Completed: {total_processed} candidates processed")
    return total_processed


def main():
    parser = argparse.ArgumentParser(
        description="Generate field-by-field embeddings and store in PostgreSQL",
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        "--file-type",
        type=str,
        default="csv",
        choices=["csv", "json"],
        help="File format (default: csv)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing (default: 100)"
    )
    
    args = parser.parse_args()
    
    if not args.jd_file and not args.candidate_file:
        parser.error("At least one of --jd-file or --candidate-file must be provided")
    
    # Show model info
    model_info = EmbeddingModelSelector().get_model_info(settings.EMBEDDING_MODEL)
    if model_info:
        logger.info(f"Using embedding model: {model_info['name']}")
        logger.info(f"  Dimensions: {model_info['dimensions']}")
    else:
        logger.info(f"Using embedding model: {settings.EMBEDDING_MODEL}")
    
    logger.info("")
    
    db: Session = SessionLocal()
    try:
        # Process JD dataset
        if args.jd_file:
            generate_field_embeddings_for_jds(
                db=db,
                jd_file=args.jd_file,
                file_type=args.file_type,
                batch_size=args.batch_size
            )
            logger.info("")
        
        # Process candidate dataset
        if args.candidate_file:
            generate_field_embeddings_for_candidates(
                db=db,
                candidate_file=args.candidate_file,
                file_type=args.file_type,
                batch_size=args.batch_size
            )
            logger.info("")
        
        logger.info("=" * 80)
        logger.info("FIELD EMBEDDING GENERATION - Completed Successfully")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error in embedding generation: {e}", exc_info=True)
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()

