"""Script to process datasets and create multi-field embeddings."""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from sqlalchemy.orm import Session
from src.database.connection import get_db, engine, Base
from src.services.multi_field_embedding_service import MultiFieldEmbeddingService
# Import models to register them with Base.metadata
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


def init_multi_field_tables():
    """Initialize multi-field embedding tables if they don't exist."""
    logger.info("=" * 80)
    logger.info("INITIALIZING DATABASE TABLES")
    logger.info("=" * 80)
    
    try:
        logger.info("Creating multi-field embedding tables...")
        Base.metadata.create_all(bind=engine, tables=[
            JobDescriptionMultiEmbedding.__table__,
            CandidateMultiEmbedding.__table__
        ])
        logger.info("✓ Multi-field embedding tables created/verified successfully!")
        logger.info("")
        return True
    except Exception as e:
        logger.error(f"✗ Error creating tables: {e}", exc_info=True)
        logger.error("")
        logger.error("Please make sure:")
        logger.error("1. PostgreSQL database is running")
        logger.error("2. Database connection settings are correct in config/settings.py")
        logger.error("3. Database user has CREATE TABLE permissions")
        return False


def process_job_descriptions(jd_file: str, batch_size: int = 100):
    """Process job description dataset and create multi-field embeddings."""
    # Initialize tables first
    if not init_multi_field_tables():
        logger.error("Cannot proceed without database tables. Exiting.")
        return 0
    
    logger.info("=" * 80)
    logger.info("PROCESSING JOB DESCRIPTIONS")
    logger.info("=" * 80)
    
    db: Session = next(get_db())
    try:
        service = MultiFieldEmbeddingService(db)
        total_processed = service.process_jd_dataset(
            file_path=jd_file,
            file_type="csv",
            batch_size=batch_size
        )
        logger.info(f"✓ Successfully processed {total_processed} job descriptions")
        return total_processed
    except Exception as e:
        logger.error(f"✗ Error processing job descriptions: {e}", exc_info=True)
        raise
    finally:
        db.close()


def process_candidates(candidate_file: str, batch_size: int = 100):
    """Process candidate dataset and create multi-field embeddings."""
    # Initialize tables first
    if not init_multi_field_tables():
        logger.error("Cannot proceed without database tables. Exiting.")
        return 0
    
    logger.info("=" * 80)
    logger.info("PROCESSING CANDIDATES")
    logger.info("=" * 80)
    
    db: Session = next(get_db())
    try:
        service = MultiFieldEmbeddingService(db)
        total_processed = service.process_candidate_dataset(
            file_path=candidate_file,
            file_type="csv",
            batch_size=batch_size
        )
        logger.info(f"✓ Successfully processed {total_processed} candidates")
        return total_processed
    except Exception as e:
        logger.error(f"✗ Error processing candidates: {e}", exc_info=True)
        raise
    finally:
        db.close()


def build_faiss_indices():
    """Build and save multi-field FAISS indices from PostgreSQL."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("BUILDING FAISS INDICES")
    logger.info("=" * 80)
    
    from src.embeddings.multi_field_generator import MultiFieldEmbeddingGenerator
    from src.vector_search.multi_field_faiss_manager import MultiFieldFAISSManager
    
    db: Session = next(get_db())
    try:
        # Get embedding dimension
        embedding_generator = MultiFieldEmbeddingGenerator()
        dimension = embedding_generator.get_embedding_dimension()
        
        # Initialize FAISS manager
        faiss_manager = MultiFieldFAISSManager(
            dimension=dimension,
            index_type="HNSW",
            index_params={
                "ef_search": 128,
                "ef_construction": 200,
                "M": 32
            },
            normalize=True
        )
        
        # Build indices from database
        logger.info("Building multi-field FAISS indices from PostgreSQL...")
        logger.info("This may take a few minutes for large datasets...")
        faiss_manager.build_indices_from_db(db, batch_size=1000)
        
        # Save indices to disk
        base_path = Path("indices/multi_field")
        base_path.mkdir(parents=True, exist_ok=True)
        faiss_manager.save_indices(base_path)
        
        logger.info("")
        logger.info(f"✓ FAISS indices saved to {base_path}")
        logger.info("  - jd_title_index.faiss")
        logger.info("  - jd_skills_index.faiss")
        logger.info("  - jd_requirement_index.faiss")
        logger.info("")
        
        return True
    except Exception as e:
        logger.error(f"✗ Error building FAISS indices: {e}", exc_info=True)
        logger.error("")
        logger.error("You can build FAISS indices later using:")
        logger.error("  python scripts/build_multi_field_faiss.py")
        logger.error("")
        return False
    finally:
        db.close()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process datasets and create multi-field embeddings')
    parser.add_argument('--jd-file', type=str, help='Path to job description CSV file')
    parser.add_argument('--candidate-file', type=str, help='Path to candidate CSV file')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing (default: 100)')
    parser.add_argument('--process-all', action='store_true', help='Process both JD and candidate datasets')
    parser.add_argument('--build-faiss', action='store_true', help='Build FAISS indices after processing embeddings')
    
    args = parser.parse_args()
    
    embeddings_processed = False
    
    if args.process_all:
        # Process both datasets
        project_root = Path(__file__).parent.parent
        jd_file = args.jd_file or str(project_root / "data" / "raw" / "job_data.csv")
        candidate_file = args.candidate_file or str(project_root / "data" / "raw" / "candidates_dataset.csv")
        
        logger.info("Processing all datasets...")
        
        # Process job descriptions first
        if os.path.exists(jd_file):
            process_job_descriptions(jd_file, batch_size=args.batch_size)
            embeddings_processed = True
        else:
            logger.warning(f"JD file not found: {jd_file}")
        
        # Process candidates
        if os.path.exists(candidate_file):
            process_candidates(candidate_file, batch_size=args.batch_size)
            embeddings_processed = True
        else:
            logger.warning(f"Candidate file not found: {candidate_file}")
    else:
        if args.jd_file:
            process_job_descriptions(args.jd_file, batch_size=args.batch_size)
            embeddings_processed = True
        elif args.candidate_file:
            process_candidates(args.candidate_file, batch_size=args.batch_size)
            embeddings_processed = True
        else:
            parser.print_help()
            logger.error("Please specify --jd-file or --candidate-file, or use --process-all")
            return
    
    # Build FAISS indices if requested
    if args.build_faiss and embeddings_processed:
        logger.info("")
        build_faiss_indices()
    elif args.build_faiss and not embeddings_processed:
        logger.warning("")
        logger.warning("Warning: --build-faiss specified but no embeddings were processed.")
        logger.warning("Skipping FAISS index building.")
        logger.warning("")


if __name__ == "__main__":
    main()

