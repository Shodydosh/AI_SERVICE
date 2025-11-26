"""Script to generate embeddings and store in PostgreSQL with FAISS index."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.services.embedding_service import EmbeddingService
from src.embeddings.model_selector import EmbeddingModelSelector
from src.vector_search.faiss_manager import FAISSIndexManager
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_embeddings_workflow(
    jd_file: str = None,
    candidate_file: str = None,
    file_type: str = "csv",
    build_faiss: bool = True,
    faiss_index_type: str = "Flat"
):
    """
    Complete embedding workflow:
    1. Process datasets and generate embeddings
    2. Store embeddings in PostgreSQL
    3. Build FAISS indices for fast search
    """
    logger.info("=" * 80)
    logger.info("EMBEDDING WORKFLOW - Starting")
    logger.info("=" * 80)
    
    # Show selected model
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
        service = EmbeddingService(db)
        
        # Process JD dataset
        if jd_file:
            logger.info("STEP 1: Processing JD Dataset")
            logger.info("-" * 80)
            try:
                count = service.process_jd_dataset(jd_file, file_type)
                logger.info(f"✓ Processed {count} job descriptions")
            except Exception as e:
                logger.error(f"✗ Error processing JD dataset: {e}")
                return False
            logger.info("")
        
        # Process candidate dataset
        if candidate_file:
            logger.info("STEP 2: Processing Candidate Dataset")
            logger.info("-" * 80)
            try:
                count = service.process_candidate_dataset(candidate_file, file_type)
                logger.info(f"✓ Processed {count} candidates")
            except Exception as e:
                logger.error(f"✗ Error processing candidate dataset: {e}")
                return False
            logger.info("")
        
        # Build FAISS indices
        if build_faiss:
            logger.info("STEP 3: Building FAISS Indices")
            logger.info("-" * 80)
            
            dimension = model_info['dimensions'] if model_info else settings.EMBEDDING_DIMENSION
            
            faiss_manager = FAISSIndexManager(
                dimension=dimension,
                index_type=faiss_index_type,
                normalize=True
            )
            
            if jd_file:
                try:
                    faiss_manager.build_index_from_db(db, dataset_type='jd')
                    # Save index
                    faiss_manager.save_index('indices/jd_index.faiss', dataset_type='jd')
                    logger.info("✓ JD FAISS index built and saved")
                except Exception as e:
                    logger.error(f"✗ Error building JD FAISS index: {e}")
            
            if candidate_file:
                try:
                    faiss_manager.build_index_from_db(db, dataset_type='candidate')
                    # Save index
                    faiss_manager.save_index('indices/candidate_index.faiss', dataset_type='candidate')
                    logger.info("✓ Candidate FAISS index built and saved")
                except Exception as e:
                    logger.error(f"✗ Error building candidate FAISS index: {e}")
            
            logger.info("")
        
        logger.info("=" * 80)
        logger.info("EMBEDDING WORKFLOW - Completed Successfully")
        logger.info("=" * 80)
        logger.info("Embeddings are stored in PostgreSQL and ready for use!")
        if build_faiss:
            logger.info("FAISS indices are built and saved for fast similarity search!")
        logger.info("")
        
        return True
    
    except Exception as e:
        logger.error(f"Error in embedding workflow: {e}")
        return False
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings and store in PostgreSQL with FAISS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process both datasets
  python scripts/generate_embeddings.py --jd-file data/jd_processed.csv --candidate-file data/candidate_processed.csv
  
  # Process only JD dataset
  python scripts/generate_embeddings.py --jd-file data/jd_processed.csv
  
  # Process without building FAISS index
  python scripts/generate_embeddings.py --jd-file data/jd_processed.csv --no-faiss
  
  # Use HNSW index for faster search (large datasets)
  python scripts/generate_embeddings.py --jd-file data/jd_processed.csv --faiss-index-type HNSW
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
        "--file-type",
        type=str,
        default="csv",
        choices=["csv", "json"],
        help="File format (default: csv)"
    )
    
    parser.add_argument(
        "--no-faiss",
        action="store_true",
        help="Skip FAISS index building"
    )
    
    parser.add_argument(
        "--faiss-index-type",
        type=str,
        default="Flat",
        choices=["Flat", "IVF", "HNSW"],
        help="FAISS index type (default: Flat)"
    )
    
    args = parser.parse_args()
    
    if not args.jd_file and not args.candidate_file:
        parser.error("At least one of --jd-file or --candidate-file must be provided")
    
    # Run workflow
    success = generate_embeddings_workflow(
        jd_file=args.jd_file,
        candidate_file=args.candidate_file,
        file_type=args.file_type,
        build_faiss=not args.no_faiss,
        faiss_index_type=args.faiss_index_type
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

