"""Generate embeddings from processed data and store in PostgreSQL."""
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_embeddings(
    jd_file: str = None,
    candidate_file: str = None,
    file_type: str = "csv",
    build_faiss: bool = True,
    faiss_index_type: str = "Flat"
):
    """
    Generate embeddings from processed data and store in PostgreSQL.
    
    Args:
        jd_file: Path to processed JD dataset
        candidate_file: Path to processed candidate dataset
        file_type: "csv" or "json"
        build_faiss: Whether to build FAISS indices
        faiss_index_type: FAISS index type (Flat, IVF, HNSW)
    """
    logger.info("=" * 80)
    logger.info("GENERATING EMBEDDINGS FROM PROCESSED DATA")
    logger.info("=" * 80)
    
    # Show model info
    model_info = EmbeddingModelSelector().get_model_info(settings.EMBEDDING_MODEL)
    if model_info:
        logger.info(f"Using model: {model_info['name']} ({model_info['dimensions']} dimensions)")
    else:
        logger.info(f"Using model: {settings.EMBEDDING_MODEL} ({settings.EMBEDDING_DIMENSION} dimensions)")
    
    logger.info("")
    
    db: Session = SessionLocal()
    try:
        service = EmbeddingService(db)
        
        # Process JD dataset
        if jd_file:
            logger.info("=" * 80)
            logger.info("PROCESSING JD DATASET")
            logger.info("=" * 80)
            count = service.process_jd_dataset(jd_file, file_type)
            logger.info("")
            logger.info(f"✓ Generated embeddings for {count} job descriptions")
            logger.info("")
        
        # Process candidate dataset
        if candidate_file:
            logger.info("=" * 80)
            logger.info("PROCESSING CANDIDATE DATASET")
            logger.info("=" * 80)
            count = service.process_candidate_dataset(candidate_file, file_type)
            logger.info("")
            logger.info(f"✓ Generated embeddings for {count} candidates")
            logger.info("")
        
        # Build FAISS indices
        if build_faiss:
            logger.info("Building FAISS indices for fast search...")
            dimension = model_info['dimensions'] if model_info else settings.EMBEDDING_DIMENSION
            
            faiss_manager = FAISSIndexManager(
                dimension=dimension,
                index_type=faiss_index_type,
                normalize=True
            )
            
            if jd_file:
                faiss_manager.build_index_from_db(db, dataset_type='jd')
                faiss_manager.save_index('indices/jd_index.faiss', dataset_type='jd')
                logger.info("✓ JD FAISS index built and saved")
            
            if candidate_file:
                faiss_manager.build_index_from_db(db, dataset_type='candidate')
                faiss_manager.save_index('indices/candidate_index.faiss', dataset_type='candidate')
                logger.info("✓ Candidate FAISS index built and saved")
            
            logger.info("")
        
        logger.info("=" * 80)
        logger.info("EMBEDDING GENERATION COMPLETE")
        logger.info("=" * 80)
        logger.info("Embeddings stored in PostgreSQL and ready for matching!")
        if build_faiss:
            logger.info("FAISS indices ready for fast similarity search!")
        logger.info("")
        
        return True
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings from processed data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate embeddings for both datasets
  python scripts/generate_embeddings_from_processed.py --jd-file data/processed/job_data.csv --candidate-file data/processed/candidates_dataset.csv
  
  # Generate only for JD dataset
  python scripts/generate_embeddings_from_processed.py --jd-file data/processed/job_data.csv
  
  # Use HNSW index for large datasets
  python scripts/generate_embeddings_from_processed.py --jd-file data/processed/job_data.csv --candidate-file data/processed/candidates_dataset.csv --faiss-index-type HNSW
        """
    )
    
    parser.add_argument("--jd-file", type=str, help="Path to processed JD dataset")
    parser.add_argument("--candidate-file", type=str, help="Path to processed candidate dataset")
    parser.add_argument("--file-type", type=str, default="csv", choices=["csv", "json"], help="File format")
    parser.add_argument("--no-faiss", action="store_true", help="Skip FAISS index building")
    parser.add_argument("--faiss-index-type", type=str, default="Flat", choices=["Flat", "IVF", "HNSW"], help="FAISS index type")
    
    args = parser.parse_args()
    
    if not args.jd_file and not args.candidate_file:
        parser.error("At least one of --jd-file or --candidate-file must be provided")
    
    success = generate_embeddings(
        jd_file=args.jd_file,
        candidate_file=args.candidate_file,
        file_type=args.file_type,
        build_faiss=not args.no_faiss,
        faiss_index_type=args.faiss_index_type
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

