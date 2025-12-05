"""
Batch reindex script for Two-Tower architecture.
Rebuilds all FAISS indices from PostgreSQL.
"""
import argparse
import logging
from pathlib import Path
from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.vector_search.two_tower_faiss_manager import TwoTowerFAISSManager
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def batch_reindex(
    db: Session,
    index_type: str = "HNSW",
    batch_size: int = 1000,
    output_dir: str = "indices/two_tower"
):
    """
    Rebuild all FAISS indices from PostgreSQL.
    
    Args:
        db: Database session
        index_type: FAISS index type ("HNSW", "IVF", "Flat")
        batch_size: Batch size for processing
        output_dir: Output directory for indices
    """
    logger.info("=" * 80)
    logger.info("TWO-TOWER BATCH REINDEX")
    logger.info("=" * 80)
    
    # Get dimension from settings
    dimension = settings.EMBEDDING_DIMENSION
    
    logger.info(f"Embedding dimension: {dimension}")
    logger.info(f"Index type: {index_type}")
    logger.info(f"Batch size: {batch_size}")
    
    # Initialize FAISS manager
    faiss_manager = TwoTowerFAISSManager(
        dimension=dimension,
        index_type=index_type,
        index_params={
            "M": 32,
            "ef_construction": 200,
            "ef_search": 128
        },
        normalize=True
    )
    
    # Build indices
    logger.info("Building FAISS indices from PostgreSQL...")
    faiss_manager.build_indices_from_db(db, batch_size=batch_size)
    
    # Save indices
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    faiss_manager.save_indices(output_path)
    
    logger.info(f"✓ Indices saved to {output_path}")
    logger.info("  - job_title_index.faiss")
    logger.info("  - job_skills_index.faiss")
    logger.info("  - job_requirement_index.faiss")
    logger.info("  - candidate_title_index.faiss")
    logger.info("  - candidate_skills_index.faiss")
    logger.info("  - candidate_experience_index.faiss")


def main():
    parser = argparse.ArgumentParser(description="Batch reindex Two-Tower FAISS indices")
    parser.add_argument("--index-type", type=str, default="HNSW", choices=["HNSW", "IVF", "Flat"])
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="indices/two_tower")
    
    args = parser.parse_args()
    
    db: Session = next(get_db())
    try:
        batch_reindex(
            db=db,
            index_type=args.index_type,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )
    except Exception as e:
        logger.error(f"Error during reindex: {e}", exc_info=True)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()

