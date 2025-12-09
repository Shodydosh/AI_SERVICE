"""
Incremental update script for Two-Tower architecture.
Updates FAISS indices with new/updated records.
"""
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.database.two_tower_repository import TwoTowerRepository
from src.vector_search.two_tower_faiss_manager import TwoTowerFAISSManager
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def incremental_upsert(
    db: Session,
    indices_dir: str = "indices/two_tower",
    since_hours: int = 1,
    batch_size: int = 100
):
    """
    Incrementally update FAISS indices with new/updated records.
    
    Args:
        db: Database session
        indices_dir: Directory containing FAISS indices
        since_hours: Only update records updated in last N hours
        batch_size: Batch size for processing
    """
    logger.info("=" * 80)
    logger.info("TWO-TOWER INCREMENTAL UPSERT")
    logger.info("=" * 80)
    
    # Initialize repository
    repository = TwoTowerRepository(db)
    
    # Load existing FAISS indices
    indices_path = Path(indices_dir)
    if not indices_path.exists():
        logger.error(f"Indices directory not found: {indices_path}")
        logger.info("Run batch_reindex_two_tower.py first to create indices")
        return
    
    faiss_manager = TwoTowerFAISSManager(
        dimension=settings.EMBEDDING_DIMENSION,
        index_type="HNSW",
        normalize=True
    )
    faiss_manager.load_indices(indices_path)
    logger.info("✓ Loaded existing FAISS indices")
    
    # Get records updated since last N hours
    since_time = datetime.utcnow() - timedelta(hours=since_hours)
    logger.info(f"Looking for records updated since: {since_time}")
    
    # Get updated jobs
    updated_jobs = repository.get_jobs_updated_since(since_time)
    logger.info(f"Found {len(updated_jobs)} updated jobs")
    
    # Get updated candidates
    updated_candidates = repository.get_candidates_updated_since(since_time)
    logger.info(f"Found {len(updated_candidates)} updated candidates")
    
    # Update FAISS indices
    if updated_jobs:
        logger.info("Updating job indices...")
        faiss_manager.update_job_indices(updated_jobs, batch_size=batch_size)
        logger.info(f"✓ Updated {len(updated_jobs)} jobs")
    
    if updated_candidates:
        logger.info("Updating candidate indices...")
        faiss_manager.update_candidate_indices(updated_candidates, batch_size=batch_size)
        logger.info(f"✓ Updated {len(updated_candidates)} candidates")
    
    # Save updated indices
    faiss_manager.save_indices(indices_path)
    logger.info(f"✓ Saved updated indices to {indices_path}")


def main():
    parser = argparse.ArgumentParser(description="Incremental update Two-Tower FAISS indices")
    parser.add_argument("--indices-dir", type=str, default="indices/two_tower")
    parser.add_argument("--since-hours", type=int, default=1, help="Update records from last N hours")
    parser.add_argument("--batch-size", type=int, default=100)
    
    args = parser.parse_args()
    
    db: Session = next(get_db())
    try:
        incremental_upsert(
            db=db,
            indices_dir=args.indices_dir,
            since_hours=args.since_hours,
            batch_size=args.batch_size
        )
    except Exception as e:
        logger.error(f"Error during incremental update: {e}", exc_info=True)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()


