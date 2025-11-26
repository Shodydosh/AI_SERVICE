"""Utility script to manage FAISS indices."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.vector_search.faiss_manager import FAISSIndexManager
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_index(dataset_type: str, index_type: str = "HNSW", save_path: str = None):
    """Build FAISS index from database."""
    logger.info(f"Building {dataset_type} FAISS index (type: {index_type})...")
    
    db: Session = SessionLocal()
    try:
        faiss_manager = FAISSIndexManager(
            dimension=settings.EMBEDDING_DIMENSION,
            index_type=index_type,
            index_params={
                "ef_search": 64,
                "ef_construction": 200,
                "M": 32
            } if index_type == "HNSW" else {},
            normalize=True
        )
        
        faiss_manager.build_index_from_db(db, dataset_type=dataset_type)
        
        # Save index
        if save_path:
            faiss_manager.save_index(save_path, dataset_type=dataset_type)
            logger.info(f"✓ Index saved to {save_path}")
        else:
            default_path = f"indices/{dataset_type}_index.faiss"
            faiss_manager.save_index(default_path, dataset_type=dataset_type)
            logger.info(f"✓ Index saved to {default_path}")
        
        # Show stats
        stats = faiss_manager.get_index_stats(dataset_type=dataset_type)
        logger.info(f"Index statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Error building index: {e}")
        return False
    finally:
        db.close()
    
    return True


def load_index(dataset_type: str, index_path: str):
    """Load and display FAISS index information."""
    logger.info(f"Loading {dataset_type} FAISS index from {index_path}...")
    
    try:
        faiss_manager = FAISSIndexManager(
            dimension=settings.EMBEDDING_DIMENSION,
            index_type="HNSW",  # Will be overridden by loaded index
            normalize=True
        )
        
        faiss_manager.load_index(index_path, dataset_type=dataset_type)
        
        # Show stats
        stats = faiss_manager.get_index_stats(dataset_type=dataset_type)
        logger.info("=" * 80)
        logger.info("INDEX STATISTICS")
        logger.info("=" * 80)
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error loading index: {e}")
        return False
    
    return True


def rebuild_index(dataset_type: str, index_type: str = "HNSW"):
    """Rebuild FAISS index from database."""
    logger.info(f"Rebuilding {dataset_type} FAISS index...")
    
    db: Session = SessionLocal()
    try:
        faiss_manager = FAISSIndexManager(
            dimension=settings.EMBEDDING_DIMENSION,
            index_type=index_type,
            index_params={
                "ef_search": 64,
                "ef_construction": 200,
                "M": 32
            } if index_type == "HNSW" else {},
            normalize=True
        )
        
        default_path = f"indices/{dataset_type}_index.faiss"
        faiss_manager.rebuild_index(db, dataset_type=dataset_type, save_path=default_path)
        
        # Show stats
        stats = faiss_manager.get_index_stats(dataset_type=dataset_type)
        logger.info(f"Index statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Error rebuilding index: {e}")
        return False
    finally:
        db.close()
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Manage FAISS indices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build JD index
  python scripts/manage_faiss.py build --dataset-type jd
  
  # Build candidate index with HNSW
  python scripts/manage_faiss.py build --dataset-type candidate --index-type HNSW
  
  # Load and display index stats
  python scripts/manage_faiss.py load --dataset-type jd --index-path indices/jd_index.faiss
  
  # Rebuild index
  python scripts/manage_faiss.py rebuild --dataset-type jd
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build FAISS index from database')
    build_parser.add_argument('--dataset-type', required=True, choices=['jd', 'candidate'],
                             help='Type of dataset')
    build_parser.add_argument('--index-type', default='HNSW', choices=['Flat', 'IVF', 'HNSW'],
                             help='FAISS index type')
    build_parser.add_argument('--save-path', type=str, help='Path to save index')
    
    # Load command
    load_parser = subparsers.add_parser('load', help='Load and display FAISS index')
    load_parser.add_argument('--dataset-type', required=True, choices=['jd', 'candidate'],
                            help='Type of dataset')
    load_parser.add_argument('--index-path', required=True, type=str,
                            help='Path to index file')
    
    # Rebuild command
    rebuild_parser = subparsers.add_parser('rebuild', help='Rebuild FAISS index from database')
    rebuild_parser.add_argument('--dataset-type', required=True, choices=['jd', 'candidate'],
                               help='Type of dataset')
    rebuild_parser.add_argument('--index-type', default='HNSW', choices=['Flat', 'IVF', 'HNSW'],
                               help='FAISS index type')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    success = False
    if args.command == 'build':
        success = build_index(args.dataset_type, args.index_type, args.save_path)
    elif args.command == 'load':
        success = load_index(args.dataset_type, args.index_path)
    elif args.command == 'rebuild':
        success = rebuild_index(args.dataset_type, args.index_type)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

