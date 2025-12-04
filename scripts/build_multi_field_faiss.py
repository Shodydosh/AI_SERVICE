"""Script to build and save multi-field FAISS indices from PostgreSQL."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.embeddings.multi_field_generator import MultiFieldEmbeddingGenerator
from src.vector_search.multi_field_faiss_manager import MultiFieldFAISSManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_faiss_indices(
    index_type: str = "HNSW",
    index_params: dict = None,
    base_path: str = "indices/multi_field"
):
    """
    Build and save multi-field FAISS indices from PostgreSQL.
    
    Args:
        index_type: Type of FAISS index ("Flat", "IVF", "HNSW")
        index_params: Parameters for index construction
        base_path: Base path to save indices
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info("BUILDING MULTI-FIELD FAISS INDICES")
    logger.info("=" * 80)
    
    # Default index parameters
    if index_params is None:
        index_params = {
            "ef_search": 128,  # Higher for better precision at k=1000
            "ef_construction": 200,
            "M": 32
        }
    
    db: Session = next(get_db())
    try:
        # Get embedding dimension
        logger.info("Getting embedding dimension...")
        embedding_generator = MultiFieldEmbeddingGenerator()
        dimension = embedding_generator.get_embedding_dimension()
        logger.info(f"✓ Embedding dimension: {dimension}")
        logger.info("")
        
        # Initialize FAISS manager
        logger.info(f"Initializing FAISS manager (index type: {index_type})...")
        faiss_manager = MultiFieldFAISSManager(
            dimension=dimension,
            index_type=index_type,
            index_params=index_params,
            normalize=True
        )
        logger.info("✓ FAISS manager initialized")
        logger.info("")
        
        # Build indices from database
        logger.info("Building indices from PostgreSQL database...")
        logger.info("This may take a few minutes for large datasets...")
        faiss_manager.build_indices_from_db(db, batch_size=1000)
        logger.info("✓ Indices built successfully")
        logger.info("")
        
        # Save indices to disk
        logger.info(f"Saving indices to: {base_path}")
        base_path_obj = Path(base_path)
        base_path_obj.mkdir(parents=True, exist_ok=True)
        faiss_manager.save_indices(base_path)
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ FAISS INDICES BUILT AND SAVED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Saved to: {base_path}")
        logger.info("")
        logger.info("Files created:")
        logger.info("  - jd_title_index.faiss")
        logger.info("  - jd_skills_index.faiss")
        logger.info("  - jd_requirement_index.faiss")
        logger.info("  - jd_title_id_map.pkl")
        logger.info("  - jd_skills_id_map.pkl")
        logger.info("  - jd_requirement_id_map.pkl")
        logger.info("  - jd_title_reverse_map.pkl")
        logger.info("  - jd_skills_reverse_map.pkl")
        logger.info("  - jd_requirement_reverse_map.pkl")
        logger.info("")
        
        # Show stats
        try:
            title_count = faiss_manager.jd_title_index.ntotal if faiss_manager.jd_title_index else 0
            logger.info(f"Index statistics:")
            logger.info(f"  - Title index: {title_count} vectors")
            logger.info(f"  - Skills index: {title_count} vectors")
            logger.info(f"  - Requirement index: {title_count} vectors")
        except Exception:
            pass
        
        logger.info("")
        return True
        
    except Exception as e:
        logger.error("")
        logger.error("=" * 80)
        logger.error("✗ ERROR BUILDING FAISS INDICES")
        logger.error("=" * 80)
        logger.error(f"Error: {e}", exc_info=True)
        logger.error("")
        logger.error("Please check:")
        logger.error("1. PostgreSQL database is running and accessible")
        logger.error("2. Embeddings exist in the database (run process_multi_field_embeddings.py first)")
        logger.error("3. Database connection settings are correct")
        logger.error("")
        return False
    finally:
        db.close()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Build and save multi-field FAISS indices from PostgreSQL'
    )
    parser.add_argument(
        '--index-type',
        type=str,
        default='HNSW',
        choices=['Flat', 'IVF', 'HNSW'],
        help='FAISS index type (default: HNSW)'
    )
    parser.add_argument(
        '--base-path',
        type=str,
        default='indices/multi_field',
        help='Base path to save indices (default: indices/multi_field)'
    )
    parser.add_argument(
        '--ef-search',
        type=int,
        default=128,
        help='ef_search parameter for HNSW index (default: 128)'
    )
    parser.add_argument(
        '--ef-construction',
        type=int,
        default=200,
        help='ef_construction parameter for HNSW index (default: 200)'
    )
    parser.add_argument(
        '--M',
        type=int,
        default=32,
        help='M parameter for HNSW index (default: 32)'
    )
    
    args = parser.parse_args()
    
    # Prepare index parameters
    index_params = {
        "ef_search": args.ef_search,
        "ef_construction": args.ef_construction,
        "M": args.M
    }
    
    success = build_faiss_indices(
        index_type=args.index_type,
        index_params=index_params,
        base_path=args.base_path
    )
    
    if success:
        logger.info("✓ All done! FAISS indices are ready for fast similarity search.")
        return 0
    else:
        logger.error("✗ Failed to build FAISS indices.")
        return 1


if __name__ == "__main__":
    exit(main())

