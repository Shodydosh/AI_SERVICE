"""Comprehensive script to check quality of all embeddings (JD + Candidate)."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.database.repository import EmbeddingRepository
from scripts.check_jd_embeddings_quality import (
    check_jd_embeddings_quality,
    print_quality_report as print_jd_report
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_candidate_embeddings_quality(
    db: Session,
    sample_size: int = 2000,
    similarity_sample_size: int = 5000
):
    """Check quality of candidate embeddings stored in database."""
    from scripts.check_jd_embeddings_quality import (
        check_embedding_normality,
        check_similarity_distribution,
        check_embedding_dimensions
    )
    import random
    import numpy as np
    from tqdm import tqdm
    
    logger.info("=" * 80)
    logger.info("CHECKING CANDIDATE EMBEDDINGS QUALITY")
    logger.info("=" * 80)
    logger.info("")
    
    repository = EmbeddingRepository(db)
    
    # Get all candidate embeddings
    logger.info("Loading candidate embeddings from database...")
    all_candidates = repository.get_all_candidate_embeddings()
    
    logger.info(f"Found {len(all_candidates)} candidate embeddings in database")
    
    if len(all_candidates) == 0:
        logger.warning("No candidate embeddings found in database")
        return None
    
    # Sample embeddings
    sample_size = min(sample_size, len(all_candidates))
    logger.info(f"Sampling {sample_size} candidate embeddings for quality check...")
    candidate_sample = random.sample(all_candidates, sample_size)
    
    # Extract embeddings
    embeddings = [c.embedding for c in candidate_sample]
    
    logger.info("")
    logger.info("Checking embedding dimensions...")
    dim_check = check_embedding_dimensions(embeddings)
    
    logger.info("Checking normalization...")
    norm_check = check_embedding_normality(embeddings)
    
    logger.info("Checking similarity distribution...")
    similarity_check = check_similarity_distribution(
        embeddings,
        sample_size=similarity_sample_size
    )
    
    # Check for zero vectors
    zero_vectors = sum(1 for emb in embeddings if emb and np.allclose(emb, 0.0))
    
    results = {
        'total_candidates': len(all_candidates),
        'sampled_candidates': len(candidate_sample),
        'zero_vectors': zero_vectors,
        'dimensions': dim_check,
        'normalization': norm_check,
        'similarity_distribution': similarity_check
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Check quality of all embeddings (JD + Candidate)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check all with default sample sizes
  python scripts/check_all_embeddings_quality.py
  
  # Check with larger samples
  python scripts/check_all_embeddings_quality.py \\
      --jd-sample 10000 \\
      --candidate-sample 5000 \\
      --similarity-sample 20000
        """
    )
    
    parser.add_argument(
        "--jd-sample",
        type=int,
        default=5000,
        help="Number of JD embeddings to sample (default: 5000)"
    )
    
    parser.add_argument(
        "--candidate-sample",
        type=int,
        default=2000,
        help="Number of candidate embeddings to sample (default: 2000)"
    )
    
    parser.add_argument(
        "--similarity-sample",
        type=int,
        default=10000,
        help="Number of similarity pairs to check (default: 10000)"
    )
    
    parser.add_argument(
        "--skip-candidates",
        action="store_true",
        help="Skip candidate embeddings check"
    )
    
    parser.add_argument(
        "--skip-jds",
        action="store_true",
        help="Skip JD embeddings check"
    )
    
    args = parser.parse_args()
    
    db: Session = SessionLocal()
    try:
        # Check JD embeddings
        if not args.skip_jds:
            logger.info("")
            jd_results = check_jd_embeddings_quality(
                db=db,
                sample_size=args.jd_sample,
                similarity_sample_size=args.similarity_sample
            )
            if jd_results:
                print_jd_report(jd_results)
        
        # Check candidate embeddings
        if not args.skip_candidates:
            logger.info("")
            candidate_results = check_candidate_embeddings_quality(
                db=db,
                sample_size=args.candidate_sample,
                similarity_sample_size=args.similarity_sample
            )
            if candidate_results:
                logger.info("")
                logger.info("=" * 80)
                logger.info("CANDIDATE EMBEDDING QUALITY REPORT")
                logger.info("=" * 80)
                logger.info("")
                logger.info(f"Total Candidate Embeddings: {candidate_results.get('total_candidates', 0)}")
                logger.info(f"Sampled for Check: {candidate_results.get('sampled_candidates', 0)}")
                logger.info(f"Zero Vectors Found: {candidate_results.get('zero_vectors', 0)}")
                logger.info("")
                
                # Dimensions
                dim_check = candidate_results.get('dimensions', {})
                if dim_check:
                    logger.info("EMBEDDING DIMENSIONS")
                    logger.info("-" * 80)
                    logger.info(f"Unique Dimensions: {dim_check.get('unique_dimensions', [])}")
                    logger.info(f"Most Common Dimension: {dim_check.get('most_common_dimension', 'N/A')}")
                    logger.info(f"All Same Dimension: {dim_check.get('all_same_dimension', False)}")
                    logger.info("")
                
                # Normalization
                norm_check = candidate_results.get('normalization', {})
                if norm_check:
                    logger.info("NORMALIZATION CHECK")
                    logger.info("-" * 80)
                    logger.info(f"Mean Norm: {norm_check.get('mean_norm', 0):.6f} (should be ~1.0)")
                    logger.info(f"Std Norm: {norm_check.get('std_norm', 0):.6f} (should be ~0.0)")
                    logger.info(f"Min Norm: {norm_check.get('min_norm', 0):.6f}")
                    logger.info(f"Max Norm: {norm_check.get('max_norm', 0):.6f}")
                    logger.info(f"All Normalized: {norm_check.get('all_normalized', False)}")
                    logger.info(f"Normalized Count: {norm_check.get('normalized_count', 0)}/{norm_check.get('total_count', 0)}")
                    
                    mean_norm = norm_check.get('mean_norm', 0)
                    if 0.99 <= mean_norm <= 1.01:
                        logger.info("✓ Normalization: EXCELLENT")
                    elif 0.95 <= mean_norm <= 1.05:
                        logger.info("⚠ Normalization: GOOD")
                    else:
                        logger.info("✗ Normalization: NEEDS IMPROVEMENT")
                    logger.info("")
                
                # Similarity Distribution
                sim_check = candidate_results.get('similarity_distribution', {})
                if sim_check:
                    logger.info("SIMILARITY DISTRIBUTION (Random Candidate Pairs)")
                    logger.info("-" * 80)
                    logger.info(f"Sample Size: {sim_check.get('sample_size', 0)}")
                    logger.info(f"Mean Similarity: {sim_check.get('mean_similarity', 0):.4f}")
                    logger.info(f"Std Similarity: {sim_check.get('std_similarity', 0):.4f}")
                    logger.info(f"Min Similarity: {sim_check.get('min_similarity', 0):.4f}")
                    logger.info(f"Max Similarity: {sim_check.get('max_similarity', 0):.4f}")
                    logger.info(f"Median Similarity: {sim_check.get('median_similarity', 0):.4f}")
                    logger.info(f"Q25: {sim_check.get('q25_similarity', 0):.4f}")
                    logger.info(f"Q75: {sim_check.get('q75_similarity', 0):.4f}")
                    logger.info(f"Q90: {sim_check.get('q90_similarity', 0):.4f}")
                    logger.info(f"Q95: {sim_check.get('q95_similarity', 0):.4f}")
                    logger.info("")
                    
                    mean_sim = sim_check.get('mean_similarity', 0)
                    std_sim = sim_check.get('std_similarity', 0)
                    logger.info("QUALITY ASSESSMENT:")
                    if mean_sim > 0.3 and std_sim > 0.1:
                        logger.info("✓ Similarity Distribution: GOOD")
                    elif mean_sim > 0.2:
                        logger.info("⚠ Similarity Distribution: ACCEPTABLE")
                    else:
                        logger.info("✗ Similarity Distribution: LOW")
                    logger.info("")
            else:
                logger.warning("No candidate embeddings found in database")
        
        logger.info("=" * 80)
        logger.info("ALL QUALITY CHECKS COMPLETE")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error checking quality: {e}", exc_info=True)
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()

