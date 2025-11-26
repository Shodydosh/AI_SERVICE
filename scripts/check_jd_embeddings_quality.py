"""Script to check JD embeddings quality from database."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import random
import numpy as np
from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.database.repository import EmbeddingRepository
from tqdm import tqdm
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_embedding_normality(embeddings: List[List[float]]) -> dict:
    """Check if embeddings are properly normalized."""
    if not embeddings:
        return {}
    
    norms = []
    for emb in embeddings:
        if emb:
            norm = np.linalg.norm(emb)
            norms.append(norm)
    
    if not norms:
        return {}
    
    norms_array = np.array(norms)
    
    return {
        'mean_norm': float(np.mean(norms_array)),
        'std_norm': float(np.std(norms_array)),
        'min_norm': float(np.min(norms_array)),
        'max_norm': float(np.max(norms_array)),
        'all_normalized': np.allclose(norms_array, 1.0, atol=1e-5),
        'normalized_count': int(np.sum(np.isclose(norms_array, 1.0, atol=1e-5))),
        'total_count': len(norms)
    }


def check_similarity_distribution(
    embeddings: List[List[float]],
    sample_size: int = 1000
) -> dict:
    """Check similarity distribution between random pairs."""
    if len(embeddings) < 2:
        return {}
    
    # Sample random pairs
    sample_size = min(sample_size, len(embeddings) * (len(embeddings) - 1) // 2)
    similarities = []
    
    logger.info(f"Calculating similarities for {sample_size} random pairs...")
    for _ in tqdm(range(sample_size), desc="Similarity calculation"):
        # Pick two different random indices
        idx1, idx2 = random.sample(range(len(embeddings)), 2)
        
        vec1 = np.array(embeddings[idx1])
        vec2 = np.array(embeddings[idx2])
        
        # Calculate cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 > 0 and norm2 > 0:
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            similarities.append(float(similarity))
    
    if not similarities:
        return {}
    
    similarities_array = np.array(similarities)
    
    return {
        'mean_similarity': float(np.mean(similarities_array)),
        'std_similarity': float(np.std(similarities_array)),
        'min_similarity': float(np.min(similarities_array)),
        'max_similarity': float(np.max(similarities_array)),
        'median_similarity': float(np.median(similarities_array)),
        'q25_similarity': float(np.percentile(similarities_array, 25)),
        'q75_similarity': float(np.percentile(similarities_array, 75)),
        'q90_similarity': float(np.percentile(similarities_array, 90)),
        'q95_similarity': float(np.percentile(similarities_array, 95)),
        'sample_size': len(similarities)
    }


def check_embedding_dimensions(embeddings: List[List[float]]) -> dict:
    """Check embedding dimensions consistency."""
    if not embeddings:
        return {}
    
    dimensions = [len(emb) for emb in embeddings if emb]
    
    if not dimensions:
        return {}
    
    unique_dims = set(dimensions)
    
    return {
        'unique_dimensions': sorted(list(unique_dims)),
        'most_common_dimension': max(set(dimensions), key=dimensions.count),
        'dimension_count': len(unique_dims),
        'all_same_dimension': len(unique_dims) == 1,
        'total_embeddings': len(embeddings)
    }


def check_jd_embeddings_quality(
    db: Session,
    sample_size: int = 2000,
    similarity_sample_size: int = 5000
):
    """Check quality of JD embeddings stored in database."""
    logger.info("=" * 80)
    logger.info("CHECKING JD EMBEDDINGS QUALITY")
    logger.info("=" * 80)
    logger.info("")
    
    repository = EmbeddingRepository(db)
    
    # Get all JD embeddings
    logger.info("Loading JD embeddings from database...")
    all_jds = repository.get_all_jd_embeddings()
    
    logger.info(f"Found {len(all_jds)} JD embeddings in database")
    
    if len(all_jds) == 0:
        logger.error("No JD embeddings found in database")
        return {}
    
    # Sample embeddings
    sample_size = min(sample_size, len(all_jds))
    logger.info(f"Sampling {sample_size} JD embeddings for quality check...")
    jd_sample = random.sample(all_jds, sample_size)
    
    # Extract embeddings
    embeddings = [j.embedding for j in jd_sample]
    
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
        'total_jds': len(all_jds),
        'sampled_jds': len(jd_sample),
        'zero_vectors': zero_vectors,
        'dimensions': dim_check,
        'normalization': norm_check,
        'similarity_distribution': similarity_check
    }
    
    return results


def print_quality_report(results: dict):
    """Print quality report."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("JD EMBEDDING QUALITY REPORT")
    logger.info("=" * 80)
    logger.info("")
    
    logger.info(f"Total JD Embeddings: {results.get('total_jds', 0)}")
    logger.info(f"Sampled for Check: {results.get('sampled_jds', 0)}")
    logger.info(f"Zero Vectors Found: {results.get('zero_vectors', 0)}")
    logger.info("")
    
    # Dimensions
    dim_check = results.get('dimensions', {})
    if dim_check:
        logger.info("EMBEDDING DIMENSIONS")
        logger.info("-" * 80)
        logger.info(f"Unique Dimensions: {dim_check.get('unique_dimensions', [])}")
        logger.info(f"Most Common Dimension: {dim_check.get('most_common_dimension', 'N/A')}")
        logger.info(f"All Same Dimension: {dim_check.get('all_same_dimension', False)}")
        logger.info("")
    
    # Normalization
    norm_check = results.get('normalization', {})
    if norm_check:
        logger.info("NORMALIZATION CHECK")
        logger.info("-" * 80)
        logger.info(f"Mean Norm: {norm_check.get('mean_norm', 0):.6f} (should be ~1.0)")
        logger.info(f"Std Norm: {norm_check.get('std_norm', 0):.6f} (should be ~0.0)")
        logger.info(f"Min Norm: {norm_check.get('min_norm', 0):.6f}")
        logger.info(f"Max Norm: {norm_check.get('max_norm', 0):.6f}")
        logger.info(f"All Normalized: {norm_check.get('all_normalized', False)}")
        logger.info(f"Normalized Count: {norm_check.get('normalized_count', 0)}/{norm_check.get('total_count', 0)}")
        
        # Quality assessment
        mean_norm = norm_check.get('mean_norm', 0)
        if 0.99 <= mean_norm <= 1.01:
            logger.info("✓ Normalization: EXCELLENT")
        elif 0.95 <= mean_norm <= 1.05:
            logger.info("⚠ Normalization: GOOD")
        else:
            logger.info("✗ Normalization: NEEDS IMPROVEMENT")
        logger.info("")
    
    # Similarity Distribution
    sim_check = results.get('similarity_distribution', {})
    if sim_check:
        logger.info("SIMILARITY DISTRIBUTION (Random JD Pairs)")
        logger.info("-" * 80)
        logger.info(f"Sample Size: {sim_check.get('sample_size', 0)}")
        logger.info(f"Mean Similarity: {sim_check.get('mean_similarity', 0):.4f}")
        logger.info(f"Std Similarity: {sim_check.get('std_similarity', 0):.4f}")
        logger.info(f"Min Similarity: {sim_check.get('min_similarity', 0):.4f}")
        logger.info(f"Max Similarity: {sim_check.get('max_similarity', 0):.4f}")
        logger.info(f"Median Similarity: {sim_check.get('median_similarity', 0):.4f}")
        logger.info(f"Q25 (25th percentile): {sim_check.get('q25_similarity', 0):.4f}")
        logger.info(f"Q75 (75th percentile): {sim_check.get('q75_similarity', 0):.4f}")
        logger.info(f"Q90 (90th percentile): {sim_check.get('q90_similarity', 0):.4f}")
        logger.info(f"Q95 (95th percentile): {sim_check.get('q95_similarity', 0):.4f}")
        logger.info("")
        
        # Quality assessment
        mean_sim = sim_check.get('mean_similarity', 0)
        std_sim = sim_check.get('std_similarity', 0)
        logger.info("QUALITY ASSESSMENT:")
        if mean_sim > 0.3 and std_sim > 0.1:
            logger.info("✓ Similarity Distribution: GOOD (good spread, meaningful differences)")
        elif mean_sim > 0.2:
            logger.info("⚠ Similarity Distribution: ACCEPTABLE")
        else:
            logger.info("✗ Similarity Distribution: LOW (embeddings may be too similar)")
        logger.info("")
    
    logger.info("=" * 80)
    logger.info("QUALITY CHECK COMPLETE")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Check JD embeddings quality from database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check with default sample size (2000 JDs, 5000 similarity pairs)
  python scripts/check_jd_embeddings_quality.py
  
  # Check with larger sample
  python scripts/check_jd_embeddings_quality.py --sample-size 5000 --similarity-sample 10000
        """
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        help="Number of JD embeddings to sample (default: 2000)"
    )
    
    parser.add_argument(
        "--similarity-sample",
        type=int,
        default=5000,
        help="Number of similarity pairs to check (default: 5000)"
    )
    
    args = parser.parse_args()
    
    db: Session = SessionLocal()
    try:
        results = check_jd_embeddings_quality(
            db=db,
            sample_size=args.sample_size,
            similarity_sample_size=args.similarity_sample
        )
        
        if results:
            print_quality_report(results)
        else:
            logger.error("No results to report")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Error checking quality: {e}", exc_info=True)
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()

