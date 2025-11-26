"""Script to check embedding quality with random large samples."""
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
from src.embeddings.field_mapping_embedding import FieldMappingEmbeddingGenerator
from src.data_processing.candidate_processor import CandidateProcessor
from src.data_processing.jd_processor import JDProcessor
from tqdm import tqdm
import pandas as pd
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_embedding_normality(embeddings: List[List[float]]) -> Dict:
    """Check if embeddings are properly normalized."""
    norms = []
    for emb in embeddings:
        norm = np.linalg.norm(emb)
        norms.append(norm)
    
    norms_array = np.array(norms)
    
    return {
        'mean_norm': float(np.mean(norms_array)),
        'std_norm': float(np.std(norms_array)),
        'min_norm': float(np.min(norms_array)),
        'max_norm': float(np.max(norms_array)),
        'all_normalized': np.allclose(norms_array, 1.0, atol=1e-5),
        'normalized_count': np.sum(np.isclose(norms_array, 1.0, atol=1e-5)),
        'total_count': len(embeddings)
    }


def check_embedding_similarity_distribution(
    candidate_embeddings: List[List[float]],
    jd_embeddings: List[List[float]],
    sample_size: int = 1000
) -> Dict:
    """Check similarity distribution between random candidate-JD pairs."""
    if len(candidate_embeddings) == 0 or len(jd_embeddings) == 0:
        return {}
    
    # Sample random pairs
    sample_size = min(sample_size, len(candidate_embeddings) * len(jd_embeddings))
    similarities = []
    
    for _ in range(sample_size):
        cand_idx = random.randint(0, len(candidate_embeddings) - 1)
        jd_idx = random.randint(0, len(jd_embeddings) - 1)
        
        cand_vec = np.array(candidate_embeddings[cand_idx])
        jd_vec = np.array(jd_embeddings[jd_idx])
        
        # Calculate cosine similarity
        similarity = np.dot(cand_vec, jd_vec) / (
            np.linalg.norm(cand_vec) * np.linalg.norm(jd_vec)
        )
        similarities.append(float(similarity))
    
    similarities_array = np.array(similarities)
    
    return {
        'mean_similarity': float(np.mean(similarities_array)),
        'std_similarity': float(np.std(similarities_array)),
        'min_similarity': float(np.min(similarities_array)),
        'max_similarity': float(np.max(similarities_array)),
        'median_similarity': float(np.median(similarities_array)),
        'q25_similarity': float(np.percentile(similarities_array, 25)),
        'q75_similarity': float(np.percentile(similarities_array, 75)),
        'sample_size': sample_size
    }


def check_field_embedding_quality(
    db: Session,
    candidate_file: str,
    jd_file: str,
    sample_size: int = 100
) -> Dict:
    """Check quality of field-by-field embeddings."""
    logger.info("=" * 80)
    logger.info("CHECKING FIELD EMBEDDING QUALITY")
    logger.info("=" * 80)
    
    # Load data
    candidate_processor = CandidateProcessor()
    candidate_processor.load_from_csv(candidate_file)
    candidate_data = candidate_processor.data
    
    jd_processor = JDProcessor()
    jd_processor.load_from_csv(jd_file)
    jd_data = jd_processor.data
    
    # Create candidate_id if missing
    if 'candidate_id' not in candidate_data.columns:
        candidate_data['candidate_id'] = candidate_data.index.astype(str).map(lambda x: f'candidate_{x}')
    
    # Sample random candidates and JDs
    sample_size = min(sample_size, len(candidate_data), len(jd_data))
    candidate_sample = candidate_data.sample(n=sample_size, random_state=42)
    jd_sample = jd_data.sample(n=min(sample_size * 2, len(jd_data)), random_state=42)
    
    logger.info(f"Sampling {len(candidate_sample)} candidates and {len(jd_sample)} JDs")
    
    # Initialize embedding generator
    embedding_generator = FieldMappingEmbeddingGenerator()
    
    # Generate field embeddings for samples
    logger.info("Generating field embeddings for samples...")
    candidate_field_embeddings_list = []
    jd_field_embeddings_list = []
    
    for idx, row in tqdm(candidate_sample.iterrows(), total=len(candidate_sample), desc="Candidates"):
        candidate_fields = {
            'skills': str(row.get('skills', '')).strip() if pd.notna(row.get('skills')) else '',
            'experience': str(row.get('experience', '')).strip() if pd.notna(row.get('experience')) else '',
            'desired_job': str(row.get('desired_job_translated', '')).strip() if pd.notna(row.get('desired_job_translated')) else '',
        }
        if not candidate_fields['experience']:
            candidate_fields['experience'] = str(row.get('work_experience', '')).strip() if pd.notna(row.get('work_experience')) else ''
        if not candidate_fields['desired_job']:
            candidate_fields['desired_job'] = str(row.get('desired_job', '')).strip() if pd.notna(row.get('desired_job')) else ''
        
        cand_embeddings = embedding_generator.generate_candidate_field_embeddings(candidate_fields)
        candidate_field_embeddings_list.append(cand_embeddings)
    
    for idx, row in tqdm(jd_sample.iterrows(), total=len(jd_sample), desc="JDs"):
        jd_fields = {
            'title': str(row.get('title', '')).strip() if pd.notna(row.get('title')) else '',
            'requirements': str(row.get('requirements', '')).strip() if pd.notna(row.get('requirements')) else '',
        }
        jd_embeddings = embedding_generator.generate_jd_field_embeddings(jd_fields)
        jd_field_embeddings_list.append(jd_embeddings)
    
    # Check quality metrics
    logger.info("")
    logger.info("Calculating quality metrics...")
    
    results = {
        'candidate_samples': len(candidate_sample),
        'jd_samples': len(jd_sample),
        'field_quality': {}
    }
    
    # Check each field mapping
    field_mappings = embedding_generator.FIELD_MAPPINGS
    
    for candidate_field, jd_field in field_mappings.items():
        logger.info(f"Checking {candidate_field} -> {jd_field} mapping...")
        
        # Collect embeddings for this field
        cand_field_embs = []
        jd_field_embs = []
        
        for cand_embs in candidate_field_embeddings_list:
            if candidate_field in cand_embs:
                cand_field_embs.append(cand_embs[candidate_field])
        
        for jd_embs in jd_field_embeddings_list:
            if jd_field in jd_embs:
                jd_field_embs.append(jd_embs[jd_field])
        
        if len(cand_field_embs) == 0 or len(jd_field_embs) == 0:
            logger.warning(f"Insufficient embeddings for {candidate_field} -> {jd_field}")
            continue
        
        # Check normalization
        cand_norms = check_embedding_normality(cand_field_embs)
        jd_norms = check_embedding_normality(jd_field_embs)
        
        # Check similarity distribution
        similarity_dist = check_embedding_similarity_distribution(
            cand_field_embs,
            jd_field_embs,
            sample_size=min(500, len(cand_field_embs) * len(jd_field_embs))
        )
        
        results['field_quality'][candidate_field] = {
            'candidate_embeddings_count': len(cand_field_embs),
            'jd_embeddings_count': len(jd_field_embs),
            'candidate_norms': cand_norms,
            'jd_norms': jd_norms,
            'similarity_distribution': similarity_dist
        }
    
    return results


def check_database_embeddings_quality(
    db: Session,
    sample_size: int = 1000
) -> Dict:
    """Check quality of embeddings stored in database."""
    logger.info("=" * 80)
    logger.info("CHECKING DATABASE EMBEDDINGS QUALITY")
    logger.info("=" * 80)
    
    repository = EmbeddingRepository(db)
    
    # Get all embeddings
    logger.info("Loading embeddings from database...")
    all_candidates = repository.get_all_candidate_embeddings()
    all_jds = repository.get_all_jd_embeddings()
    
    logger.info(f"Found {len(all_candidates)} candidate embeddings")
    logger.info(f"Found {len(all_jds)} JD embeddings")
    
    if len(all_candidates) == 0 or len(all_jds) == 0:
        logger.error("No embeddings found in database")
        return {}
    
    # Sample embeddings
    sample_size = min(sample_size, len(all_candidates), len(all_jds))
    candidate_sample = random.sample(all_candidates, min(sample_size, len(all_candidates)))
    jd_sample = random.sample(all_jds, min(sample_size * 2, len(all_jds)))
    
    logger.info(f"Sampling {len(candidate_sample)} candidates and {len(jd_sample)} JDs")
    
    # Extract embeddings
    candidate_embeddings = [c.embedding for c in candidate_sample]
    jd_embeddings = [j.embedding for j in jd_sample]
    
    # Check normalization
    logger.info("Checking normalization...")
    cand_norms = check_embedding_normality(candidate_embeddings)
    jd_norms = check_embedding_normality(jd_embeddings)
    
    # Check similarity distribution
    logger.info("Checking similarity distribution...")
    similarity_dist = check_embedding_similarity_distribution(
        candidate_embeddings,
        jd_embeddings,
        sample_size=min(1000, len(candidate_embeddings) * len(jd_embeddings))
    )
    
    results = {
        'total_candidates': len(all_candidates),
        'total_jds': len(all_jds),
        'sampled_candidates': len(candidate_sample),
        'sampled_jds': len(jd_sample),
        'candidate_norms': cand_norms,
        'jd_norms': jd_norms,
        'similarity_distribution': similarity_dist
    }
    
    return results


def print_quality_report(results: Dict, field_results: Dict = None):
    """Print quality report."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("EMBEDDING QUALITY REPORT")
    logger.info("=" * 80)
    logger.info("")
    
    # Database embeddings quality
    if results:
        logger.info("DATABASE EMBEDDINGS QUALITY")
        logger.info("-" * 80)
        logger.info(f"Total Candidates: {results.get('total_candidates', 0)}")
        logger.info(f"Total JDs: {results.get('total_jds', 0)}")
        logger.info(f"Sampled Candidates: {results.get('sampled_candidates', 0)}")
        logger.info(f"Sampled JDs: {results.get('sampled_jds', 0)}")
        logger.info("")
        
        # Candidate normalization
        cand_norms = results.get('candidate_norms', {})
        if cand_norms:
            logger.info("Candidate Embeddings Normalization:")
            logger.info(f"  Mean norm: {cand_norms.get('mean_norm', 0):.6f}")
            logger.info(f"  Std norm: {cand_norms.get('std_norm', 0):.6f}")
            logger.info(f"  Min norm: {cand_norms.get('min_norm', 0):.6f}")
            logger.info(f"  Max norm: {cand_norms.get('max_norm', 0):.6f}")
            logger.info(f"  All normalized: {cand_norms.get('all_normalized', False)}")
            logger.info(f"  Normalized count: {cand_norms.get('normalized_count', 0)}/{cand_norms.get('total_count', 0)}")
            logger.info("")
        
        # JD normalization
        jd_norms = results.get('jd_norms', {})
        if jd_norms:
            logger.info("JD Embeddings Normalization:")
            logger.info(f"  Mean norm: {jd_norms.get('mean_norm', 0):.6f}")
            logger.info(f"  Std norm: {jd_norms.get('std_norm', 0):.6f}")
            logger.info(f"  Min norm: {jd_norms.get('min_norm', 0):.6f}")
            logger.info(f"  Max norm: {jd_norms.get('max_norm', 0):.6f}")
            logger.info(f"  All normalized: {jd_norms.get('all_normalized', False)}")
            logger.info(f"  Normalized count: {jd_norms.get('normalized_count', 0)}/{jd_norms.get('total_count', 0)}")
            logger.info("")
        
        # Similarity distribution
        sim_dist = results.get('similarity_distribution', {})
        if sim_dist:
            logger.info("Similarity Distribution (Random Pairs):")
            logger.info(f"  Sample size: {sim_dist.get('sample_size', 0)}")
            logger.info(f"  Mean similarity: {sim_dist.get('mean_similarity', 0):.4f}")
            logger.info(f"  Std similarity: {sim_dist.get('std_similarity', 0):.4f}")
            logger.info(f"  Min similarity: {sim_dist.get('min_similarity', 0):.4f}")
            logger.info(f"  Max similarity: {sim_dist.get('max_similarity', 0):.4f}")
            logger.info(f"  Median similarity: {sim_dist.get('median_similarity', 0):.4f}")
            logger.info(f"  Q25 similarity: {sim_dist.get('q25_similarity', 0):.4f}")
            logger.info(f"  Q75 similarity: {sim_dist.get('q75_similarity', 0):.4f}")
            logger.info("")
    
    # Field-by-field quality
    if field_results and field_results.get('field_quality'):
        logger.info("FIELD-BY-FIELD EMBEDDING QUALITY")
        logger.info("-" * 80)
        
        for field, quality in field_results['field_quality'].items():
            logger.info(f"\n{field.upper()} Field:")
            logger.info(f"  Candidate embeddings: {quality.get('candidate_embeddings_count', 0)}")
            logger.info(f"  JD embeddings: {quality.get('jd_embeddings_count', 0)}")
            
            cand_norms = quality.get('candidate_norms', {})
            if cand_norms:
                logger.info(f"  Candidate norm - Mean: {cand_norms.get('mean_norm', 0):.6f}, All normalized: {cand_norms.get('all_normalized', False)}")
            
            jd_norms = quality.get('jd_norms', {})
            if jd_norms:
                logger.info(f"  JD norm - Mean: {jd_norms.get('mean_norm', 0):.6f}, All normalized: {jd_norms.get('all_normalized', False)}")
            
            sim_dist = quality.get('similarity_distribution', {})
            if sim_dist:
                logger.info(f"  Similarity - Mean: {sim_dist.get('mean_similarity', 0):.4f}, Median: {sim_dist.get('median_similarity', 0):.4f}")
        
        logger.info("")
    
    logger.info("=" * 80)
    logger.info("QUALITY CHECK COMPLETE")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Check embedding quality with random large samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check database embeddings quality
  python scripts/check_embedding_quality_batch.py --check-database --sample-size 1000
  
  # Check field-by-field quality
  python scripts/check_embedding_quality_batch.py \\
      --candidate-file data/processed/candidate_processed.csv \\
      --jd-file data/processed/jd_processed.csv \\
      --sample-size 200
  
  # Check both
  python scripts/check_embedding_quality_batch.py \\
      --check-database \\
      --candidate-file data/processed/candidate_processed.csv \\
      --jd-file data/processed/jd_processed.csv \\
      --sample-size 500
        """
    )
    
    parser.add_argument(
        "--check-database",
        action="store_true",
        help="Check embeddings stored in database"
    )
    
    parser.add_argument(
        "--candidate-file",
        type=str,
        help="Path to candidate CSV file (for field-by-field check)"
    )
    
    parser.add_argument(
        "--jd-file",
        type=str,
        help="Path to JD CSV file (for field-by-field check)"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of samples to check (default: 1000)"
    )
    
    args = parser.parse_args()
    
    if not args.check_database and (not args.candidate_file or not args.jd_file):
        parser.error("Either --check-database or both --candidate-file and --jd-file must be provided")
    
    db: Session = SessionLocal()
    try:
        results = {}
        field_results = None
        
        # Check database embeddings
        if args.check_database:
            results = check_database_embeddings_quality(db, sample_size=args.sample_size)
        
        # Check field-by-field quality
        if args.candidate_file and args.jd_file:
            field_results = check_field_embedding_quality(
                db=db,
                candidate_file=args.candidate_file,
                jd_file=args.jd_file,
                sample_size=min(args.sample_size, 200)  # Limit for field check
            )
        
        # Print report
        print_quality_report(results, field_results)
        
    except Exception as e:
        logger.error(f"Error checking quality: {e}", exc_info=True)
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()

