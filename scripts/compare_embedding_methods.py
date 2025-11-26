"""Script to compare different embedding combination methods and find the best one."""
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
from src.embeddings.field_mapping_embedding import FieldMappingEmbeddingGenerator
from src.embeddings.improved_field_mapping_embedding import ImprovedFieldMappingEmbeddingGenerator
from src.data_processing.jd_processor import JDProcessor
from src.data_processing.candidate_processor import CandidateProcessor
from config.settings import settings
from tqdm import tqdm
import pandas as pd
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_similarity_stats(embeddings1: List[List[float]], embeddings2: List[List[float]], sample_size: int = 1000) -> Dict:
    """Calculate similarity statistics between two sets of embeddings."""
    if len(embeddings1) == 0 or len(embeddings2) == 0:
        return {}
    
    # Filter out zero embeddings
    valid_emb1 = []
    valid_emb2 = []
    
    for emb in embeddings1:
        emb_array = np.array(emb)
        norm = np.linalg.norm(emb_array)
        if norm > 1e-6:  # Not zero
            valid_emb1.append(emb)
    
    for emb in embeddings2:
        emb_array = np.array(emb)
        norm = np.linalg.norm(emb_array)
        if norm > 1e-6:  # Not zero
            valid_emb2.append(emb)
    
    if len(valid_emb1) == 0 or len(valid_emb2) == 0:
        logger.warning(f"No valid embeddings found: emb1={len(valid_emb1)}, emb2={len(valid_emb2)}")
        return {}
    
    sample_size = min(sample_size, len(valid_emb1) * len(valid_emb2))
    similarities = []
    
    for _ in range(sample_size):
        idx1 = random.randint(0, len(valid_emb1) - 1)
        idx2 = random.randint(0, len(valid_emb2) - 1)
        
        vec1 = np.array(valid_emb1[idx1])
        vec2 = np.array(valid_emb2[idx2])
        
        # Normalize
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 > 1e-6 and norm2 > 1e-6:
            # Cosine similarity (already normalized, so just dot product)
            similarity = np.dot(vec1, vec2)
            similarities.append(float(similarity))
    
    if not similarities:
        logger.warning("No valid similarities calculated")
        return {}
    
    sim_array = np.array(similarities)
    
    return {
        'mean': float(np.mean(sim_array)),
        'std': float(np.std(sim_array)),
        'min': float(np.min(sim_array)),
        'max': float(np.max(sim_array)),
        'median': float(np.median(sim_array)),
        'q25': float(np.percentile(sim_array, 25)),
        'q75': float(np.percentile(sim_array, 75)),
        'sample_size': len(similarities),
        'valid_emb1': len(valid_emb1),
        'valid_emb2': len(valid_emb2)
    }


def compare_methods(
    candidate_file: str,
    jd_file: str,
    sample_size: int = 100
):
    """Compare different embedding combination methods."""
    logger.info("=" * 80)
    logger.info("COMPARING EMBEDDING METHODS")
    logger.info("=" * 80)
    logger.info("")
    
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
    
    # Sample
    sample_size = min(sample_size, len(candidate_data), len(jd_data))
    candidate_sample = candidate_data.sample(n=sample_size, random_state=42)
    jd_sample = jd_data.sample(n=min(sample_size * 2, len(jd_data)), random_state=42)
    
    logger.info(f"Sampling {len(candidate_sample)} candidates and {len(jd_sample)} JDs")
    logger.info("")
    
    # Methods to compare
    from src.embeddings.advanced_field_mapping_embedding import AdvancedFieldMappingEmbeddingGenerator
    
    methods = {
        'baseline': FieldMappingEmbeddingGenerator(),
        'improved_weighted_avg': ImprovedFieldMappingEmbeddingGenerator(combination_method="weighted_average"),
        'improved_weighted_concat': ImprovedFieldMappingEmbeddingGenerator(combination_method="weighted_concatenate"),
        'improved_attention': ImprovedFieldMappingEmbeddingGenerator(combination_method="attention_weighted"),
        'advanced': AdvancedFieldMappingEmbeddingGenerator(use_semantic_expansion=True, use_keyword_boost=True)
    }
    
    results = {}
    
    for method_name, generator in methods.items():
        logger.info(f"Testing method: {method_name}")
        logger.info("-" * 80)
        
        # Generate embeddings
        candidate_embeddings_list = []
        jd_embeddings_list = []
        
        # Use FieldMappingMatchingService to extract fields properly
        from src.services.field_mapping_matching_service import FieldMappingMatchingService
        model_name = generator.model_name if hasattr(generator, 'model_name') else None
        if not model_name:
            model_name = settings.EMBEDDING_MODEL
        temp_service = FieldMappingMatchingService(None, model_name=model_name)
        
        for idx, row in tqdm(candidate_sample.iterrows(), total=len(candidate_sample), desc=f"  Candidates ({method_name})"):
            # Use proper field extraction
            candidate_fields = temp_service.extract_candidate_fields(row)
            
            # Skip if all fields are empty
            if not candidate_fields or not any(candidate_fields.values()):
                # Use a default embedding (zero vector will be filtered out)
                candidate_embeddings_list.append([0.0] * generator.get_embedding_dimension())
                continue
            
            # Generate field embeddings
            from src.embeddings.advanced_field_mapping_embedding import AdvancedFieldMappingEmbeddingGenerator
            
            if isinstance(generator, AdvancedFieldMappingEmbeddingGenerator):
                field_embs = generator.generate_candidate_field_embeddings(candidate_fields)
                weights = {'skills': 0.48, 'experience': 0.35, 'desired_job': 0.17}
                content_lengths = {k: len(v) for k, v in candidate_fields.items() if v}
                combined = generator.combine_field_embeddings_advanced(field_embs, weights, content_lengths)
            elif isinstance(generator, ImprovedFieldMappingEmbeddingGenerator):
                field_embs = generator.generate_candidate_field_embeddings(candidate_fields)
                # Combine using the method
                weights = {'skills': 0.45, 'experience': 0.35, 'desired_job': 0.20}
                content_lengths = {k: len(v) for k, v in candidate_fields.items() if v}
                combined = generator.combine_field_embeddings(field_embs, weights, content_lengths)
            else:
                field_embs = generator.generate_candidate_field_embeddings(candidate_fields)
                # Use weighted average for baseline
                import numpy as np
                combined = None
                total_weight = 0.0
                weights = {'skills': 0.4, 'experience': 0.35, 'desired_job': 0.25}
                for field, emb in field_embs.items():
                    if field in weights:
                        emb_array = np.array(emb) * weights[field]
                        if combined is None:
                            combined = emb_array
                        else:
                            combined += emb_array
                        total_weight += weights[field]
                if combined is not None and total_weight > 0:
                    combined = combined / total_weight
                    norm = np.linalg.norm(combined)
                    if norm > 0:
                        combined = combined / norm
                    combined = combined.tolist()
                else:
                    combined = [0.0] * generator.get_embedding_dimension()
            
            # Ensure it's a list
            if not isinstance(combined, list):
                combined = combined.tolist() if hasattr(combined, 'tolist') else list(combined)
            
            candidate_embeddings_list.append(combined)
        
        for idx, row in tqdm(jd_sample.iterrows(), total=len(jd_sample), desc=f"  JDs ({method_name})"):
            jd_fields = {
                'title': str(row.get('title', '')).strip() if pd.notna(row.get('title')) else '',
                'requirements': str(row.get('requirements', '')).strip() if pd.notna(row.get('requirements')) else '',
            }
            
            # Skip if all fields are empty
            if not any(jd_fields.values()):
                jd_embeddings_list.append([0.0] * generator.get_embedding_dimension())
                continue
            
            from src.embeddings.advanced_field_mapping_embedding import AdvancedFieldMappingEmbeddingGenerator
            
            if isinstance(generator, AdvancedFieldMappingEmbeddingGenerator):
                field_embs = generator.generate_jd_field_embeddings(jd_fields)
                weights = {'title': 0.25, 'requirements': 0.50, 'description': 0.25}
                content_lengths = {k: len(v) for k, v in jd_fields.items() if v}
                combined = generator.combine_field_embeddings_advanced(field_embs, weights, content_lengths)
            elif isinstance(generator, ImprovedFieldMappingEmbeddingGenerator):
                field_embs = generator.generate_jd_field_embeddings(jd_fields)
                weights = {'title': 0.25, 'requirements': 0.45, 'description': 0.30}
                content_lengths = {k: len(v) for k, v in jd_fields.items() if v}
                combined = generator.combine_field_embeddings(field_embs, weights, content_lengths)
            else:
                field_embs = generator.generate_jd_field_embeddings(jd_fields)
                import numpy as np
                combined = None
                total_weight = 0.0
                weights = {'title': 0.25, 'requirements': 0.45, 'description': 0.30}
                for field, emb in field_embs.items():
                    if field in weights:
                        emb_array = np.array(emb) * weights[field]
                        if combined is None:
                            combined = emb_array
                        else:
                            combined += emb_array
                        total_weight += weights[field]
                if combined is not None and total_weight > 0:
                    combined = combined / total_weight
                    norm = np.linalg.norm(combined)
                    if norm > 0:
                        combined = combined / norm
                    combined = combined.tolist()
                else:
                    combined = [0.0] * generator.get_embedding_dimension()
            
            # Ensure it's a list
            if not isinstance(combined, list):
                combined = combined.tolist() if hasattr(combined, 'tolist') else list(combined)
            
            jd_embeddings_list.append(combined)
        
        # Calculate statistics
        cand_stats = calculate_similarity_stats(candidate_embeddings_list, candidate_embeddings_list, sample_size=500)
        jd_stats = calculate_similarity_stats(jd_embeddings_list, jd_embeddings_list, sample_size=500)
        cross_stats = calculate_similarity_stats(candidate_embeddings_list, jd_embeddings_list, sample_size=1000)
        
        results[method_name] = {
            'candidate_similarity': cand_stats,
            'jd_similarity': jd_stats,
            'cross_similarity': cross_stats
        }
        
        logger.info(f"  Candidate-Candidate Mean Similarity: {cand_stats.get('mean', 0):.4f} (valid: {cand_stats.get('valid_emb1', 0)})")
        logger.info(f"  JD-JD Mean Similarity: {jd_stats.get('mean', 0):.4f} (valid: {jd_stats.get('valid_emb1', 0)})")
        logger.info(f"  Candidate-JD Mean Similarity: {cross_stats.get('mean', 0):.4f} (valid: {cross_stats.get('valid_emb1', 0)}/{cross_stats.get('valid_emb2', 0)})")
        logger.info("")
    
    # Print comparison
    logger.info("=" * 80)
    logger.info("METHOD COMPARISON SUMMARY")
    logger.info("=" * 80)
    logger.info("")
    
    logger.info(f"{'Method':<30} {'Cand-Cand':<12} {'JD-JD':<12} {'Cand-JD':<12} {'Std':<10}")
    logger.info("-" * 80)
    
    for method_name, stats in results.items():
        cand_mean = stats['candidate_similarity'].get('mean', 0)
        jd_mean = stats['jd_similarity'].get('mean', 0)
        cross_mean = stats['cross_similarity'].get('mean', 0)
        cross_std = stats['cross_similarity'].get('std', 0)
        
        logger.info(f"{method_name:<30} {cand_mean:>10.4f}   {jd_mean:>10.4f}   {cross_mean:>10.4f}   {cross_std:>8.4f}")
    
    logger.info("")
    logger.info("Best method selection criteria:")
    logger.info("  - Lower candidate-candidate similarity = better differentiation")
    logger.info("  - Lower JD-JD similarity = better differentiation")
    logger.info("  - Higher candidate-JD similarity = better matching potential")
    logger.info("  - Higher std = better spread/distribution")
    logger.info("")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare different embedding combination methods",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--candidate-file",
        type=str,
        required=True,
        help="Path to candidate CSV file"
    )
    
    parser.add_argument(
        "--jd-file",
        type=str,
        required=True,
        help="Path to JD CSV file"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of samples to test (default: 100)"
    )
    
    args = parser.parse_args()
    
    try:
        compare_methods(
            candidate_file=args.candidate_file,
            jd_file=args.jd_file,
            sample_size=args.sample_size
        )
        sys.exit(0)
    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

