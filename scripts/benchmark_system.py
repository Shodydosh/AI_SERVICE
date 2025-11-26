"""Comprehensive system benchmark for embedding quality and performance."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import time
import random
import numpy as np
from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.embeddings.field_mapping_embedding import FieldMappingEmbeddingGenerator
from src.embeddings.improved_field_mapping_embedding import ImprovedFieldMappingEmbeddingGenerator
from src.embeddings.advanced_field_mapping_embedding import AdvancedFieldMappingEmbeddingGenerator
from src.data_processing.jd_processor import JDProcessor
from src.data_processing.candidate_processor import CandidateProcessor
from src.services.field_mapping_matching_service import FieldMappingMatchingService
from src.database.repository import EmbeddingRepository
from config.settings import settings
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystemBenchmark:
    """Comprehensive system benchmark."""
    
    def __init__(self):
        self.results = {
            'embedding_quality': {},
            'performance': {},
            'matching_quality': {},
            'field_extraction': {}
        }
    
    def benchmark_embedding_quality(
        self,
        generator,
        method_name: str,
        texts: List[str],
        sample_size: int = 1000
    ) -> Dict:
        """Benchmark embedding quality metrics."""
        logger.info(f"Benchmarking embedding quality for {method_name}...")
        
        # Generate embeddings
        embeddings = []
        start_time = time.time()
        
        for text in tqdm(texts[:sample_size], desc=f"  Generating embeddings ({method_name})"):
            try:
                # Use field embedding generation
                if isinstance(generator, (ImprovedFieldMappingEmbeddingGenerator, AdvancedFieldMappingEmbeddingGenerator)):
                    # For testing, create a simple field
                    field_emb = generator.generate_field_embedding(
                        text,
                        field_name='skills',
                        field_type='candidate'
                    )
                    embeddings.append(field_emb)
                else:
                    # Baseline method - doesn't support field_type parameter
                    field_emb = generator.generate_field_embedding(
                        text,
                        field_name='skills'
                    )
                    embeddings.append(field_emb)
            except Exception as e:
                logger.warning(f"Error generating embedding: {e}")
                continue
        
        generation_time = time.time() - start_time
        
        if not embeddings:
            logger.warning(f"No embeddings generated for {method_name}")
            return {}
        
        # Calculate quality metrics
        embeddings_array = np.array(embeddings)
        
        # Norm statistics
        norms = np.linalg.norm(embeddings_array, axis=1)
        norm_mean = float(np.mean(norms))
        norm_std = float(np.std(norms))
        norm_min = float(np.min(norms))
        norm_max = float(np.max(norms))
        
        # Similarity statistics (random pairs)
        similarities = []
        sample_pairs = min(1000, len(embeddings) * (len(embeddings) - 1) // 2)
        
        for _ in range(sample_pairs):
            idx1, idx2 = random.sample(range(len(embeddings)), 2)
            vec1 = embeddings_array[idx1]
            vec2 = embeddings_array[idx2]
            
            # Cosine similarity (already normalized)
            similarity = np.dot(vec1, vec2)
            similarities.append(float(similarity))
        
        sim_array = np.array(similarities)
        
        return {
            'method': method_name,
            'num_embeddings': len(embeddings),
            'generation_time': generation_time,
            'embeddings_per_second': len(embeddings) / generation_time if generation_time > 0 else 0,
            'norm_mean': norm_mean,
            'norm_std': norm_std,
            'norm_min': norm_min,
            'norm_max': norm_max,
            'similarity_mean': float(np.mean(sim_array)),
            'similarity_std': float(np.std(sim_array)),
            'similarity_min': float(np.min(sim_array)),
            'similarity_max': float(np.max(sim_array)),
            'similarity_median': float(np.median(sim_array)),
            'similarity_q25': float(np.percentile(sim_array, 25)),
            'similarity_q75': float(np.percentile(sim_array, 75))
        }
    
    def benchmark_field_extraction(
        self,
        candidate_data: pd.DataFrame,
        sample_size: int = 100
    ) -> Dict:
        """Benchmark field extraction quality."""
        logger.info("Benchmarking field extraction...")
        
        service = FieldMappingMatchingService(None)
        sample = candidate_data.sample(n=min(sample_size, len(candidate_data)), random_state=42)
        
        extraction_stats = {
            'total_samples': len(sample),
            'successful_extractions': 0,
            'fields_found': {'skills': 0, 'experience': 0, 'desired_job': 0},
            'fallback_used': 0,
            'empty_extractions': 0
        }
        
        for idx, row in tqdm(sample.iterrows(), total=len(sample), desc="  Extracting fields"):
            fields = service.extract_candidate_fields(row)
            
            if fields:
                extraction_stats['successful_extractions'] += 1
                for field_name in ['skills', 'experience', 'desired_job']:
                    if field_name in fields:
                        extraction_stats['fields_found'][field_name] += 1
            else:
                extraction_stats['empty_extractions'] += 1
        
        extraction_stats['success_rate'] = extraction_stats['successful_extractions'] / len(sample) if len(sample) > 0 else 0
        
        return extraction_stats
    
    def benchmark_matching_quality(
        self,
        db: Session,
        candidate_file: str,
        jd_file: str,
        sample_size: int = 50
    ) -> Dict:
        """Benchmark matching quality."""
        logger.info("Benchmarking matching quality...")
        
        # Load data
        candidate_processor = CandidateProcessor()
        candidate_processor.load_from_csv(candidate_file)
        candidate_data = candidate_processor.data
        
        jd_processor = JDProcessor()
        jd_processor.load_from_csv(jd_file)
        jd_data = jd_processor.data
        
        # Sample
        candidate_sample = candidate_data.sample(n=min(sample_size, len(candidate_data)), random_state=42)
        jd_sample = jd_data.sample(n=min(sample_size * 2, len(jd_data)), random_state=42)
        
        # Use advanced method
        service = FieldMappingMatchingService(db)
        service.embedding_generator = AdvancedFieldMappingEmbeddingGenerator(
            use_semantic_expansion=True,
            use_keyword_boost=True
        )
        
        matching_results = []
        
        for idx, candidate_row in tqdm(candidate_sample.iterrows(), total=len(candidate_sample), desc="  Matching candidates"):
            try:
                candidate_fields = service.extract_candidate_fields(candidate_row)
                if not candidate_fields:
                    continue
                
                # Find top jobs
                recommendations = service.find_top_jobs_for_candidate(
                    candidate_id=str(candidate_row.get('candidate_id', f'candidate_{idx}')),
                    candidate_fields=candidate_fields,
                    jd_data=jd_sample,
                    top_k=10
                )
                
                if recommendations:
                    top_similarity = recommendations[0].get('similarity_score', 0)
                    avg_similarity = np.mean([r.get('similarity_score', 0) for r in recommendations])
                    
                    matching_results.append({
                        'top_similarity': top_similarity,
                        'avg_similarity': avg_similarity,
                        'num_recommendations': len(recommendations)
                    })
            except Exception as e:
                logger.warning(f"Error matching candidate {idx}: {e}")
                continue
        
        if not matching_results:
            return {}
        
        return {
            'num_matches': len(matching_results),
            'top_similarity_mean': float(np.mean([r['top_similarity'] for r in matching_results])),
            'top_similarity_std': float(np.std([r['top_similarity'] for r in matching_results])),
            'avg_similarity_mean': float(np.mean([r['avg_similarity'] for r in matching_results])),
            'avg_similarity_std': float(np.std([r['avg_similarity'] for r in matching_results])),
            'recommendations_per_candidate': float(np.mean([r['num_recommendations'] for r in matching_results]))
        }
    
    def run_full_benchmark(
        self,
        candidate_file: str,
        jd_file: str,
        sample_size: int = 100
    ) -> Dict:
        """Run full system benchmark."""
        logger.info("=" * 80)
        logger.info("SYSTEM BENCHMARK")
        logger.info("=" * 80)
        logger.info("")
        
        # Load sample texts
        candidate_processor = CandidateProcessor()
        candidate_processor.load_from_csv(candidate_file)
        candidate_data = candidate_processor.data
        
        jd_processor = JDProcessor()
        jd_processor.load_from_csv(jd_file)
        jd_data = jd_processor.data
        
        # Prepare test texts
        test_texts = []
        for idx, row in candidate_data.head(sample_size).iterrows():
            skills = str(row.get('skills', '')).strip() if pd.notna(row.get('skills')) else ''
            if skills:
                test_texts.append(skills)
        
        if not test_texts:
            logger.warning("No test texts found, using sample texts")
            test_texts = ["Python, Machine Learning, Data Science", "Java, Spring Boot, Microservices"]
        
        logger.info(f"Using {len(test_texts)} test texts")
        logger.info("")
        
        # Benchmark different methods
        methods = {
            'baseline': FieldMappingEmbeddingGenerator(),
            'improved_weighted_concat': ImprovedFieldMappingEmbeddingGenerator(combination_method="weighted_concatenate"),
            'improved_attention': ImprovedFieldMappingEmbeddingGenerator(combination_method="attention_weighted"),
            'advanced': AdvancedFieldMappingEmbeddingGenerator(use_semantic_expansion=True, use_keyword_boost=True)
        }
        
        # 1. Embedding Quality
        logger.info("=" * 80)
        logger.info("1. EMBEDDING QUALITY BENCHMARK")
        logger.info("=" * 80)
        logger.info("")
        
        for method_name, generator in methods.items():
            quality = self.benchmark_embedding_quality(generator, method_name, test_texts, sample_size=min(50, len(test_texts)))
            if quality:
                self.results['embedding_quality'][method_name] = quality
                logger.info(f"{method_name}:")
                logger.info(f"  Norm: {quality['norm_mean']:.4f} ± {quality['norm_std']:.4f}")
                logger.info(f"  Similarity: {quality['similarity_mean']:.4f} ± {quality['similarity_std']:.4f}")
                logger.info(f"  Speed: {quality['embeddings_per_second']:.2f} emb/s")
                logger.info("")
        
        # 2. Field Extraction
        logger.info("=" * 80)
        logger.info("2. FIELD EXTRACTION BENCHMARK")
        logger.info("=" * 80)
        logger.info("")
        
        extraction_stats = self.benchmark_field_extraction(candidate_data, sample_size=min(100, len(candidate_data)))
        self.results['field_extraction'] = extraction_stats
        
        logger.info(f"Success Rate: {extraction_stats['success_rate']*100:.2f}%")
        logger.info(f"Fields Found:")
        for field, count in extraction_stats['fields_found'].items():
            logger.info(f"  {field}: {count}/{extraction_stats['total_samples']}")
        logger.info("")
        
        # 3. Matching Quality (if database available)
        logger.info("=" * 80)
        logger.info("3. MATCHING QUALITY BENCHMARK")
        logger.info("=" * 80)
        logger.info("")
        
        try:
            db = SessionLocal()
            matching_stats = self.benchmark_matching_quality(db, candidate_file, jd_file, sample_size=min(20, len(candidate_data)))
            if matching_stats:
                self.results['matching_quality'] = matching_stats
                logger.info(f"Top Similarity: {matching_stats['top_similarity_mean']:.4f} ± {matching_stats['top_similarity_std']:.4f}")
                logger.info(f"Avg Similarity: {matching_stats['avg_similarity_mean']:.4f} ± {matching_stats['avg_similarity_std']:.4f}")
                logger.info(f"Recommendations per Candidate: {matching_stats['recommendations_per_candidate']:.2f}")
            db.close()
        except Exception as e:
            logger.warning(f"Could not benchmark matching quality: {e}")
        
        logger.info("")
        
        # Summary
        logger.info("=" * 80)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 80)
        logger.info("")
        
        # Best method by similarity (lower is better for differentiation)
        if self.results['embedding_quality']:
            best_method = min(
                self.results['embedding_quality'].items(),
                key=lambda x: x[1]['similarity_mean']
            )
            logger.info(f"Best Differentiation: {best_method[0]} (similarity: {best_method[1]['similarity_mean']:.4f})")
        
        # Fastest method
        if self.results['embedding_quality']:
            fastest_method = max(
                self.results['embedding_quality'].items(),
                key=lambda x: x[1]['embeddings_per_second']
            )
            logger.info(f"Fastest Method: {fastest_method[0]} ({fastest_method[1]['embeddings_per_second']:.2f} emb/s)")
        
        logger.info("")
        
        return self.results


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive system benchmark",
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
        help="Sample size for benchmarking (default: 100)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results (optional)"
    )
    
    args = parser.parse_args()
    
    try:
        benchmark = SystemBenchmark()
        results = benchmark.run_full_benchmark(
            candidate_file=args.candidate_file,
            jd_file=args.jd_file,
            sample_size=args.sample_size
        )
        
        # Save results if output specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {args.output}")
        
        sys.exit(0)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

