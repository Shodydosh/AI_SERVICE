"""Comprehensive benchmark script for 10 model variations."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import json
import logging
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from src.embeddings.model_variations import get_variation, list_all_variations
from src.embeddings.parameter_variations import (
    generate_all_variations, 
    get_variation_by_id as get_param_variation,
    list_all_variations as list_param_variations
)
from src.database.connection import get_db_session
from src.database.models import JobDescriptionEmbedding, CandidateEmbedding
from src.vector_search.faiss_manager import FAISSManager
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelVariationBenchmark:
    """Benchmark system for model variations."""
    
    def __init__(self, test_sample_size: int = 100):
        self.test_sample_size = test_sample_size
        self.results = []
        self.reports_dir = Path("reports/benchmark_variations")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def load_test_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load sample JD and candidate data for testing."""
        logger.info("Loading test data from database...")
        
        with get_db_session() as session:
            # Get sample JDs
            jds = session.query(JobDescriptionEmbedding).limit(self.test_sample_size).all()
            jd_data = []
            for jd in jds:
                jd_data.append({
                    'id': jd.job_id,
                    'title': getattr(jd, 'title', '') or '',
                    'description': getattr(jd, 'description', '') or '',
                    'requirements': getattr(jd, 'requirements', '') or '',
                    'skills': getattr(jd, 'skills', '') or ''
                })
            
            # Get sample candidates
            candidates = session.query(CandidateEmbedding).limit(self.test_sample_size).all()
            candidate_data = []
            for cand in candidates:
                candidate_data.append({
                    'id': cand.candidate_id,
                    'skills': getattr(cand, 'skills', '') or '',
                    'experience': getattr(cand, 'experience', '') or '',
                    'education': getattr(cand, 'education', '') or '',
                    'summary': getattr(cand, 'summary', '') or ''
                })
        
        logger.info(f"Loaded {len(jd_data)} JDs and {len(candidate_data)} candidates")
        return jd_data, candidate_data
    
    def benchmark_embedding_generation(self, variation, jd_data: List[Dict], 
                                       candidate_data: List[Dict]) -> Dict:
        """Benchmark embedding generation speed and quality."""
        logger.info(f"Benchmarking embedding generation for {variation.name}...")
        
        metrics = {
            'variation_id': variation.variation_id,
            'variation_name': variation.name,
            'model_name': variation.model_name,
            'dimension': variation.dimension
        }
        
        # Test JD embedding generation
        jd_times = []
        jd_embeddings = []
        
        for jd in jd_data[:10]:  # Test with 10 samples
            start_time = time.time()
            embedding = variation.generate_jd_embedding(
                title=jd.get('title', ''),
                description=jd.get('description', ''),
                requirements=jd.get('requirements', ''),
                skills=jd.get('skills', '')
            )
            elapsed = time.time() - start_time
            jd_times.append(elapsed)
            jd_embeddings.append(embedding)
        
        # Test batch JD embedding generation
        batch_start = time.time()
        batch_jd_texts = [
            f"Title: {jd.get('title', '')} Skills: {jd.get('skills', '')} "
            f"Requirements: {jd.get('requirements', '')} Description: {jd.get('description', '')}"
            for jd in jd_data[:50]
        ]
        batch_jd_embeddings = variation.generate_embeddings_batch(batch_jd_texts)
        batch_jd_time = time.time() - batch_start
        
        # Test candidate embedding generation
        candidate_times = []
        candidate_embeddings = []
        
        for cand in candidate_data[:10]:  # Test with 10 samples
            start_time = time.time()
            embedding = variation.generate_candidate_embedding(
                skills=cand.get('skills', ''),
                experience=cand.get('experience', ''),
                education=cand.get('education', ''),
                summary=cand.get('summary', '')
            )
            elapsed = time.time() - start_time
            candidate_times.append(elapsed)
            candidate_embeddings.append(embedding)
        
        # Test batch candidate embedding generation
        batch_start = time.time()
        batch_candidate_texts = [
            f"Skills: {cand.get('skills', '')} Experience: {cand.get('experience', '')} "
            f"Summary: {cand.get('summary', '')} Education: {cand.get('education', '')}"
            for cand in candidate_data[:50]
        ]
        batch_candidate_embeddings = variation.generate_embeddings_batch(batch_candidate_texts)
        batch_candidate_time = time.time() - batch_start
        
        # Calculate metrics
        metrics['jd_single_avg_time'] = np.mean(jd_times)
        metrics['jd_single_min_time'] = np.min(jd_times)
        metrics['jd_single_max_time'] = np.max(jd_times)
        metrics['jd_batch_time'] = batch_jd_time
        metrics['jd_batch_throughput'] = len(batch_jd_texts) / batch_jd_time if batch_jd_time > 0 else 0
        
        metrics['candidate_single_avg_time'] = np.mean(candidate_times)
        metrics['candidate_single_min_time'] = np.min(candidate_times)
        metrics['candidate_single_max_time'] = np.max(candidate_times)
        metrics['candidate_batch_time'] = batch_candidate_time
        metrics['candidate_batch_throughput'] = len(batch_candidate_texts) / batch_candidate_time if batch_candidate_time > 0 else 0
        
        # Calculate embedding quality metrics (variance, norm)
        all_embeddings = jd_embeddings + candidate_embeddings
        if all_embeddings:
            embedding_matrix = np.array(all_embeddings)
            metrics['embedding_mean_norm'] = float(np.mean(np.linalg.norm(embedding_matrix, axis=1)))
            metrics['embedding_std_norm'] = float(np.std(np.linalg.norm(embedding_matrix, axis=1)))
            metrics['embedding_mean_variance'] = float(np.mean(np.var(embedding_matrix, axis=0)))
        
        return metrics
    
    def benchmark_similarity_quality(self, variation, jd_data: List[Dict], 
                                     candidate_data: List[Dict]) -> Dict:
        """Benchmark similarity quality using cosine similarity."""
        logger.info(f"Benchmarking similarity quality for {variation.name}...")
        
        metrics = {}
        
        # Generate embeddings for all test data
        jd_embeddings = []
        for jd in jd_data[:20]:  # Use 20 samples for similarity test
            emb = variation.generate_jd_embedding(
                title=jd.get('title', ''),
                description=jd.get('description', ''),
                requirements=jd.get('requirements', ''),
                skills=jd.get('skills', '')
            )
            jd_embeddings.append(emb)
        
        candidate_embeddings = []
        for cand in candidate_data[:20]:
            emb = variation.generate_candidate_embedding(
                skills=cand.get('skills', ''),
                experience=cand.get('experience', ''),
                education=cand.get('education', ''),
                summary=cand.get('summary', '')
            )
            candidate_embeddings.append(emb)
        
        # Calculate pairwise similarities
        jd_matrix = np.array(jd_embeddings)
        candidate_matrix = np.array(candidate_embeddings)
        
        # JD-JD similarity (should have some diversity)
        jd_similarities = cosine_similarity(jd_matrix)
        np.fill_diagonal(jd_similarities, 0)  # Remove self-similarity
        
        # Candidate-Candidate similarity
        candidate_similarities = cosine_similarity(candidate_matrix)
        np.fill_diagonal(candidate_similarities, 0)
        
        # JD-Candidate similarity (matching quality)
        cross_similarities = cosine_similarity(jd_matrix, candidate_matrix)
        
        metrics['jd_self_similarity_mean'] = float(np.mean(jd_similarities))
        metrics['jd_self_similarity_std'] = float(np.std(jd_similarities))
        metrics['candidate_self_similarity_mean'] = float(np.mean(candidate_similarities))
        metrics['candidate_self_similarity_std'] = float(np.std(candidate_similarities))
        metrics['cross_similarity_mean'] = float(np.mean(cross_similarities))
        metrics['cross_similarity_std'] = float(np.std(cross_similarities))
        metrics['cross_similarity_max'] = float(np.max(cross_similarities))
        metrics['cross_similarity_min'] = float(np.min(cross_similarities))
        
        # Calculate top-k statistics
        top_k = 5
        top_k_similarities = []
        for i in range(len(jd_embeddings)):
            similarities = cross_similarities[i]
            top_k_indices = np.argsort(similarities)[-top_k:]
            top_k_similarities.extend(similarities[top_k_indices].tolist())
        
        metrics['top_5_similarity_mean'] = float(np.mean(top_k_similarities))
        metrics['top_5_similarity_std'] = float(np.std(top_k_similarities))
        
        return metrics
    
    def benchmark_memory_usage(self, variation) -> Dict:
        """Benchmark memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate some embeddings to load model into memory
        test_texts = [f"Test text {i}" for i in range(100)]
        _ = variation.generate_embeddings_batch(test_texts)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        return {
            'memory_usage_mb': memory_used,
            'memory_after_mb': memory_after
        }
    
    def run_benchmark(self, variation_ids: List[int] = None, use_parameter_variations: bool = True):
        """Run comprehensive benchmark for all or specified variations."""
        # Determine which variation system to use
        if use_parameter_variations:
            # Use parameter variations (5 per model)
            all_variations = generate_all_variations()
            if variation_ids is None:
                variation_ids = list(range(1, len(all_variations) + 1))
            logger.info(f"Starting benchmark for {len(variation_ids)} parameter variations...")
            get_var_func = get_param_variation
            var_type = "Parameter Variation"
        else:
            # Use original variations (10 total)
            if variation_ids is None:
                variation_ids = list(range(1, 11))
            logger.info(f"Starting benchmark for {len(variation_ids)} variations...")
            get_var_func = get_variation
            var_type = "Variation"
        
        # Load test data
        jd_data, candidate_data = self.load_test_data()
        
        if not jd_data or not candidate_data:
            logger.error("No test data available. Please ensure embeddings are generated first.")
            return
        
        all_results = []
        
        for var_id in variation_ids:
            try:
                logger.info(f"\n{'='*80}")
                logger.info(f"Benchmarking {var_type} {var_id}...")
                logger.info(f"{'='*80}")
                
                variation = get_var_func(var_id)
                
                # Run benchmarks
                embedding_metrics = self.benchmark_embedding_generation(
                    variation, jd_data, candidate_data
                )
                
                similarity_metrics = self.benchmark_similarity_quality(
                    variation, jd_data, candidate_data
                )
                
                memory_metrics = self.benchmark_memory_usage(variation)
                
                # Combine all metrics
                result = {
                    **embedding_metrics,
                    **similarity_metrics,
                    **memory_metrics,
                    'benchmark_timestamp': datetime.now().isoformat()
                }
                
                all_results.append(result)
                self.results.append(result)
                
                logger.info(f"Completed benchmark for {variation.name}")
                logger.info(f"  JD Single Avg Time: {result['jd_single_avg_time']:.4f}s")
                logger.info(f"  JD Batch Throughput: {result['jd_batch_throughput']:.2f} embeddings/s")
                logger.info(f"  Cross Similarity Mean: {result['cross_similarity_mean']:.4f}")
                logger.info(f"  Memory Usage: {result['memory_usage_mb']:.2f} MB")
                
            except Exception as e:
                logger.error(f"Error benchmarking variation {var_id}: {e}", exc_info=True)
                continue
        
        # Save results
        self.save_results(all_results)
        
        # Generate comparison report
        self.generate_comparison_report(all_results)
        
        logger.info(f"\n{'='*80}")
        logger.info("Benchmark completed!")
        logger.info(f"{'='*80}")
    
    def save_results(self, results: List[Dict]):
        """Save benchmark results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.reports_dir / f"benchmark_results_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_file}")
    
    def generate_comparison_report(self, results: List[Dict]):
        """Generate comparison report."""
        if not results:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"benchmark_report_{timestamp}.md"
        
        # Sort by overall score (combination of speed and quality)
        for result in results:
            # Calculate composite score (higher is better)
            speed_score = 1 / (result['jd_single_avg_time'] + 0.001)  # Inverse of time
            quality_score = result['cross_similarity_mean']  # Similarity quality
            throughput_score = result['jd_batch_throughput'] / 100  # Normalized throughput
            
            result['composite_score'] = (speed_score * 0.3 + quality_score * 0.5 + throughput_score * 0.2)
        
        results_sorted = sorted(results, key=lambda x: x.get('composite_score', 0), reverse=True)
        
        # Generate markdown report
        report_lines = [
            "# Model Variations Benchmark Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"Benchmarked {len(results)} model variations.",
            "",
            "## Rankings (by Composite Score)",
            "",
            "| Rank | Variation | Model | Composite Score | JD Time (s) | Batch Throughput | Cross Similarity | Memory (MB) |",
            "|------|-----------|-------|----------------|-------------|------------------|------------------|-------------|"
        ]
        
        for rank, result in enumerate(results_sorted, 1):
            report_lines.append(
                f"| {rank} | {result['variation_name']} | {result['model_name']} | "
                f"{result.get('composite_score', 0):.4f} | {result['jd_single_avg_time']:.4f} | "
                f"{result['jd_batch_throughput']:.2f} | {result['cross_similarity_mean']:.4f} | "
                f"{result['memory_usage_mb']:.2f} |"
            )
        
        report_lines.extend([
            "",
            "## Detailed Metrics",
            ""
        ])
        
        for result in results_sorted:
            report_lines.extend([
                f"### {result['variation_name']}",
                "",
                f"- **Model**: {result['model_name']}",
                f"- **Dimension**: {result['dimension']}",
                f"- **JD Single Avg Time**: {result['jd_single_avg_time']:.4f}s",
                f"- **JD Batch Throughput**: {result['jd_batch_throughput']:.2f} embeddings/s",
                f"- **Candidate Single Avg Time**: {result['candidate_single_avg_time']:.4f}s",
                f"- **Cross Similarity Mean**: {result['cross_similarity_mean']:.4f}",
                f"- **Cross Similarity Std**: {result['cross_similarity_std']:.4f}",
                f"- **Top-5 Similarity Mean**: {result['top_5_similarity_mean']:.4f}",
                f"- **Memory Usage**: {result['memory_usage_mb']:.2f} MB",
                f"- **Composite Score**: {result.get('composite_score', 0):.4f}",
                ""
            ])
        
        report_lines.extend([
            "## Recommendations",
            "",
            f"**Best Overall**: {results_sorted[0]['variation_name']}",
            f"  - Model: {results_sorted[0]['model_name']}",
            f"  - Composite Score: {results_sorted[0].get('composite_score', 0):.4f}",
            "",
            f"**Fastest**: {min(results, key=lambda x: x['jd_single_avg_time'])['variation_name']}",
            f"  - Time: {min(results, key=lambda x: x['jd_single_avg_time'])['jd_single_avg_time']:.4f}s",
            "",
            f"**Best Quality**: {max(results, key=lambda x: x['cross_similarity_mean'])['variation_name']}",
            f"  - Similarity: {max(results, key=lambda x: x['cross_similarity_mean'])['cross_similarity_mean']:.4f}",
            ""
        ])
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Report saved to {report_file}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark model variations')
    parser.add_argument('--sample-size', type=int, default=100,
                       help='Number of samples to use for testing')
    parser.add_argument('--variations', type=int, nargs='+', default=None,
                       help='Specific variation IDs to benchmark (default: all)')
    parser.add_argument('--use-param-variations', action='store_true', default=True,
                       help='Use parameter variations (5 per model) instead of original 10 variations')
    parser.add_argument('--use-original', action='store_true', default=False,
                       help='Use original 10 variations instead of parameter variations')
    
    args = parser.parse_args()
    
    use_param = args.use_param_variations and not args.use_original
    
    benchmark = ModelVariationBenchmark(test_sample_size=args.sample_size)
    benchmark.run_benchmark(
        variation_ids=args.variations,
        use_parameter_variations=use_param
    )


if __name__ == "__main__":
    main()

