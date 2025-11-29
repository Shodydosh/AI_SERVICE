"""Comprehensive benchmark script with detailed logging for report generation."""
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
import traceback

from src.embeddings.parameter_variations import (
    generate_all_variations, 
    get_variation_by_id as get_param_variation,
    list_all_variations as list_param_variations
)
from src.database.connection import SessionLocal
from src.database.models import JobDescriptionEmbedding, CandidateEmbedding
from sklearn.metrics.pairwise import cosine_similarity

# Setup detailed logging
log_dir = Path("reports/benchmark_variations/logs")
log_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"benchmark_{timestamp}.log"

# Create logger
logger = logging.getLogger('benchmark')
logger.setLevel(logging.DEBUG)

# File handler with detailed formatting
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
file_handler.setFormatter(file_formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("="*80)
logger.info("Starting Comprehensive Benchmark with Detailed Logging")
logger.info("="*80)
logger.info(f"Log file: {log_file}")
logger.info(f"Timestamp: {timestamp}")


class DetailedBenchmark:
    """Comprehensive benchmark with detailed logging."""
    
    def __init__(self, test_sample_size: int = 50):
        self.test_sample_size = test_sample_size
        self.results = []
        self.reports_dir = Path("reports/benchmark_variations")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        
        logger.info(f"Initialized benchmark with sample size: {test_sample_size}")
    
    def load_test_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load sample JD and candidate data for testing."""
        logger.info("Loading test data from database...")
        
        try:
            session = SessionLocal()
            try:
                # Get sample JDs
                logger.debug("Querying JobDescriptionEmbedding table...")
                jds = session.query(JobDescriptionEmbedding).limit(self.test_sample_size).all()
                logger.info(f"Found {len(jds)} JDs in database")
                
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
                logger.debug("Querying CandidateEmbedding table...")
                candidates = session.query(CandidateEmbedding).limit(self.test_sample_size).all()
                logger.info(f"Found {len(candidates)} candidates in database")
                
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
            finally:
                session.close()
            
        except Exception as e:
            logger.error(f"Error loading test data: {e}", exc_info=True)
            raise
    
    def benchmark_embedding_generation(self, variation, jd_data: List[Dict], 
                                       candidate_data: List[Dict]) -> Dict:
        """Benchmark embedding generation speed and quality."""
        logger.info(f"Benchmarking embedding generation for {variation.name}...")
        logger.debug(f"  Model: {variation.model_name}")
        logger.debug(f"  Batch size: {variation.batch_size}")
        logger.debug(f"  Normalize: {variation.normalize}")
        logger.debug(f"  Use tokenization: {variation.use_tokenization}")
        
        metrics = {
            'variation_id': variation.variation_id,
            'variation_name': variation.name,
            'model_name': variation.model_name,
            'base_name': getattr(variation, 'base_name', 'Unknown'),
            'dimension': variation.dimension,
            'batch_size': variation.batch_size,
            'normalize': variation.normalize,
            'use_tokenization': variation.use_tokenization
        }
        
        # Test JD embedding generation (single)
        logger.debug("Testing single JD embedding generation...")
        jd_times = []
        jd_embeddings = []
        
        test_jds = jd_data[:10]
        for idx, jd in enumerate(test_jds):
            try:
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
                logger.debug(f"  JD {idx+1}: {elapsed:.4f}s")
            except Exception as e:
                logger.error(f"  Error generating JD {idx+1} embedding: {e}")
                continue
        
        # Test batch JD embedding generation
        logger.debug("Testing batch JD embedding generation...")
        batch_start = time.time()
        batch_jd_texts = [
            f"Title: {jd.get('title', '')} Skills: {jd.get('skills', '')} "
            f"Requirements: {jd.get('requirements', '')} Description: {jd.get('description', '')}"
            for jd in jd_data[:50]
        ]
        try:
            batch_jd_embeddings = variation.generate_embeddings_batch(batch_jd_texts)
            batch_jd_time = time.time() - batch_start
            logger.debug(f"  Batch JD time: {batch_jd_time:.4f}s for {len(batch_jd_texts)} texts")
        except Exception as e:
            logger.error(f"  Error in batch JD generation: {e}")
            batch_jd_time = 0
            batch_jd_embeddings = []
        
        # Test candidate embedding generation (single)
        logger.debug("Testing single candidate embedding generation...")
        candidate_times = []
        candidate_embeddings = []
        
        test_candidates = candidate_data[:10]
        for idx, cand in enumerate(test_candidates):
            try:
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
                logger.debug(f"  Candidate {idx+1}: {elapsed:.4f}s")
            except Exception as e:
                logger.error(f"  Error generating candidate {idx+1} embedding: {e}")
                continue
        
        # Test batch candidate embedding generation
        logger.debug("Testing batch candidate embedding generation...")
        batch_start = time.time()
        batch_candidate_texts = [
            f"Skills: {cand.get('skills', '')} Experience: {cand.get('experience', '')} "
            f"Summary: {cand.get('summary', '')} Education: {cand.get('education', '')}"
            for cand in candidate_data[:50]
        ]
        try:
            batch_candidate_embeddings = variation.generate_embeddings_batch(batch_candidate_texts)
            batch_candidate_time = time.time() - batch_start
            logger.debug(f"  Batch candidate time: {batch_candidate_time:.4f}s for {len(batch_candidate_texts)} texts")
        except Exception as e:
            logger.error(f"  Error in batch candidate generation: {e}")
            batch_candidate_time = 0
            batch_candidate_embeddings = []
        
        # Calculate metrics
        metrics['jd_single_avg_time'] = float(np.mean(jd_times)) if jd_times else 0.0
        metrics['jd_single_min_time'] = float(np.min(jd_times)) if jd_times else 0.0
        metrics['jd_single_max_time'] = float(np.max(jd_times)) if jd_times else 0.0
        metrics['jd_single_std_time'] = float(np.std(jd_times)) if jd_times else 0.0
        metrics['jd_batch_time'] = batch_jd_time
        metrics['jd_batch_throughput'] = len(batch_jd_texts) / batch_jd_time if batch_jd_time > 0 else 0
        
        metrics['candidate_single_avg_time'] = float(np.mean(candidate_times)) if candidate_times else 0.0
        metrics['candidate_single_min_time'] = float(np.min(candidate_times)) if candidate_times else 0.0
        metrics['candidate_single_max_time'] = float(np.max(candidate_times)) if candidate_times else 0.0
        metrics['candidate_single_std_time'] = float(np.std(candidate_times)) if candidate_times else 0.0
        metrics['candidate_batch_time'] = batch_candidate_time
        metrics['candidate_batch_throughput'] = len(batch_candidate_texts) / batch_candidate_time if batch_candidate_time > 0 else 0
        
        # Calculate embedding quality metrics
        all_embeddings = jd_embeddings + candidate_embeddings
        if all_embeddings:
            embedding_matrix = np.array(all_embeddings)
            metrics['embedding_mean_norm'] = float(np.mean(np.linalg.norm(embedding_matrix, axis=1)))
            metrics['embedding_std_norm'] = float(np.std(np.linalg.norm(embedding_matrix, axis=1)))
            metrics['embedding_mean_variance'] = float(np.mean(np.var(embedding_matrix, axis=0)))
            metrics['embedding_min'] = float(np.min(embedding_matrix))
            metrics['embedding_max'] = float(np.max(embedding_matrix))
            metrics['embedding_mean'] = float(np.mean(embedding_matrix))
        
        logger.info(f"  JD Single Avg Time: {metrics['jd_single_avg_time']:.4f}s")
        logger.info(f"  JD Batch Throughput: {metrics['jd_batch_throughput']:.2f} embeddings/s")
        logger.info(f"  Candidate Single Avg Time: {metrics['candidate_single_avg_time']:.4f}s")
        
        return metrics
    
    def benchmark_similarity_quality(self, variation, jd_data: List[Dict], 
                                     candidate_data: List[Dict]) -> Dict:
        """Benchmark similarity quality using cosine similarity."""
        logger.info(f"Benchmarking similarity quality for {variation.name}...")
        
        metrics = {}
        
        # Generate embeddings for test data
        test_size = min(20, len(jd_data), len(candidate_data))
        logger.debug(f"  Generating embeddings for {test_size} JDs and {test_size} candidates...")
        
        jd_embeddings = []
        for idx, jd in enumerate(jd_data[:test_size]):
            try:
                emb = variation.generate_jd_embedding(
                    title=jd.get('title', ''),
                    description=jd.get('description', ''),
                    requirements=jd.get('requirements', ''),
                    skills=jd.get('skills', '')
                )
                jd_embeddings.append(emb)
            except Exception as e:
                logger.error(f"  Error generating JD {idx+1} embedding: {e}")
                continue
        
        candidate_embeddings = []
        for idx, cand in enumerate(candidate_data[:test_size]):
            try:
                emb = variation.generate_candidate_embedding(
                    skills=cand.get('skills', ''),
                    experience=cand.get('experience', ''),
                    education=cand.get('education', ''),
                    summary=cand.get('summary', '')
                )
                candidate_embeddings.append(emb)
            except Exception as e:
                logger.error(f"  Error generating candidate {idx+1} embedding: {e}")
                continue
        
        if not jd_embeddings or not candidate_embeddings:
            logger.warning("  Not enough embeddings for similarity test")
            return metrics
        
        # Calculate pairwise similarities
        logger.debug("  Calculating similarity matrices...")
        jd_matrix = np.array(jd_embeddings)
        candidate_matrix = np.array(candidate_embeddings)
        
        # JD-JD similarity
        jd_similarities = cosine_similarity(jd_matrix)
        np.fill_diagonal(jd_similarities, 0)
        
        # Candidate-Candidate similarity
        candidate_similarities = cosine_similarity(candidate_matrix)
        np.fill_diagonal(candidate_similarities, 0)
        
        # JD-Candidate similarity (cross similarity)
        cross_similarities = cosine_similarity(jd_matrix, candidate_matrix)
        
        metrics['jd_self_similarity_mean'] = float(np.mean(jd_similarities))
        metrics['jd_self_similarity_std'] = float(np.std(jd_similarities))
        metrics['jd_self_similarity_min'] = float(np.min(jd_similarities))
        metrics['jd_self_similarity_max'] = float(np.max(jd_similarities))
        
        metrics['candidate_self_similarity_mean'] = float(np.mean(candidate_similarities))
        metrics['candidate_self_similarity_std'] = float(np.std(candidate_similarities))
        metrics['candidate_self_similarity_min'] = float(np.min(candidate_similarities))
        metrics['candidate_self_similarity_max'] = float(np.max(candidate_similarities))
        
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
        metrics['top_5_similarity_min'] = float(np.min(top_k_similarities))
        metrics['top_5_similarity_max'] = float(np.max(top_k_similarities))
        
        logger.info(f"  Cross Similarity Mean: {metrics['cross_similarity_mean']:.4f}")
        logger.info(f"  Top-5 Similarity Mean: {metrics['top_5_similarity_mean']:.4f}")
        
        return metrics
    
    def benchmark_memory_usage(self, variation) -> Dict:
        """Benchmark memory usage."""
        logger.debug(f"Benchmarking memory usage for {variation.name}...")
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate some embeddings to load model into memory
        test_texts = [f"Test text {i}" for i in range(100)]
        try:
            _ = variation.generate_embeddings_batch(test_texts)
        except Exception as e:
            logger.error(f"  Error in memory test: {e}")
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        logger.debug(f"  Memory usage: {memory_used:.2f} MB")
        
        return {
            'memory_usage_mb': memory_used,
            'memory_after_mb': memory_after
        }
    
    def run_benchmark(self, variation_ids: List[int] = None):
        """Run comprehensive benchmark for all or specified variations."""
        all_variations = generate_all_variations()
        
        if variation_ids is None:
            variation_ids = list(range(1, len(all_variations) + 1))
        
        logger.info(f"Starting benchmark for {len(variation_ids)} variations...")
        logger.info(f"Total variations available: {len(all_variations)}")
        
        # Load test data
        jd_data, candidate_data = self.load_test_data()
        
        if not jd_data or not candidate_data:
            logger.error("No test data available. Please ensure embeddings are generated first.")
            return
        
        all_results = []
        successful = 0
        failed = 0
        
        for idx, var_id in enumerate(variation_ids, 1):
            try:
                logger.info(f"\n{'='*80}")
                logger.info(f"Benchmarking Variation {var_id} ({idx}/{len(variation_ids)})")
                logger.info(f"{'='*80}")
                
                variation_start = time.time()
                variation = get_param_variation(var_id)
                
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
                    'benchmark_timestamp': datetime.now().isoformat(),
                    'variation_time_seconds': time.time() - variation_start
                }
                
                all_results.append(result)
                self.results.append(result)
                successful += 1
                
                logger.info(f"✓ Completed benchmark for {variation.name}")
                logger.info(f"  Time taken: {result['variation_time_seconds']:.2f}s")
                logger.info(f"  JD Single Avg Time: {result['jd_single_avg_time']:.4f}s")
                logger.info(f"  JD Batch Throughput: {result['jd_batch_throughput']:.2f} embeddings/s")
                logger.info(f"  Cross Similarity Mean: {result.get('cross_similarity_mean', 0):.4f}")
                logger.info(f"  Memory Usage: {result['memory_usage_mb']:.2f} MB")
                
            except Exception as e:
                failed += 1
                logger.error(f"✗ Error benchmarking variation {var_id}: {e}", exc_info=True)
                logger.error(traceback.format_exc())
                continue
        
        # Save results
        self.save_results(all_results)
        
        # Generate comparison report
        self.generate_comparison_report(all_results)
        
        total_time = time.time() - self.start_time
        logger.info(f"\n{'='*80}")
        logger.info("Benchmark completed!")
        logger.info(f"{'='*80}")
        logger.info(f"Total time: {total_time/60:.2f} minutes")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Results saved to: {self.reports_dir}")
    
    def save_results(self, results: List[Dict]):
        """Save benchmark results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.reports_dir / f"benchmark_results_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_file}")
        
        # Also save as CSV for easy analysis
        if results:
            df = pd.DataFrame(results)
            csv_file = self.reports_dir / f"benchmark_results_{timestamp}.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8')
            logger.info(f"Results also saved as CSV to {csv_file}")
    
    def generate_comparison_report(self, results: List[Dict]):
        """Generate detailed comparison report."""
        if not results:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"benchmark_report_{timestamp}.md"
        
        # Calculate composite scores
        for result in results:
            speed_score = 1 / (result['jd_single_avg_time'] + 0.001)
            quality_score = result.get('cross_similarity_mean', 0)
            throughput_score = result['jd_batch_throughput'] / 100
            
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
            f"Total time: {(time.time() - self.start_time)/60:.2f} minutes",
            "",
            "## Rankings (by Composite Score)",
            "",
            "| Rank | Variation | Model | Batch Size | Norm | Composite | JD Time (s) | Throughput | Cross Sim | Memory (MB) |",
            "|------|-----------|-------|------------|------|-----------|-------------|-------------|-----------|-------------|"
        ]
        
        for rank, result in enumerate(results_sorted, 1):
            report_lines.append(
                f"| {rank} | {result['variation_name']} | {result['model_name'][:30]}... | "
                f"{result['batch_size']} | {result['normalize']} | "
                f"{result.get('composite_score', 0):.4f} | {result['jd_single_avg_time']:.4f} | "
                f"{result['jd_batch_throughput']:.2f} | {result.get('cross_similarity_mean', 0):.4f} | "
                f"{result['memory_usage_mb']:.2f} |"
            )
        
        report_lines.extend([
            "",
            "## Detailed Metrics by Model",
            ""
        ])
        
        # Group by base model
        by_model = {}
        for result in results:
            base_name = result.get('base_name', 'Unknown')
            if base_name not in by_model:
                by_model[base_name] = []
            by_model[base_name].append(result)
        
        for base_name, model_results in sorted(by_model.items()):
            report_lines.extend([
                f"### {base_name}",
                "",
                "| Variation | Batch Size | Norm | JD Time | Throughput | Cross Sim | Memory |",
                "|-----------|------------|------|---------|-------------|-----------|--------|"
            ])
            
            for result in sorted(model_results, key=lambda x: x.get('composite_score', 0), reverse=True):
                report_lines.append(
                    f"| {result['variation_name']} | {result['batch_size']} | {result['normalize']} | "
                    f"{result['jd_single_avg_time']:.4f}s | {result['jd_batch_throughput']:.2f} | "
                    f"{result.get('cross_similarity_mean', 0):.4f} | {result['memory_usage_mb']:.2f}MB |"
                )
            report_lines.append("")
        
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
            f"**Best Quality**: {max(results, key=lambda x: x.get('cross_similarity_mean', 0))['variation_name']}",
            f"  - Similarity: {max(results, key=lambda x: x.get('cross_similarity_mean', 0)).get('cross_similarity_mean', 0):.4f}",
            ""
        ])
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Report saved to {report_file}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark model variations with detailed logging')
    parser.add_argument('--sample-size', type=int, default=50,
                       help='Number of samples to use for testing')
    parser.add_argument('--variations', type=int, nargs='+', default=None,
                       help='Specific variation IDs to benchmark (default: all)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of variations to benchmark (for testing)')
    
    args = parser.parse_args()
    
    benchmark = DetailedBenchmark(test_sample_size=args.sample_size)
    
    variation_ids = args.variations
    if variation_ids and args.limit:
        variation_ids = variation_ids[:args.limit]
    elif args.limit and not variation_ids:
        variation_ids = list(range(1, args.limit + 1))
    
    benchmark.run_benchmark(variation_ids=variation_ids)


if __name__ == "__main__":
    main()

