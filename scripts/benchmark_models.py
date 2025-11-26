"""Benchmark comparison between two embedding models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import time
import pandas as pd
from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.database.repository import EmbeddingRepository
from src.embeddings.generator import EmbeddingGenerator
from src.embeddings.weighted_embedding import WeightedEmbeddingGenerator
import numpy as np
from typing import List, Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelBenchmark:
    """Benchmark comparison between embedding models."""
    
    def __init__(self, model1_name: str, model2_name: str):
        """
        Initialize benchmark with two models.
        
        Args:
            model1_name: First model to compare (e.g., old model)
            model2_name: Second model to compare (e.g., new model)
        """
        self.model1_name = model1_name
        self.model2_name = model2_name
        
        logger.info(f"Initializing Model 1: {model1_name}")
        self.generator1 = EmbeddingGenerator(model_name=model1_name)
        
        logger.info(f"Initializing Model 2: {model2_name}")
        self.generator2 = EmbeddingGenerator(model_name=model2_name)
        
        self.dimension1 = self.generator1.get_embedding_dimension()
        self.dimension2 = self.generator2.get_embedding_dimension()
        
        logger.info(f"Model 1 dimension: {self.dimension1}")
        logger.info(f"Model 2 dimension: {self.dimension2}")
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def benchmark_embedding_speed(self, texts: List[str], num_runs: int = 3) -> Dict:
        """Benchmark embedding generation speed for both models."""
        logger.info("=" * 80)
        logger.info("BENCHMARKING EMBEDDING GENERATION SPEED")
        logger.info("=" * 80)
        
        results = {
            'model1': {'times': [], 'avg_time': 0, 'texts_per_sec': 0},
            'model2': {'times': [], 'avg_time': 0, 'texts_per_sec': 0}
        }
        
        for run in range(num_runs):
            logger.info(f"\nRun {run + 1}/{num_runs}")
            
            # Model 1
            start = time.time()
            embeddings1 = self.generator1.generate_embeddings_batch(texts)
            time1 = time.time() - start
            results['model1']['times'].append(time1)
            
            # Model 2
            start = time.time()
            embeddings2 = self.generator2.generate_embeddings_batch(texts)
            time2 = time.time() - start
            results['model2']['times'].append(time2)
            
            logger.info(f"  Model 1 ({self.model1_name}): {time1:.3f}s ({len(texts)/time1:.1f} texts/sec)")
            logger.info(f"  Model 2 ({self.model2_name}): {time2:.3f}s ({len(texts)/time2:.1f} texts/sec)")
        
        # Calculate averages
        results['model1']['avg_time'] = np.mean(results['model1']['times'])
        results['model2']['avg_time'] = np.mean(results['model2']['times'])
        results['model1']['texts_per_sec'] = len(texts) / results['model1']['avg_time']
        results['model2']['texts_per_sec'] = len(texts) / results['model2']['avg_time']
        
        logger.info("\n" + "-" * 80)
        logger.info("AVERAGE RESULTS:")
        logger.info(f"  Model 1: {results['model1']['avg_time']:.3f}s ({results['model1']['texts_per_sec']:.1f} texts/sec)")
        logger.info(f"  Model 2: {results['model2']['avg_time']:.3f}s ({results['model2']['texts_per_sec']:.1f} texts/sec)")
        
        speedup = results['model1']['avg_time'] / results['model2']['avg_time']
        if speedup > 1:
            logger.info(f"  Model 1 is {speedup:.2f}x faster")
        else:
            logger.info(f"  Model 2 is {1/speedup:.2f}x faster")
        
        return results
    
    def benchmark_matching_quality(
        self,
        candidate_texts: List[str],
        jd_embeddings_db: List,
        top_k: int = 5
    ) -> Dict:
        """Benchmark matching quality between two models."""
        logger.info("=" * 80)
        logger.info("BENCHMARKING MATCHING QUALITY")
        logger.info("=" * 80)
        
        results = {
            'model1': {
                'similarities': [],
                'avg_similarity': 0,
                'max_similarity': 0,
                'min_similarity': 0,
                'matches': []
            },
            'model2': {
                'similarities': [],
                'avg_similarity': 0,
                'max_similarity': 0,
                'min_similarity': 0,
                'matches': []
            }
        }
        
        # Generate embeddings for all JDs with both models
        logger.info("Generating JD embeddings with both models...")
        jd_texts = []
        for jd in jd_embeddings_db[:100]:  # Use first 100 JDs for comparison
            # JD model has: title, description, requirements (no skills field)
            jd_text = f"{jd.title} {jd.description or ''} {jd.requirements or ''}"
            jd_texts.append(jd_text.strip())
        
        logger.info(f"Generating embeddings for {len(jd_texts)} JDs with Model 1...")
        jd_embeddings1 = self.generator1.generate_embeddings_batch(jd_texts)
        
        logger.info(f"Generating embeddings for {len(jd_texts)} JDs with Model 2...")
        jd_embeddings2 = self.generator2.generate_embeddings_batch(jd_texts)
        
        # Test with each candidate
        for idx, candidate_text in enumerate(candidate_texts, 1):
            logger.info(f"\nTesting Candidate {idx}/{len(candidate_texts)}")
            logger.info(f"  Text: {candidate_text[:100]}...")
            
            # Generate candidate embeddings
            candidate_emb1 = self.generator1.generate_embedding(candidate_text)
            candidate_emb2 = self.generator2.generate_embedding(candidate_text)
            
            # Find top matches for Model 1
            similarities1 = []
            for jd_emb in jd_embeddings1:
                sim = self.cosine_similarity(candidate_emb1, jd_emb)
                similarities1.append(sim)
            
            # Find top matches for Model 2
            similarities2 = []
            for jd_emb in jd_embeddings2:
                sim = self.cosine_similarity(candidate_emb2, jd_emb)
                similarities2.append(sim)
            
            # Get top K matches
            top_indices1 = np.argsort(similarities1)[-top_k:][::-1]
            top_indices2 = np.argsort(similarities2)[-top_k:][::-1]
            
            top_sims1 = [similarities1[i] for i in top_indices1]
            top_sims2 = [similarities2[i] for i in top_indices2]
            
            results['model1']['similarities'].extend(top_sims1)
            results['model2']['similarities'].extend(top_sims2)
            
            results['model1']['matches'].append({
                'candidate_idx': idx,
                'top_similarities': top_sims1,
                'top_indices': top_indices1.tolist()
            })
            results['model2']['matches'].append({
                'candidate_idx': idx,
                'top_similarities': top_sims2,
                'top_indices': top_indices2.tolist()
            })
            
            logger.info(f"  Model 1 - Top similarity: {max(top_sims1)*100:.2f}%")
            logger.info(f"  Model 2 - Top similarity: {max(top_sims2)*100:.2f}%")
        
        # Calculate statistics
        results['model1']['avg_similarity'] = np.mean(results['model1']['similarities'])
        results['model2']['avg_similarity'] = np.mean(results['model2']['similarities'])
        results['model1']['max_similarity'] = np.max(results['model1']['similarities'])
        results['model2']['max_similarity'] = np.max(results['model2']['similarities'])
        results['model1']['min_similarity'] = np.min(results['model1']['similarities'])
        results['model2']['min_similarity'] = np.min(results['model2']['similarities'])
        
        logger.info("\n" + "-" * 80)
        logger.info("MATCHING QUALITY SUMMARY:")
        logger.info(f"  Model 1 ({self.model1_name}):")
        logger.info(f"    Avg similarity: {results['model1']['avg_similarity']*100:.2f}%")
        logger.info(f"    Max similarity: {results['model1']['max_similarity']*100:.2f}%")
        logger.info(f"    Min similarity: {results['model1']['min_similarity']*100:.2f}%")
        logger.info(f"  Model 2 ({self.model2_name}):")
        logger.info(f"    Avg similarity: {results['model2']['avg_similarity']*100:.2f}%")
        logger.info(f"    Max similarity: {results['model2']['max_similarity']*100:.2f}%")
        logger.info(f"    Min similarity: {results['model2']['min_similarity']*100:.2f}%")
        
        improvement = ((results['model2']['avg_similarity'] - results['model1']['avg_similarity']) / 
                      results['model1']['avg_similarity']) * 100
        logger.info(f"\n  Improvement: {improvement:+.2f}%")
        
        return results
    
    def detailed_comparison(
        self,
        candidate_texts: List[str],
        jd_embeddings_db: List,
        top_k: int = 5
    ):
        """Detailed side-by-side comparison of matching results."""
        logger.info("=" * 80)
        logger.info("DETAILED SIDE-BY-SIDE COMPARISON")
        logger.info("=" * 80)
        
        # Generate JD embeddings
        jd_texts = []
        for jd in jd_embeddings_db[:100]:
            # JD model has: title, description, requirements (no skills field)
            jd_text = f"{jd.title} {jd.description or ''} {jd.requirements or ''}"
            jd_texts.append(jd_text.strip())
        
        jd_embeddings1 = self.generator1.generate_embeddings_batch(jd_texts)
        jd_embeddings2 = self.generator2.generate_embeddings_batch(jd_texts)
        
        for idx, candidate_text in enumerate(candidate_texts, 1):
            logger.info("\n" + "=" * 80)
            logger.info(f"CANDIDATE {idx}: {candidate_text[:80]}...")
            logger.info("=" * 80)
            
            # Generate candidate embeddings
            candidate_emb1 = self.generator1.generate_embedding(candidate_text)
            candidate_emb2 = self.generator2.generate_embedding(candidate_text)
            
            # Calculate similarities
            similarities1 = [self.cosine_similarity(candidate_emb1, jd_emb) for jd_emb in jd_embeddings1]
            similarities2 = [self.cosine_similarity(candidate_emb2, jd_emb) for jd_emb in jd_embeddings2]
            
            # Get top K
            top_indices1 = np.argsort(similarities1)[-top_k:][::-1]
            top_indices2 = np.argsort(similarities2)[-top_k:][::-1]
            
            logger.info(f"\n{self.model1_name} - TOP {top_k} MATCHES:")
            logger.info("-" * 80)
            for i, jd_idx in enumerate(top_indices1, 1):
                jd = jd_embeddings_db[jd_idx]
                sim = similarities1[jd_idx]
                logger.info(f"  {i}. {jd.title} (Similarity: {sim*100:.2f}%)")
                logger.info(f"     Job ID: {jd.job_id}, Company: {jd.company or 'N/A'}")
            
            logger.info(f"\n{self.model2_name} - TOP {top_k} MATCHES:")
            logger.info("-" * 80)
            for i, jd_idx in enumerate(top_indices2, 1):
                jd = jd_embeddings_db[jd_idx]
                sim = similarities2[jd_idx]
                logger.info(f"  {i}. {jd.title} (Similarity: {sim*100:.2f}%)")
                logger.info(f"     Job ID: {jd.job_id}, Company: {jd.company or 'N/A'}")
            
            # Compare overlap
            overlap = len(set(top_indices1) & set(top_indices2))
            logger.info(f"\n  Overlap: {overlap}/{top_k} jobs appear in both top {top_k}")


def run_benchmark():
    """Run comprehensive benchmark comparison."""
    logger.info("=" * 80)
    logger.info("EMBEDDING MODEL BENCHMARK COMPARISON")
    logger.info("=" * 80)
    
    # Model configurations
    model1_name = "sentence-transformers/all-MiniLM-L6-v2"  # Old model (English-only)
    model2_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # New model (Vietnamese)
    
    logger.info(f"\nModel 1 (Old): {model1_name}")
    logger.info(f"Model 2 (New): {model2_name}")
    logger.info("")
    
    # Initialize benchmark
    benchmark = ModelBenchmark(model1_name, model2_name)
    
    # Test candidate texts (Vietnamese)
    test_candidates = [
        "Skills: Python, Machine Learning, Deep Learning, TensorFlow, PyTorch Experience: 5 năm kinh nghiệm phát triển phần mềm, chuyên về AI và Machine Learning Professional Summary: Lập trình viên Python với kinh nghiệm trong phát triển ứng dụng AI",
        "Skills: Java, Spring Boot, MySQL, REST API Experience: 3 năm phát triển backend với Java và Spring Framework Professional Summary: Backend developer chuyên về Java và microservices",
        "Skills: React, JavaScript, Node.js, MongoDB Experience: 4 năm phát triển full-stack web applications Professional Summary: Full-stack developer với kinh nghiệm React và Node.js",
        "Skills: C#, .NET, SQL Server, ASP.NET Experience: 6 năm phát triển enterprise applications Professional Summary: Senior .NET developer với kinh nghiệm trong các dự án lớn",
        "Skills: Angular, TypeScript, RxJS, HTML/CSS Experience: 3 năm phát triển frontend applications Professional Summary: Frontend developer chuyên về Angular framework"
    ]
    
    # Load JDs from database
    db: Session = SessionLocal()
    try:
        repository = EmbeddingRepository(db)
        jd_embeddings = repository.get_all_jd_embeddings()
        
        if len(jd_embeddings) < 10:
            logger.error(f"Not enough JDs in database ({len(jd_embeddings)}). Need at least 10.")
            return
        
        logger.info(f"Loaded {len(jd_embeddings)} job descriptions from database")
        logger.info("")
        
        # Benchmark 1: Embedding Speed
        logger.info("\n" + "=" * 80)
        logger.info("BENCHMARK 1: EMBEDDING GENERATION SPEED")
        logger.info("=" * 80)
        speed_results = benchmark.benchmark_embedding_speed(test_candidates, num_runs=3)
        
        # Benchmark 2: Matching Quality
        logger.info("\n" + "=" * 80)
        logger.info("BENCHMARK 2: MATCHING QUALITY")
        logger.info("=" * 80)
        quality_results = benchmark.benchmark_matching_quality(
            test_candidates,
            jd_embeddings,
            top_k=5
        )
        
        # Benchmark 3: Detailed Comparison
        logger.info("\n" + "=" * 80)
        logger.info("BENCHMARK 3: DETAILED SIDE-BY-SIDE COMPARISON")
        logger.info("=" * 80)
        benchmark.detailed_comparison(
            test_candidates,
            jd_embeddings,
            top_k=5
        )
        
        # Final Summary
        logger.info("\n" + "=" * 80)
        logger.info("FINAL BENCHMARK SUMMARY")
        logger.info("=" * 80)
        logger.info(f"\nModel 1 ({model1_name}):")
        logger.info(f"  Dimension: {benchmark.dimension1}")
        logger.info(f"  Avg Speed: {speed_results['model1']['texts_per_sec']:.1f} texts/sec")
        logger.info(f"  Avg Similarity: {quality_results['model1']['avg_similarity']*100:.2f}%")
        logger.info(f"  Max Similarity: {quality_results['model1']['max_similarity']*100:.2f}%")
        
        logger.info(f"\nModel 2 ({model2_name}):")
        logger.info(f"  Dimension: {benchmark.dimension2}")
        logger.info(f"  Avg Speed: {speed_results['model2']['texts_per_sec']:.1f} texts/sec")
        logger.info(f"  Avg Similarity: {quality_results['model2']['avg_similarity']*100:.2f}%")
        logger.info(f"  Max Similarity: {quality_results['model2']['max_similarity']*100:.2f}%")
        
        logger.info("\n" + "-" * 80)
        speed_ratio = speed_results['model1']['avg_time'] / speed_results['model2']['avg_time']
        if speed_ratio < 1:
            logger.info(f"Speed: Model 1 is {1/speed_ratio:.2f}x faster")
        else:
            logger.info(f"Speed: Model 2 is {speed_ratio:.2f}x faster")
        
        similarity_improvement = ((quality_results['model2']['avg_similarity'] - 
                                  quality_results['model1']['avg_similarity']) / 
                                 quality_results['model1']['avg_similarity']) * 100
        logger.info(f"Quality: Model 2 improves similarity by {similarity_improvement:+.2f}%")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    run_benchmark()

