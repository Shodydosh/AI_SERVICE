"""Comprehensive embedding evaluation research script."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import time
import json
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from tqdm import tqdm

from src.database.connection import SessionLocal, engine, Base
from src.database.evaluation_models import (
    EmbeddingEvaluationJD,
    EmbeddingEvaluationCandidate,
    EmbeddingEvaluationResults
)
from src.embeddings.embedding_methods import get_embedding_method
from src.data_processing.jd_processor import JDProcessor
from src.data_processing.candidate_processor import CandidateProcessor
from src.vector_search.faiss_manager import FAISSIndexManager
from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingEvaluator:
    """Comprehensive embedding evaluation system."""
    
    def __init__(self, db: Session):
        self.db = db
        self.methods = {}
        self.results = {}
    
    def initialize_methods(self):
        """Initialize all 5 embedding methods."""
        logger.info("Initializing embedding methods...")
        
        methods_config = [
            (1, {}),  # Baseline
            (2, {}),  # Weighted
            (3, {}),  # Field-Specific
            (4, {"pooling": "weighted"}),  # Multi-Vector
            (5, {}),  # Ensemble
        ]
        
        for method_id, kwargs in methods_config:
            try:
                method = get_embedding_method(method_id, **kwargs)
                self.methods[method_id] = method
                logger.info(f"✓ Initialized Method {method_id}: {method.method_name}")
            except Exception as e:
                logger.error(f"✗ Failed to initialize Method {method_id}: {e}")
        
        logger.info(f"Initialized {len(self.methods)} methods")
    
    def create_evaluation_tables(self):
        """Create evaluation tables in database."""
        logger.info("Creating evaluation tables...")
        try:
            Base.metadata.create_all(bind=engine, tables=[
                EmbeddingEvaluationJD.__table__,
                EmbeddingEvaluationCandidate.__table__,
                EmbeddingEvaluationResults.__table__
            ])
            logger.info("✓ Evaluation tables created")
        except Exception as e:
            logger.error(f"✗ Error creating tables: {e}")
            raise
    
    def generate_and_save_embeddings(
        self,
        jd_file: str = None,
        candidate_file: str = None,
        sample_size: int = None
    ):
        """Generate embeddings using all methods and save to database."""
        logger.info("=" * 80)
        logger.info("GENERATING EMBEDDINGS WITH ALL METHODS")
        logger.info("=" * 80)
        
        # Process JD dataset
        if jd_file:
            logger.info(f"Processing JD dataset: {jd_file}")
            jd_processor = JDProcessor()
            jd_processor.load_from_csv(jd_file)
            jd_records = jd_processor.get_records()
            
            if sample_size:
                jd_records = jd_records[:sample_size]
                logger.info(f"Using sample size: {sample_size}")
            
            logger.info(f"Total JD records: {len(jd_records)}")
            
            # Generate embeddings for each method
            for method_id, method in self.methods.items():
                logger.info(f"\nGenerating JD embeddings with Method {method_id}: {method.method_name}")
                self._generate_jd_embeddings(jd_records, method_id, method)
        
        # Process Candidate dataset
        if candidate_file:
            logger.info(f"\nProcessing Candidate dataset: {candidate_file}")
            candidate_processor = CandidateProcessor(auto_map_columns=True)
            candidate_processor.load_from_csv(candidate_file)
            
            # Convert to records (list of dicts)
            if hasattr(candidate_processor, 'get_records'):
                candidate_records = candidate_processor.get_records()
            else:
                # Fallback: convert DataFrame to records
                df = candidate_processor.data
                candidate_records = df.to_dict('records')
            
            if sample_size:
                candidate_records = candidate_records[:sample_size]
            
            logger.info(f"Total Candidate records: {len(candidate_records)}")
            
            # Generate embeddings for each method
            for method_id, method in self.methods.items():
                logger.info(f"\nGenerating Candidate embeddings with Method {method_id}: {method.method_name}")
                self._generate_candidate_embeddings(candidate_records, method_id, method)
    
    def _generate_jd_embeddings(self, records: List[Dict], method_id: int, method):
        """Generate and save JD embeddings for a method."""
        batch_size = 100
        total = len(records)
        
        generation_times = []
        
        with tqdm(total=total, desc=f"Method {method_id} - JD") as pbar:
            for i in range(0, total, batch_size):
                batch = records[i:i + batch_size]
                batch_objects = []
                
                for record in batch:
                    try:
                        # Extract fields
                        title = record.get('title', '') or record.get('Job Title', '')
                        description = record.get('description', '') or record.get('Job Description', '')
                        requirements = record.get('requirements', '') or record.get('Job Requirements', '')
                        skills = record.get('skills', '')
                        company = record.get('company', '') or record.get('Name Company', '')
                        location = record.get('location', '') or record.get('Job Address', '')
                        job_id = str(record.get('job_id', '') or record.get('JobID', ''))
                        
                        # Generate embedding
                        start_time = time.time()
                        embedding = method.generate_jd_embedding(
                            title=title,
                            description=description,
                            requirements=requirements,
                            skills=skills,
                            company=company
                        )
                        gen_time = (time.time() - start_time) * 1000
                        generation_times.append(gen_time)
                        
                        # Create database object
                        eval_obj = EmbeddingEvaluationJD(
                            job_id=job_id,
                            method_id=method_id,
                            method_name=method.method_name,
                            title=title,
                            company=company,
                            description=description,
                            requirements=requirements,
                            location=location,
                            embedding=embedding
                        )
                        batch_objects.append(eval_obj)
                    
                    except Exception as e:
                        logger.warning(f"Error processing JD {record.get('job_id')}: {e}")
                        continue
                
                # Save batch
                if batch_objects:
                    try:
                        self.db.bulk_save_objects(batch_objects)
                        self.db.commit()
                    except Exception as e:
                        logger.error(f"Error saving batch: {e}")
                        self.db.rollback()
                
                pbar.update(len(batch))
        
        avg_time = np.mean(generation_times) if generation_times else 0
        logger.info(f"✓ Method {method_id} - Average generation time: {avg_time:.2f}ms")
    
    def _generate_candidate_embeddings(self, records: List[Dict], method_id: int, method):
        """Generate and save candidate embeddings for a method."""
        batch_size = 100
        total = len(records)
        
        generation_times = []
        
        with tqdm(total=total, desc=f"Method {method_id} - Candidate") as pbar:
            for i in range(0, total, batch_size):
                batch = records[i:i + batch_size]
                batch_objects = []
                
                for record in batch:
                    try:
                        # Extract fields - handle various column name formats
                        candidate_id = str(record.get('candidate_id', '') or 
                                         record.get('CandidateID', '') or 
                                         record.get('ID', '') or
                                         f"candidate_{i}")
                        name = record.get('name', '') or record.get('Name', '')
                        email = record.get('email', '') or record.get('Email', '')
                        skills = record.get('skills', '') or record.get('Skills', '')
                        experience = record.get('experience', '') or record.get('Experience', '')
                        education = record.get('education', '') or record.get('Education', '')
                        summary = record.get('summary', '') or record.get('Summary', '')
                        
                        # Generate embedding
                        start_time = time.time()
                        embedding = method.generate_candidate_embedding(
                            skills=skills,
                            experience=experience,
                            education=education,
                            summary=summary
                        )
                        gen_time = (time.time() - start_time) * 1000
                        generation_times.append(gen_time)
                        
                        # Create database object
                        eval_obj = EmbeddingEvaluationCandidate(
                            candidate_id=candidate_id,
                            method_id=method_id,
                            method_name=method.method_name,
                            name=name,
                            email=email,
                            skills=skills,
                            experience=experience,
                            education=education,
                            summary=summary,
                            embedding=embedding
                        )
                        batch_objects.append(eval_obj)
                    
                    except Exception as e:
                        logger.warning(f"Error processing candidate {record.get('candidate_id')}: {e}")
                        continue
                
                # Save batch
                if batch_objects:
                    try:
                        self.db.bulk_save_objects(batch_objects)
                        self.db.commit()
                    except Exception as e:
                        logger.error(f"Error saving batch: {e}")
                        self.db.rollback()
                
                pbar.update(len(batch))
        
        avg_time = np.mean(generation_times) if generation_times else 0
        logger.info(f"✓ Method {method_id} - Average generation time: {avg_time:.2f}ms")
    
    def verify_embeddings_saved(self):
        """Verify embeddings are saved correctly."""
        logger.info("=" * 80)
        logger.info("VERIFYING EMBEDDINGS IN DATABASE")
        logger.info("=" * 80)
        
        for method_id in self.methods.keys():
            # Check JD embeddings
            jd_count = self.db.query(EmbeddingEvaluationJD).filter(
                EmbeddingEvaluationJD.method_id == method_id
            ).count()
            
            # Check candidate embeddings
            candidate_count = self.db.query(EmbeddingEvaluationCandidate).filter(
                EmbeddingEvaluationCandidate.method_id == method_id
            ).count()
            
            method_name = self.methods[method_id].method_name
            logger.info(f"Method {method_id} ({method_name}):")
            logger.info(f"  JD embeddings: {jd_count}")
            logger.info(f"  Candidate embeddings: {candidate_count}")
            
            # Verify embedding dimensions
            if jd_count > 0:
                sample = self.db.query(EmbeddingEvaluationJD).filter(
                    EmbeddingEvaluationJD.method_id == method_id
                ).first()
                if sample:
                    dim = len(sample.embedding)
                    logger.info(f"  Embedding dimension: {dim}")
        
        logger.info("")
    
    def evaluate_methods(self, test_samples: int = 100):
        """Evaluate all methods and calculate metrics."""
        logger.info("=" * 80)
        logger.info("EVALUATING EMBEDDING METHODS")
        logger.info("=" * 80)
        
        for method_id, method in self.methods.items():
            logger.info(f"\nEvaluating Method {method_id}: {method.method_name}")
            results = self._evaluate_single_method(method_id, test_samples)
            self.results[method_id] = results
            
            # Save to database
            self._save_evaluation_results(method_id, method.method_name, results)
        
        # Print comparison
        self._print_comparison()
    
    def _evaluate_single_method(self, method_id: int, test_samples: int) -> Dict:
        """Evaluate a single method."""
        # Get embeddings
        jd_embeddings = self.db.query(EmbeddingEvaluationJD).filter(
            EmbeddingEvaluationJD.method_id == method_id
        ).limit(test_samples).all()
        
        candidate_embeddings = self.db.query(EmbeddingEvaluationCandidate).filter(
            EmbeddingEvaluationCandidate.method_id == method_id
        ).limit(test_samples).all()
        
        if not jd_embeddings or not candidate_embeddings:
            logger.warning(f"Not enough data for method {method_id}")
            return {}
        
        # Build FAISS index
        logger.info("  Building FAISS index...")
        faiss_manager = FAISSIndexManager(
            dimension=settings.EMBEDDING_DIMENSION,
            index_type="HNSW",
            normalize=True
        )
        
        # Add JD embeddings to index
        jd_vectors = []
        jd_ids = []
        for jd in jd_embeddings:
            jd_vectors.append(jd.embedding)
            jd_ids.append(jd.job_id)
        
        jd_vectors = np.array(jd_vectors, dtype=np.float32)
        faiss_manager.jd_index = faiss_manager._initialize_index("HNSW")
        faiss_manager.jd_index.add(jd_vectors)
        faiss_manager.jd_id_map = {i: jd_id for i, jd_id in enumerate(jd_ids)}
        
        # Test search performance
        search_times = []
        similarities = []
        
        logger.info("  Testing search performance...")
        for candidate in tqdm(candidate_embeddings[:min(50, len(candidate_embeddings))], 
                             desc="  Searching", leave=False):
            start_time = time.time()
            results = faiss_manager.search(
                query_embedding=candidate.embedding,
                k=10,
                dataset_type='jd'
            )
            search_time = (time.time() - start_time) * 1000
            search_times.append(search_time)
            
            if results:
                similarities.append(results[0][1])  # Top similarity score
        
        # Calculate metrics
        metrics = {
            'avg_search_time_ms': np.mean(search_times) if search_times else 0,
            'avg_similarity_score': np.mean(similarities) if similarities else 0,
            'similarity_std': np.std(similarities) if similarities else 0,
            'high_similarity_coverage': sum(1 for s in similarities if s > 0.8) / len(similarities) * 100 if similarities else 0,
            'test_samples': len(candidate_embeddings)
        }
        
        return metrics
    
    def _save_evaluation_results(self, method_id: int, method_name: str, results: Dict):
        """Save evaluation results to database."""
        try:
            eval_result = EmbeddingEvaluationResults(
                method_id=method_id,
                method_name=method_name,
                avg_search_time_ms=results.get('avg_search_time_ms', 0),
                avg_similarity_score=results.get('avg_similarity_score', 0),
                similarity_std=results.get('similarity_std', 0),
                high_similarity_coverage=results.get('high_similarity_coverage', 0),
                test_samples=results.get('test_samples', 0)
            )
            
            # Check if exists, update or insert
            existing = self.db.query(EmbeddingEvaluationResults).filter(
                EmbeddingEvaluationResults.method_id == method_id
            ).first()
            
            if existing:
                for key, value in results.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
            else:
                self.db.add(eval_result)
            
            self.db.commit()
            logger.info(f"✓ Saved evaluation results for Method {method_id}")
        except Exception as e:
            logger.error(f"✗ Error saving results: {e}")
            self.db.rollback()
    
    def _print_comparison(self):
        """Print comparison of all methods."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("EVALUATION RESULTS COMPARISON")
        logger.info("=" * 80)
        
        # Get all results from database
        all_results = self.db.query(EmbeddingEvaluationResults).all()
        
        if not all_results:
            logger.warning("No evaluation results found")
            return
        
        # Print table
        logger.info(f"{'Method':<20} {'Search Time':<15} {'Avg Similarity':<15} {'Coverage (>0.8)':<15}")
        logger.info("-" * 80)
        
        for result in sorted(all_results, key=lambda x: x.avg_similarity_score, reverse=True):
            logger.info(
                f"{result.method_name:<20} "
                f"{result.avg_search_time_ms:>10.2f}ms  "
                f"{result.avg_similarity_score:>13.4f}  "
                f"{result.high_similarity_coverage:>13.2f}%"
            )
        
        logger.info("")
        
        # Find best method
        best = max(all_results, key=lambda x: x.avg_similarity_score)
        logger.info(f"🏆 Best Method: {best.method_name} (Method {best.method_id})")
        logger.info(f"   Average Similarity: {best.avg_similarity_score:.4f}")
        logger.info(f"   Search Time: {best.avg_search_time_ms:.2f}ms")
        logger.info(f"   High Similarity Coverage: {best.high_similarity_coverage:.2f}%")
    
    def generate_report(self, output_file: str = "reports/embedding_evaluation_report.json"):
        """Generate evaluation report."""
        logger.info("=" * 80)
        logger.info("GENERATING EVALUATION REPORT")
        logger.info("=" * 80)
        
        # Get all results
        all_results = self.db.query(EmbeddingEvaluationResults).all()
        
        report = {
            "evaluation_date": datetime.now().isoformat(),
            "methods": [],
            "summary": {}
        }
        
        for result in all_results:
            method_data = {
                "method_id": result.method_id,
                "method_name": result.method_name,
                "metrics": {
                    "avg_search_time_ms": result.avg_search_time_ms,
                    "avg_similarity_score": result.avg_similarity_score,
                    "similarity_std": result.similarity_std,
                    "high_similarity_coverage": result.high_similarity_coverage,
                    "test_samples": result.test_samples
                }
            }
            report["methods"].append(method_data)
        
        # Summary
        if report["methods"]:
            best = max(report["methods"], key=lambda x: x["metrics"]["avg_similarity_score"])
            report["summary"] = {
                "best_method": best["method_name"],
                "best_method_id": best["method_id"],
                "best_avg_similarity": best["metrics"]["avg_similarity_score"]
            }
        
        # Save report
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Report saved to {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive embedding evaluation research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full evaluation with all methods
  python scripts/evaluate_embeddings_research.py \
    --jd-file data/processed/jd_processed.csv \
    --candidate-file data/processed/candidate_processed.csv
  
  # Quick test with sample
  python scripts/evaluate_embeddings_research.py \
    --jd-file data/processed/jd_processed.csv \
    --candidate-file data/processed/candidate_processed.csv \
    --sample-size 1000 \
    --test-samples 50
        """
    )
    
    parser.add_argument("--jd-file", type=str, help="Path to JD processed dataset")
    parser.add_argument("--candidate-file", type=str, help="Path to candidate processed dataset")
    parser.add_argument("--sample-size", type=int, help="Sample size for embedding generation")
    parser.add_argument("--test-samples", type=int, default=100, help="Number of test samples for evaluation")
    parser.add_argument("--skip-generation", action="store_true", help="Skip embedding generation (use existing)")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation (only generate)")
    
    args = parser.parse_args()
    
    db: Session = SessionLocal()
    try:
        evaluator = EmbeddingEvaluator(db)
        
        # Initialize methods
        evaluator.initialize_methods()
        
        # Create tables
        evaluator.create_evaluation_tables()
        
        # Generate embeddings
        if not args.skip_generation:
            evaluator.generate_and_save_embeddings(
                jd_file=args.jd_file,
                candidate_file=args.candidate_file,
                sample_size=args.sample_size
            )
        
        # Verify
        evaluator.verify_embeddings_saved()
        
        # Evaluate
        if not args.skip_evaluation:
            evaluator.evaluate_methods(test_samples=args.test_samples)
            evaluator.generate_report()
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ EVALUATION COMPLETE")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error in evaluation: {e}", exc_info=True)
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()

