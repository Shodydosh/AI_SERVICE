"""Comprehensive system evaluation script."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import time
from datetime import datetime
from typing import Dict, List
from sqlalchemy.orm import Session
from sqlalchemy import text

from src.database.connection import SessionLocal
from src.database.models import JobDescriptionEmbedding, CandidateEmbedding, ProcessedCandidateRecommendation
from src.database.evaluation_models import EmbeddingEvaluationJD, EmbeddingEvaluationCandidate
from src.vector_search.faiss_manager import FAISSIndexManager
from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystemEvaluator:
    """Comprehensive system evaluator."""
    
    def __init__(self):
        self.results = {
            "database": {},
            "embeddings": {},
            "faiss": {},
            "precomputed": {},
            "evaluation_embeddings": {},
            "summary": {}
        }
    
    def evaluate_database(self, db: Session) -> Dict:
        """Evaluate database connection and data."""
        logger.info("=" * 80)
        logger.info("1. DATABASE EVALUATION")
        logger.info("=" * 80)
        
        results = {
            "connection": False,
            "tables": {},
            "data_counts": {},
            "health": "unknown"
        }
        
        try:
            # Test connection
            db.execute(text("SELECT 1"))
            results["connection"] = True
            logger.info("✓ Database connection: OK")
            
            # Check main tables
            jd_count = db.query(JobDescriptionEmbedding).count()
            candidate_count = db.query(CandidateEmbedding).count()
            precomputed_count = db.query(ProcessedCandidateRecommendation).count()
            
            results["data_counts"] = {
                "job_descriptions": jd_count,
                "candidates": candidate_count,
                "precomputed_recommendations": precomputed_count
            }
            
            logger.info(f"  Job descriptions: {jd_count}")
            logger.info(f"  Candidates: {candidate_count}")
            logger.info(f"  Pre-computed recommendations: {precomputed_count}")
            
            # Check evaluation tables
            eval_jd_count = db.query(EmbeddingEvaluationJD).count()
            eval_candidate_count = db.query(EmbeddingEvaluationCandidate).count()
            
            results["data_counts"]["evaluation_jd"] = eval_jd_count
            results["data_counts"]["evaluation_candidates"] = eval_candidate_count
            
            logger.info(f"  Evaluation JD embeddings: {eval_jd_count}")
            logger.info(f"  Evaluation Candidate embeddings: {eval_candidate_count}")
            
            # Health check
            if jd_count > 0 and candidate_count > 0:
                results["health"] = "healthy"
            elif jd_count > 0:
                results["health"] = "partial"
            else:
                results["health"] = "empty"
            
            logger.info(f"✓ Database health: {results['health']}")
            
        except Exception as e:
            logger.error(f"✗ Database error: {e}")
            results["health"] = "error"
        
        return results
    
    def evaluate_embeddings(self, db: Session) -> Dict:
        """Evaluate embedding quality."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("2. EMBEDDING QUALITY EVALUATION")
        logger.info("=" * 80)
        
        results = {
            "jd_embeddings": {},
            "candidate_embeddings": {},
            "dimension_check": False
        }
        
        try:
            # Check JD embeddings
            jd_sample = db.query(JobDescriptionEmbedding).first()
            if jd_sample:
                dim = len(jd_sample.embedding)
                results["jd_embeddings"] = {
                    "dimension": dim,
                    "non_zero_count": sum(1 for x in jd_sample.embedding if x != 0),
                    "sample_id": jd_sample.job_id
                }
                logger.info(f"✓ JD embedding dimension: {dim}")
                logger.info(f"  Non-zero values: {results['jd_embeddings']['non_zero_count']}/{dim}")
                
                if dim == settings.EMBEDDING_DIMENSION:
                    results["dimension_check"] = True
            
            # Check candidate embeddings
            candidate_sample = db.query(CandidateEmbedding).first()
            if candidate_sample:
                dim = len(candidate_sample.embedding)
                results["candidate_embeddings"] = {
                    "dimension": dim,
                    "non_zero_count": sum(1 for x in candidate_sample.embedding if x != 0),
                    "sample_id": candidate_sample.candidate_id
                }
                logger.info(f"✓ Candidate embedding dimension: {dim}")
                logger.info(f"  Non-zero values: {results['candidate_embeddings']['non_zero_count']}/{dim}")
            
        except Exception as e:
            logger.error(f"✗ Embedding evaluation error: {e}")
        
        return results
    
    def evaluate_faiss(self) -> Dict:
        """Evaluate FAISS indices."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("3. FAISS INDICES EVALUATION")
        logger.info("=" * 80)
        
        results = {
            "jd_index": {"exists": False, "loaded": False},
            "candidate_index": {"exists": False, "loaded": False}
        }
        
        try:
            faiss_manager = FAISSIndexManager(
                dimension=settings.EMBEDDING_DIMENSION,
                index_type="HNSW",
                normalize=True
            )
            
            # Check JD index
            jd_index_path = Path("indices/jd_index.faiss")
            if jd_index_path.exists():
                results["jd_index"]["exists"] = True
                logger.info("✓ JD FAISS index file exists")
                
                try:
                    faiss_manager.load_index(str(jd_index_path), dataset_type='jd')
                    stats = faiss_manager.get_index_stats('jd')
                    results["jd_index"]["loaded"] = True
                    results["jd_index"]["stats"] = stats
                    logger.info(f"  Index size: {stats.get('size', 0)} vectors")
                except Exception as e:
                    logger.warning(f"  Could not load JD index: {e}")
            else:
                logger.warning("⚠ JD FAISS index file not found")
            
            # Check candidate index
            candidate_index_path = Path("indices/candidate_index.faiss")
            if candidate_index_path.exists():
                results["candidate_index"]["exists"] = True
                logger.info("✓ Candidate FAISS index file exists")
                
                try:
                    faiss_manager.load_index(str(candidate_index_path), dataset_type='candidate')
                    stats = faiss_manager.get_index_stats('candidate')
                    results["candidate_index"]["loaded"] = True
                    results["candidate_index"]["stats"] = stats
                    logger.info(f"  Index size: {stats.get('size', 0)} vectors")
                except Exception as e:
                    logger.warning(f"  Could not load candidate index: {e}")
            else:
                logger.warning("⚠ Candidate FAISS index file not found")
            
        except Exception as e:
            logger.error(f"✗ FAISS evaluation error: {e}")
        
        return results
    
    def evaluate_precomputed(self, db: Session) -> Dict:
        """Evaluate pre-computed recommendations."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("4. PRE-COMPUTED RECOMMENDATIONS EVALUATION")
        logger.info("=" * 80)
        
        results = {
            "total_recommendations": 0,
            "unique_candidates": 0,
            "avg_recommendations_per_candidate": 0,
            "sample_quality": {}
        }
        
        try:
            total = db.query(ProcessedCandidateRecommendation).count()
            results["total_recommendations"] = total
            
            # Count unique candidates
            unique_candidates = db.query(ProcessedCandidateRecommendation.candidate_id).distinct().count()
            results["unique_candidates"] = unique_candidates
            
            if unique_candidates > 0:
                results["avg_recommendations_per_candidate"] = total / unique_candidates
            
            logger.info(f"✓ Total recommendations: {total}")
            logger.info(f"✓ Unique candidates: {unique_candidates}")
            logger.info(f"✓ Avg recommendations per candidate: {results['avg_recommendations_per_candidate']:.2f}")
            
            # Check sample quality
            if unique_candidates > 0:
                sample = db.query(ProcessedCandidateRecommendation).first()
                if sample:
                    results["sample_quality"] = {
                        "has_similarity_score": sample.similarity_score is not None,
                        "similarity_score": sample.similarity_score,
                        "has_rank": sample.rank is not None
                    }
                    logger.info(f"  Sample similarity score: {sample.similarity_score}")
            
        except Exception as e:
            logger.error(f"✗ Pre-computed evaluation error: {e}")
        
        return results
    
    def evaluate_evaluation_embeddings(self, db: Session) -> Dict:
        """Evaluate research embedding methods."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("5. RESEARCH EMBEDDING METHODS EVALUATION")
        logger.info("=" * 80)
        
        results = {
            "methods": {},
            "total_jd_embeddings": 0,
            "total_candidate_embeddings": 0
        }
        
        try:
            for method_id in range(1, 6):
                jd_count = db.query(EmbeddingEvaluationJD).filter(
                    EmbeddingEvaluationJD.method_id == method_id
                ).count()
                
                candidate_count = db.query(EmbeddingEvaluationCandidate).filter(
                    EmbeddingEvaluationCandidate.method_id == method_id
                ).count()
                
                if jd_count > 0 or candidate_count > 0:
                    sample = db.query(EmbeddingEvaluationJD).filter(
                        EmbeddingEvaluationJD.method_id == method_id
                    ).first()
                    
                    method_name = sample.method_name if sample else f"Method_{method_id}"
                    
                    results["methods"][method_id] = {
                        "name": method_name,
                        "jd_count": jd_count,
                        "candidate_count": candidate_count
                    }
                    
                    results["total_jd_embeddings"] += jd_count
                    results["total_candidate_embeddings"] += candidate_count
                    
                    logger.info(f"✓ Method {method_id} ({method_name}):")
                    logger.info(f"  JD embeddings: {jd_count}")
                    logger.info(f"  Candidate embeddings: {candidate_count}")
            
            logger.info(f"\nTotal evaluation JD embeddings: {results['total_jd_embeddings']}")
            logger.info(f"Total evaluation candidate embeddings: {results['total_candidate_embeddings']}")
            
        except Exception as e:
            logger.error(f"✗ Evaluation embeddings error: {e}")
        
        return results
    
    def generate_summary(self) -> Dict:
        """Generate evaluation summary."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 80)
        
        summary = {
            "overall_status": "unknown",
            "components": {},
            "recommendations": []
        }
        
        # Database
        db_status = self.results["database"].get("health", "unknown")
        summary["components"]["database"] = db_status
        
        # Embeddings
        has_jd = self.results["database"]["data_counts"].get("job_descriptions", 0) > 0
        has_candidates = self.results["database"]["data_counts"].get("candidates", 0) > 0
        summary["components"]["embeddings"] = "complete" if (has_jd and has_candidates) else "partial"
        
        # FAISS
        jd_index_ok = self.results["faiss"]["jd_index"].get("loaded", False)
        candidate_index_ok = self.results["faiss"]["candidate_index"].get("loaded", False)
        summary["components"]["faiss"] = "complete" if (jd_index_ok and candidate_index_ok) else "partial"
        
        # Pre-computed
        has_precomputed = self.results["precomputed"]["total_recommendations"] > 0
        summary["components"]["precomputed"] = "available" if has_precomputed else "missing"
        
        # Overall status
        if db_status == "healthy" and has_jd and has_candidates and jd_index_ok:
            summary["overall_status"] = "operational"
        elif db_status == "partial" or (has_jd and not has_candidates):
            summary["overall_status"] = "partial"
        else:
            summary["overall_status"] = "needs_setup"
        
        # Recommendations
        if not has_candidates:
            summary["recommendations"].append("Generate candidate embeddings")
        
        if not candidate_index_ok:
            summary["recommendations"].append("Build candidate FAISS index")
        
        if not has_precomputed:
            summary["recommendations"].append("Run pre-computation for recommendations")
        
        # Print detailed summary (using ASCII-safe characters)
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        
        # Overall status
        status_icon = {
            "operational": "[OK]",
            "partial": "[!]",
            "needs_setup": "[X]"
        }
        icon = status_icon.get(summary["overall_status"], "[?]")
        print(f"\n{icon} Overall Status: {summary['overall_status'].upper()}")
        
        # Component status
        print("\nComponent Status:")
        print("-" * 80)
        for component, status in summary["components"].items():
            if status in ["healthy", "complete", "available"]:
                icon = "[OK]"
            elif status == "partial":
                icon = "[!]"
            else:
                icon = "[X]"
            print(f"  {icon} {component.upper():<20} : {status}")
        
        # Data counts
        print("\nData Statistics:")
        print("-" * 80)
        counts = self.results["database"]["data_counts"]
        print(f"  Job Descriptions          : {counts.get('job_descriptions', 0):>8,}")
        print(f"  Candidates                : {counts.get('candidates', 0):>8,}")
        print(f"  Pre-computed Recommendations: {counts.get('precomputed_recommendations', 0):>8,}")
        print(f"  Evaluation JD Embeddings  : {counts.get('evaluation_jd', 0):>8,}")
        print(f"  Evaluation Candidate Embs : {counts.get('evaluation_candidates', 0):>8,}")
        
        # Embedding quality
        if self.results["embeddings"].get("jd_embeddings"):
            jd_emb = self.results["embeddings"]["jd_embeddings"]
            print("\nEmbedding Quality:")
            print("-" * 80)
            print(f"  JD Embedding Dimension   : {jd_emb.get('dimension', 0)}")
            print(f"  Non-zero Values          : {jd_emb.get('non_zero_count', 0)}/{jd_emb.get('dimension', 0)}")
            print(f"  Quality                  : {'GOOD' if jd_emb.get('non_zero_count', 0) == jd_emb.get('dimension', 0) else 'ISSUES'}")
        
        # FAISS status
        print("\nFAISS Indices:")
        print("-" * 80)
        jd_exists = self.results["faiss"]["jd_index"].get("exists", False)
        jd_loaded = self.results["faiss"]["jd_index"].get("loaded", False)
        cand_exists = self.results["faiss"]["candidate_index"].get("exists", False)
        cand_loaded = self.results["faiss"]["candidate_index"].get("loaded", False)
        
        jd_icon = "[OK]" if jd_loaded else ("[!]" if jd_exists else "[X]")
        cand_icon = "[OK]" if cand_loaded else ("[!]" if cand_exists else "[X]")
        
        jd_status = "Loaded" if jd_loaded else ("Exists" if jd_exists else "Missing")
        cand_status = "Loaded" if cand_loaded else ("Exists" if cand_exists else "Missing")
        
        print(f"  {jd_icon} JD Index       : {jd_status}")
        print(f"  {cand_icon} Candidate Index: {cand_status}")
        
        # Research methods
        if self.results["evaluation_embeddings"].get("methods"):
            print("\nResearch Embedding Methods:")
            print("-" * 80)
            methods = self.results["evaluation_embeddings"]["methods"]
            print(f"  {'Method':<25} {'JD Embs':<12} {'Candidate Embs':<15}")
            print("  " + "-" * 52)
            for method_id in sorted(methods.keys(), key=int):
                method = methods[method_id]
                jd_count = method.get("jd_count", 0)
                cand_count = method.get("candidate_count", 0)
                jd_status_icon = "[OK]" if jd_count > 0 else "[X]"
                cand_status_icon = "[OK]" if cand_count > 0 else "[X]"
                print(f"  {method.get('name', 'Unknown'):<25} {jd_status_icon} {jd_count:>6}     {cand_status_icon} {cand_count:>6}")
        
        # Recommendations
        if summary["recommendations"]:
            print("\nRecommendations:")
            print("-" * 80)
            for i, rec in enumerate(summary["recommendations"], 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "=" * 80)
        
        return summary
    
    def run_evaluation(self):
        """Run complete system evaluation."""
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE SYSTEM EVALUATION")
        logger.info("=" * 80)
        logger.info(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("")
        
        db: Session = SessionLocal()
        try:
            # 1. Database
            self.results["database"] = self.evaluate_database(db)
            
            # 2. Embeddings
            self.results["embeddings"] = self.evaluate_embeddings(db)
            
            # 3. FAISS
            self.results["faiss"] = self.evaluate_faiss()
            
            # 4. Pre-computed
            self.results["precomputed"] = self.evaluate_precomputed(db)
            
            # 5. Evaluation embeddings
            self.results["evaluation_embeddings"] = self.evaluate_evaluation_embeddings(db)
            
            # Summary
            self.results["summary"] = self.generate_summary()
            
            logger.info("")
            logger.info("=" * 80)
            logger.info("✓ EVALUATION COMPLETE")
            logger.info("=" * 80)
            
            # Also print to console for better visibility
            print("\n" + "=" * 80)
            print("[OK] EVALUATION COMPLETE - Check summary above")
            print("=" * 80)
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}", exc_info=True)
        finally:
            db.close()
        
        return self.results


def main():
    """Main function."""
    evaluator = SystemEvaluator()
    results = evaluator.run_evaluation()
    
    # Save results to file
    import json
    output_file = Path("reports/system_evaluation_report.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert results to JSON-serializable format
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    serializable_results = make_serializable(results)
    serializable_results["evaluation_date"] = datetime.now().isoformat()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✓ Report saved to {output_file}")


if __name__ == "__main__":
    main()

