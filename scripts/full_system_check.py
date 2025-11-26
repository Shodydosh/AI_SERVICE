"""Full system check: recheck all, precompute, and evaluate embeddings."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from sqlalchemy.orm import Session

from src.database.connection import SessionLocal
from src.services.precompute_service import PrecomputeService
from scripts.evaluate_system_comprehensive import SystemEvaluator
from scripts.evaluate_embeddings_research import EmbeddingEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run full system check."""
    print("\n" + "=" * 80)
    print("FULL SYSTEM CHECK")
    print("=" * 80)
    print("1. Rechecking all system components")
    print("2. Running pre-computation (if candidates available)")
    print("3. Evaluating embedding methods")
    print("=" * 80 + "\n")
    
    # Step 1: Recheck system
    print("\n" + "=" * 80)
    print("STEP 1: SYSTEM EVALUATION")
    print("=" * 80)
    evaluator = SystemEvaluator()
    system_results = evaluator.run_evaluation()
    
    # Step 2: Try pre-computation
    print("\n" + "=" * 80)
    print("STEP 2: PRE-COMPUTATION")
    print("=" * 80)
    
    db: Session = SessionLocal()
    try:
        candidate_count = system_results["database"]["data_counts"].get("candidates", 0)
        
        if candidate_count > 0:
            logger.info(f"Found {candidate_count} candidates, running pre-computation...")
            precompute_service = PrecomputeService(db)
            precompute_results = precompute_service.precompute_all_candidates(top_k=10)
            
            print("\nPre-computation Results:")
            print("-" * 80)
            print(f"  Total candidates: {precompute_results.get('total_candidates', 0)}")
            print(f"  Processed: {precompute_results.get('processed', 0)}")
            print(f"  Failed: {precompute_results.get('failed', 0)}")
            print(f"  Total recommendations: {precompute_results.get('total_recommendations', 0)}")
        else:
            logger.warning("No candidates found in database. Cannot run pre-computation.")
            logger.info("  Please generate candidate embeddings first.")
            print("\n[!] Pre-computation skipped: No candidates available")
            print("    Recommendation: Generate candidate embeddings first")
    
    except Exception as e:
        logger.error(f"Error in pre-computation: {e}", exc_info=True)
        print(f"\n[X] Pre-computation failed: {e}")
    finally:
        db.close()
    
    # Step 3: Evaluate embedding methods
    print("\n" + "=" * 80)
    print("STEP 3: EMBEDDING METHODS EVALUATION")
    print("=" * 80)
    
    db: Session = SessionLocal()
    try:
        embedding_evaluator = EmbeddingEvaluator(db)
        
        # Initialize methods
        embedding_evaluator.initialize_methods()
        
        # Verify embeddings
        embedding_evaluator.verify_embeddings_saved()
        
        # Evaluate methods (if we have both JD and candidate embeddings)
        eval_jd_count = system_results["evaluation_embeddings"]["total_jd_embeddings"]
        eval_candidate_count = system_results["evaluation_embeddings"]["total_candidate_embeddings"]
        
        if eval_jd_count > 0 and eval_candidate_count > 0:
            logger.info("Running full embedding evaluation...")
            embedding_evaluator.evaluate_methods(test_samples=min(100, eval_candidate_count))
            embedding_evaluator.generate_report()
        else:
            logger.warning("Incomplete embedding data for evaluation:")
            logger.info(f"  Evaluation JD embeddings: {eval_jd_count}")
            logger.info(f"  Evaluation Candidate embeddings: {eval_candidate_count}")
            print("\n[!] Embedding evaluation skipped: Incomplete data")
            print("    Recommendation: Generate candidate embeddings for all methods")
        
    except Exception as e:
        logger.error(f"Error in embedding evaluation: {e}", exc_info=True)
        print(f"\n[X] Embedding evaluation failed: {e}")
    finally:
        db.close()
    
    # Final summary
    print("\n" + "=" * 80)
    print("FULL SYSTEM CHECK COMPLETE")
    print("=" * 80)
    print("\nSummary:")
    print("-" * 80)
    print(f"  System Status: {system_results['summary']['overall_status'].upper()}")
    print(f"  Job Descriptions: {system_results['database']['data_counts'].get('job_descriptions', 0):,}")
    print(f"  Candidates: {system_results['database']['data_counts'].get('candidates', 0):,}")
    print(f"  Pre-computed Recommendations: {system_results['precomputed']['total_recommendations']:,}")
    print(f"  Evaluation JD Embeddings: {system_results['evaluation_embeddings']['total_jd_embeddings']:,}")
    print(f"  Evaluation Candidate Embeddings: {system_results['evaluation_embeddings']['total_candidate_embeddings']:,}")
    
    if system_results['summary']['recommendations']:
        print("\nRecommendations:")
        for i, rec in enumerate(system_results['summary']['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()


