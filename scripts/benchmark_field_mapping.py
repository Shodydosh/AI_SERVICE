"""Script to benchmark field-by-field embedding approach vs traditional approach."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import time
from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.services.field_mapping_matching_service import FieldMappingMatchingService
from src.services.matching_service import MatchingService
from src.database.repository import EmbeddingRepository
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def benchmark_approaches(
    candidate_file: str,
    jd_file: str,
    test_candidates: int = 10,
    top_k: int = 10
):
    """
    Benchmark field-by-field approach vs traditional approach.
    
    Args:
        candidate_file: Path to candidate CSV file
        jd_file: Path to JD CSV file
        test_candidates: Number of candidates to test
        top_k: Number of top jobs to retrieve
    """
    logger.info("=" * 80)
    logger.info("BENCHMARK: Field-by-Field vs Traditional Embedding")
    logger.info("=" * 80)
    logger.info("")
    
    # Load test candidates
    candidate_processor = CandidateProcessor()
    candidate_processor.load_from_csv(candidate_file)
    candidate_data = candidate_processor.data.head(test_candidates)
    
    logger.info(f"Testing with {len(candidate_data)} candidates")
    logger.info("")
    
    db: Session = SessionLocal()
    try:
        # Initialize services
        field_mapping_service = FieldMappingMatchingService(db)
        traditional_service = MatchingService(db, use_faiss=False, use_reranking=False)
        repository = EmbeddingRepository(db)
        
        results = []
        
        for idx, candidate_row in candidate_data.iterrows():
            candidate_id = candidate_row.get('candidate_id', f'candidate_{idx}')
            logger.info(f"Testing candidate: {candidate_id}")
            
            # Test Field-by-Field Approach
            start_time = time.time()
            try:
                field_results = field_mapping_service.find_top_jobs_for_candidate(
                    candidate_id=candidate_id,
                    top_k=top_k,
                    candidate_file=candidate_file,
                    jd_file=jd_file
                )
                field_time = time.time() - start_time
                field_top_score = field_results[0]['similarity_score'] if field_results else 0.0
                field_avg_score = sum(r['similarity_score'] for r in field_results) / len(field_results) if field_results else 0.0
            except Exception as e:
                logger.error(f"Field mapping error: {e}")
                field_time = 0
                field_top_score = 0.0
                field_avg_score = 0.0
                field_results = []
            
            # Test Traditional Approach
            start_time = time.time()
            try:
                traditional_results = traditional_service.find_jobs_for_candidate(
                    candidate_id=candidate_id,
                    top_k=top_k
                )
                traditional_time = time.time() - start_time
                traditional_top_score = traditional_results[0]['similarity_score'] if traditional_results else 0.0
                traditional_avg_score = sum(r['similarity_score'] for r in traditional_results) / len(traditional_results) if traditional_results else 0.0
            except Exception as e:
                logger.error(f"Traditional error: {e}")
                traditional_time = 0
                traditional_top_score = 0.0
                traditional_avg_score = 0.0
                traditional_results = []
            
            # Compare results
            results.append({
                'candidate_id': candidate_id,
                'field_mapping_time': field_time,
                'field_mapping_top_score': field_top_score,
                'field_mapping_avg_score': field_avg_score,
                'field_mapping_count': len(field_results),
                'traditional_time': traditional_time,
                'traditional_top_score': traditional_top_score,
                'traditional_avg_score': traditional_avg_score,
                'traditional_count': len(traditional_results),
                'time_diff': field_time - traditional_time,
                'top_score_diff': field_top_score - traditional_top_score,
                'avg_score_diff': field_avg_score - traditional_avg_score
            })
            
            logger.info(f"  Field Mapping: {field_time:.2f}s, Top: {field_top_score:.4f}, Avg: {field_avg_score:.4f}")
            logger.info(f"  Traditional: {traditional_time:.2f}s, Top: {traditional_top_score:.4f}, Avg: {traditional_avg_score:.4f}")
            logger.info("")
        
        # Summary statistics
        df_results = pd.DataFrame(results)
        
        logger.info("=" * 80)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Field-by-Field Approach:")
        logger.info(f"  Average time: {df_results['field_mapping_time'].mean():.2f}s")
        logger.info(f"  Average top score: {df_results['field_mapping_top_score'].mean():.4f}")
        logger.info(f"  Average score: {df_results['field_mapping_avg_score'].mean():.4f}")
        logger.info("")
        logger.info("Traditional Approach:")
        logger.info(f"  Average time: {df_results['traditional_time'].mean():.2f}s")
        logger.info(f"  Average top score: {df_results['traditional_top_score'].mean():.4f}")
        logger.info(f"  Average score: {df_results['traditional_avg_score'].mean():.4f}")
        logger.info("")
        logger.info("Comparison:")
        logger.info(f"  Time difference: {df_results['time_diff'].mean():.2f}s (Field - Traditional)")
        logger.info(f"  Top score difference: {df_results['top_score_diff'].mean():.4f} (Field - Traditional)")
        logger.info(f"  Avg score difference: {df_results['avg_score_diff'].mean():.4f} (Field - Traditional)")
        logger.info("")
        
        # Save results
        output_file = "benchmark_results.csv"
        df_results.to_csv(output_file, index=False)
        logger.info(f"Results saved to: {output_file}")
        logger.info("")
        
        return df_results
    
    except Exception as e:
        logger.error(f"Error in benchmark: {e}", exc_info=True)
        raise
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark field-by-field embedding approach vs traditional approach",
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
        "--test-candidates",
        type=int,
        default=10,
        help="Number of candidates to test (default: 10)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top jobs to retrieve (default: 10)"
    )
    
    args = parser.parse_args()
    
    try:
        benchmark_approaches(
            candidate_file=args.candidate_file,
            jd_file=args.jd_file,
            test_candidates=args.test_candidates,
            top_k=args.top_k
        )
        sys.exit(0)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    from src.data_processing.candidate_processor import CandidateProcessor
    main()

