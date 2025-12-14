"""Advanced Embedding Evaluation Script with Clustering and Adversarial Testing."""
import sys
import os
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sqlalchemy.orm import Session

from src.database.connection import SessionLocal
from src.database.multi_field_repository import MultiFieldEmbeddingRepository
from src.models.clustering_evaluator import ClusteringEvaluator
from src.models.adversarial_evaluator import AdversarialEvaluator
from src.utils.evaluation_visualizer import EvaluationVisualizer
from src.embeddings.job_tower_encoder import JobTowerEncoder
from src.embeddings.candidate_tower_encoder import CandidateTowerEncoder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_embeddings_from_database(
    db: Session,
    max_candidates: Optional[int] = None,
    max_jobs: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Load embeddings from database.
    
    Args:
        db: Database session
        max_candidates: Maximum number of candidates to load (None for all)
        max_jobs: Maximum number of jobs to load (None for all)
    
    Returns:
        Dictionary with candidate and job embeddings
    """
    logger.info("Loading embeddings from database...")
    
    repository = MultiFieldEmbeddingRepository(db)
    
    # Load candidates
    all_candidates = repository.get_all_candidate_multi_embeddings()
    if max_candidates and len(all_candidates) > max_candidates:
        np.random.seed(42)
        indices = np.random.choice(len(all_candidates), max_candidates, replace=False)
        candidates = [all_candidates[i] for i in indices]
    else:
        candidates = all_candidates
    
    # Load jobs
    all_jobs = repository.get_all_job_multi_embeddings()
    if max_jobs and len(all_jobs) > max_jobs:
        np.random.seed(42)
        indices = np.random.choice(len(all_jobs), max_jobs, replace=False)
        jobs = [all_jobs[i] for i in indices]
    else:
        jobs = all_jobs
    
    logger.info(f"Loaded {len(candidates)} candidates and {len(jobs)} jobs")
    
    # Extract embeddings
    candidate_embeddings = {
        'title': [],
        'skills': [],
        'experience': []
    }
    
    job_embeddings = {
        'title': [],
        'skills': [],
        'requirement': []
    }
    
    for candidate in candidates:
        if candidate.title_embedding:
            candidate_embeddings['title'].append(np.array(candidate.title_embedding))
        if candidate.skills_embedding:
            candidate_embeddings['skills'].append(np.array(candidate.skills_embedding))
        if candidate.experience_embedding:
            candidate_embeddings['experience'].append(np.array(candidate.experience_embedding))
    
    for job in jobs:
        if job.title_embedding:
            job_embeddings['title'].append(np.array(job.title_embedding))
        if job.skills_embedding:
            job_embeddings['skills'].append(np.array(job.skills_embedding))
        if job.requirement_embedding:
            job_embeddings['requirement'].append(np.array(job.requirement_embedding))
    
    # Convert to numpy arrays
    result = {}
    for field in ['title', 'skills', 'experience']:
        if candidate_embeddings[field]:
            result[f'candidate_{field}'] = np.array(candidate_embeddings[field])
        else:
            result[f'candidate_{field}'] = np.array([])
    
    for field in ['title', 'skills', 'requirement']:
        if job_embeddings[field]:
            result[f'job_{field}'] = np.array(job_embeddings[field])
        else:
            result[f'job_{field}'] = np.array([])
    
    return result


def run_clustering_evaluation(
    embeddings: Dict[str, np.ndarray],
    n_clusters_range: List[int] = [2, 5, 10, 20]
) -> Dict[str, any]:
    """
    Run clustering evaluation.
    
    Args:
        embeddings: Dictionary with embeddings
        n_clusters_range: List of k values to test
    
    Returns:
        Clustering evaluation results
    """
    logger.info("=" * 80)
    logger.info("RUNNING CLUSTERING EVALUATION")
    logger.info("=" * 80)
    
    evaluator = ClusteringEvaluator()
    
    # Prepare candidate embeddings
    candidate_embeddings_dict = {
        'title': embeddings.get('candidate_title', np.array([])),
        'skills': embeddings.get('candidate_skills', np.array([])),
        'experience': embeddings.get('candidate_experience', np.array([]))
    }
    
    # Prepare job embeddings
    job_embeddings_dict = {
        'title': embeddings.get('job_title', np.array([])),
        'skills': embeddings.get('job_skills', np.array([])),
        'requirement': embeddings.get('job_requirement', np.array([]))
    }
    
    # Run comprehensive evaluation
    results = evaluator.evaluate_cluster_consistency(
        candidate_embeddings_dict,
        job_embeddings_dict,
        n_clusters_range=n_clusters_range
    )
    
    # Print summary
    logger.info("\nClustering Evaluation Summary:")
    logger.info("-" * 80)
    
    if 'candidates' in results and 'combined' in results['candidates']:
        best_k = results['candidates']['combined'].get('best_k')
        best_silhouette = results['candidates']['combined'].get('best_silhouette', 0)
        logger.info(f"Candidates - Best k: {best_k}, Best Silhouette: {best_silhouette:.4f}")
    
    if 'jobs' in results and 'combined' in results['jobs']:
        best_k = results['jobs']['combined'].get('best_k')
        best_silhouette = results['jobs']['combined'].get('best_silhouette', 0)
        logger.info(f"Jobs - Best k: {best_k}, Best Silhouette: {best_silhouette:.4f}")
    
    if 'overlap' in results:
        overlap_score = results['overlap'].get('overlap_score', 0)
        logger.info(f"Cluster Overlap Score: {overlap_score:.4f}")
    
    return results


def run_adversarial_evaluation(
    db: Session,
    n_samples: int = 20,
    n_typos: int = 3,
    n_replacements: int = 3
) -> Dict[str, any]:
    """
    Run adversarial evaluation.
    
    Args:
        db: Database session
        n_samples: Number of samples to test
        n_typos: Number of typos to inject
        n_replacements: Number of synonym replacements
    
    Returns:
        Adversarial evaluation results
    """
    logger.info("=" * 80)
    logger.info("RUNNING ADVERSARIAL EVALUATION")
    logger.info("=" * 80)
    
    evaluator = AdversarialEvaluator()
    repository = MultiFieldEmbeddingRepository(db)
    
    # Initialize encoders
    job_encoder = JobTowerEncoder()
    candidate_encoder = CandidateTowerEncoder()
    
    # Load sample candidates and jobs
    all_candidates = repository.get_all_candidate_multi_embeddings()
    all_jobs = repository.get_all_job_multi_embeddings()
    
    if len(all_candidates) == 0 or len(all_jobs) == 0:
        logger.warning("No candidates or jobs found in database. Skipping adversarial evaluation.")
        return {}
    
    # Sample candidates and jobs
    np.random.seed(42)
    n_candidates = min(n_samples, len(all_candidates))
    n_jobs = min(n_samples, len(all_jobs))
    
    candidate_indices = np.random.choice(len(all_candidates), n_candidates, replace=False)
    job_indices = np.random.choice(len(all_jobs), n_jobs, replace=False)
    
    candidates = [all_candidates[i] for i in candidate_indices]
    jobs = [all_jobs[i] for i in job_indices]
    
    results = {
        'typo_injection': [],
        'synonym_replacement': [],
        'keyword_removal': [],
        'translation_roundtrip': []
    }
    
    # Test typo injection
    logger.info(f"\nTesting typo injection on {n_candidates} candidates...")
    for candidate in candidates[:n_candidates]:
        text = f"{candidate.title or ''} {candidate.skills or ''} {candidate.experience or ''}".strip()
        if text:
            def embeddings_func(txt):
                encoded = candidate_encoder.encode_candidate(
                    title=txt.split('|')[0] if '|' in txt else txt,
                    skills="",
                    experience=""
                )
                # Combine embeddings
                combined = (
                    np.array(encoded['title_embedding']) +
                    np.array(encoded['skills_embedding']) +
                    np.array(encoded['experience_embedding'])
                ) / 3.0
                return combined
            
            result = evaluator.test_typo_injection(text, embeddings_func, n_typos=n_typos)
            results['typo_injection'].append(result)
    
    # Test synonym replacement
    logger.info(f"\nTesting synonym replacement on {n_candidates} candidates...")
    for candidate in candidates[:n_candidates]:
        text = f"{candidate.title or ''} {candidate.skills or ''} {candidate.experience or ''}".strip()
        if text:
            def embeddings_func(txt):
                encoded = candidate_encoder.encode_candidate(
                    title=txt.split('|')[0] if '|' in txt else txt,
                    skills="",
                    experience=""
                )
                combined = (
                    np.array(encoded['title_embedding']) +
                    np.array(encoded['skills_embedding']) +
                    np.array(encoded['experience_embedding'])
                ) / 3.0
                return combined
            
            result = evaluator.test_synonym_replacement(text, embeddings_func, n_replacements=n_replacements)
            results['synonym_replacement'].append(result)
    
    # Test keyword removal
    logger.info(f"\nTesting keyword removal on {n_jobs} jobs...")
    common_keywords = ['python', 'java', 'javascript', 'react', 'sql', 'docker', 'aws']
    for job in jobs[:n_jobs]:
        text = f"{job.title or ''} {job.skills or ''} {job.requirement or ''}".strip()
        if text:
            def embeddings_func(txt):
                encoded = job_encoder.encode_job(
                    title=txt.split('|')[0] if '|' in txt else txt,
                    skills="",
                    requirement=""
                )
                combined = (
                    np.array(encoded['title_embedding']) +
                    np.array(encoded['skills_embedding']) +
                    np.array(encoded['requirement_embedding'])
                ) / 3.0
                return combined
            
            result = evaluator.test_keyword_removal(text, common_keywords, embeddings_func)
            results['keyword_removal'].append(result)
    
    # Test translation round-trip
    logger.info(f"\nTesting translation round-trip on {n_candidates} candidates...")
    for candidate in candidates[:n_candidates]:
        text = f"{candidate.title or ''} {candidate.skills or ''} {candidate.experience or ''}".strip()
        if text:
            def embeddings_func(txt):
                encoded = candidate_encoder.encode_candidate(
                    title=txt.split('|')[0] if '|' in txt else txt,
                    skills="",
                    experience=""
                )
                combined = (
                    np.array(encoded['title_embedding']) +
                    np.array(encoded['skills_embedding']) +
                    np.array(encoded['experience_embedding'])
                ) / 3.0
                return combined
            
            result = evaluator.test_translation_roundtrip(text, embeddings_func)
            results['translation_roundtrip'].append(result)
    
    # Compute overall metrics
    all_test_results = []
    for test_type, test_results in results.items():
        for result in test_results:
            if 'robustness_score' in result:
                all_test_results.append(result)
    
    overall_robustness = evaluator.compute_robustness_score(all_test_results)
    results['overall_robustness'] = overall_robustness
    
    # Print summary
    logger.info("\nAdversarial Evaluation Summary:")
    logger.info("-" * 80)
    logger.info(f"Overall Robustness: {overall_robustness.get('overall_robustness', 0):.4f}")
    logger.info(f"Average Similarity: {overall_robustness.get('avg_similarity', 0):.4f}")
    logger.info(f"Min Similarity: {overall_robustness.get('min_similarity', 0):.4f}")
    logger.info(f"Max Similarity: {overall_robustness.get('max_similarity', 0):.4f}")
    
    return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Advanced Embedding Evaluation')
    parser.add_argument('--max-candidates', type=int, default=None,
                       help='Maximum number of candidates to evaluate')
    parser.add_argument('--max-jobs', type=int, default=None,
                       help='Maximum number of jobs to evaluate')
    parser.add_argument('--n-clusters', type=int, nargs='+', default=[2, 5, 10, 20],
                       help='List of k values for clustering')
    parser.add_argument('--n-samples', type=int, default=20,
                       help='Number of samples for adversarial testing')
    parser.add_argument('--output-dir', type=str, default='reports/embedding_evaluation',
                       help='Output directory for results')
    parser.add_argument('--skip-clustering', action='store_true',
                       help='Skip clustering evaluation')
    parser.add_argument('--skip-adversarial', action='store_true',
                       help='Skip adversarial evaluation')
    parser.add_argument('--skip-visualization', action='store_true',
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize database
    db: Session = SessionLocal()
    
    try:
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'clustering': {},
            'adversarial': {},
            'preservation': {}
        }
        
        # Load embeddings
        embeddings = load_embeddings_from_database(
            db,
            max_candidates=args.max_candidates,
            max_jobs=args.max_jobs
        )
        
        # Run clustering evaluation
        if not args.skip_clustering:
            clustering_results = run_clustering_evaluation(
                embeddings,
                n_clusters_range=args.n_clusters
            )
            all_results['clustering'] = clustering_results
            
            # Save clustering results
            clustering_file = output_dir / "clustering_metrics.json"
            with open(clustering_file, 'w', encoding='utf-8') as f:
                json.dump(clustering_results, f, indent=2, default=str)
            logger.info(f"Clustering results saved to: {clustering_file}")
        
        # Run adversarial evaluation
        if not args.skip_adversarial:
            adversarial_results = run_adversarial_evaluation(
                db,
                n_samples=args.n_samples
            )
            all_results['adversarial'] = adversarial_results
            
            # Save adversarial results
            adversarial_file = output_dir / "adversarial_test_results.json"
            with open(adversarial_file, 'w', encoding='utf-8') as f:
                json.dump(adversarial_results, f, indent=2, default=str)
            logger.info(f"Adversarial results saved to: {adversarial_file}")
        
        # Generate visualizations
        if not args.skip_visualization:
            logger.info("=" * 80)
            logger.info("GENERATING VISUALIZATIONS")
            logger.info("=" * 80)
            
            visualizer = EvaluationVisualizer(
                output_dir=str(output_dir / "visualizations")
            )
            
            # Plot clustering metrics
            if 'clustering' in all_results and 'candidates' in all_results['clustering']:
                if 'combined' in all_results['clustering']['candidates']:
                    visualizer.plot_clustering_metrics(
                        all_results['clustering']['candidates']['combined']
                    )
            
            # Plot adversarial results
            if 'adversarial' in all_results:
                # Prepare adversarial results for plotting
                plot_data = {}
                for test_type, test_results in all_results['adversarial'].items():
                    if test_type != 'overall_robustness' and isinstance(test_results, list):
                        if test_results:
                            # Average results for this test type
                            avg_robustness = np.mean([
                                r.get('robustness_score', 0) for r in test_results
                            ])
                            avg_similarity = np.mean([
                                r.get('similarity_score', 0) for r in test_results
                            ])
                            plot_data[test_type] = {
                                'robustness_score': avg_robustness,
                                'similarity_score': avg_similarity
                            }
                
                plot_data['overall_robustness'] = all_results['adversarial'].get('overall_robustness', {})
                visualizer.plot_adversarial_results(plot_data)
            
            # Plot comprehensive evaluation
            if 'clustering' in all_results and 'adversarial' in all_results:
                preservation_results = all_results.get('preservation', {})
                visualizer.plot_comprehensive_evaluation(
                    all_results['clustering'],
                    all_results['adversarial'],
                    preservation_results
                )
        
        # Save all results
        all_results_file = output_dir / "evaluation_results.json"
        with open(all_results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"\nAll results saved to: {all_results_file}")
        
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()





