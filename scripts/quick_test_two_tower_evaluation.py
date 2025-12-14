"""Quick test Two-Tower với đánh giá và visualization - sử dụng data có sẵn."""
import sys
import logging
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.services.two_tower_matching_service import TwoTowerMatchingService
from src.database.two_tower_repository import TwoTowerRepository

# Setup logging
logging.basicConfig(level=logging.WARNING)  # Reduce logging
logger = logging.getLogger(__name__)

# Setup matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")


def calculate_ground_truth_score(candidate_title: str, job_title: str, 
                                  candidate_skills: str, job_skills: str) -> float:
    """Calculate ground truth matching score."""
    if not candidate_title or not job_title:
        return 0.0
    
    title_words = set(candidate_title.lower().split())
    job_title_words = set(job_title.lower().split())
    title_overlap = len(title_words & job_title_words) / max(len(title_words | job_title_words), 1)
    
    if not candidate_skills or not job_skills:
        return title_overlap
    
    candidate_skill_set = set([s.strip().lower() for s in candidate_skills.split(',')])
    job_skill_set = set([s.strip().lower() for s in job_skills.split(',')])
    skills_overlap = len(candidate_skill_set & job_skill_set) / max(len(job_skill_set), 1)
    
    score = 0.5 * title_overlap + 0.5 * skills_overlap
    return score


def evaluate_matching(results: List[Dict], candidate, all_jobs: List) -> Dict:
    """Evaluate matching quality."""
    # Calculate ground truth scores
    ground_truth_scores = []
    for job in all_jobs:
        gt_score = calculate_ground_truth_score(
            candidate.title or "",
            job.title or "",
            candidate.skills or "",
            job.skills or ""
        )
        ground_truth_scores.append((job.job_id, gt_score))
    
    # Sort by ground truth
    ground_truth_scores.sort(key=lambda x: x[1], reverse=True)
    top_gt_jobs = {job_id for job_id, _ in ground_truth_scores[:10]}
    
    # Get predicted top jobs
    predicted_jobs = {r['job_id'] for r in results[:10]}
    
    # Calculate metrics
    intersection = top_gt_jobs & predicted_jobs
    precision = len(intersection) / len(predicted_jobs) if predicted_jobs else 0
    recall = len(intersection) / len(top_gt_jobs) if top_gt_jobs else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # NDCG@10
    def dcg(scores, k=10):
        scores = scores[:k]
        return sum(score / np.log2(i + 2) for i, score in enumerate(scores))
    
    ideal_scores = [score for _, score in ground_truth_scores[:10]]
    ideal_dcg = dcg(ideal_scores) if ideal_scores else 1.0
    
    predicted_scores = []
    for result in results[:10]:
        job_id = result['job_id']
        for gt_job_id, gt_score in ground_truth_scores:
            if gt_job_id == job_id:
                predicted_scores.append(gt_score)
                break
        else:
            predicted_scores.append(0)
    
    predicted_dcg = dcg(predicted_scores)
    ndcg = predicted_dcg / ideal_dcg if ideal_dcg > 0 else 0
    
    return {
        'precision@10': precision,
        'recall@10': recall,
        'f1@10': f1,
        'ndcg@10': ndcg,
        'overlap_count': len(intersection)
    }


def test_and_evaluate(db: Session, num_tests: int = 50, top_k: int = 10):
    """Test matching và đánh giá."""
    print(f"\n{'='*100}")
    print(f"TEST VÀ ĐÁNH GIÁ TWO-TOWER MATCHING")
    print(f"{'='*100}\n")
    
    service = TwoTowerMatchingService(db)
    repository = service.repository
    
    all_candidates = repository.get_all_candidates()
    all_jobs = repository.get_all_jobs()
    
    print(f"📊 Database Status:")
    print(f"   Candidates: {len(all_candidates)}")
    print(f"   Jobs: {len(all_jobs)}")
    
    if len(all_candidates) == 0 or len(all_jobs) == 0:
        print("\n⚠️  Không có data trong database!")
        print("   Vui lòng chạy indexing trước:")
        print("   python scripts/test_two_tower_1000_with_evaluation.py --num-candidates 100 --num-jobs 100")
        return None
    
    if len(all_candidates) < num_tests:
        num_tests = len(all_candidates)
    
    test_candidates = random.sample(all_candidates, num_tests)
    
    # Metrics storage
    all_metrics = []
    all_times = []
    score_distributions = []
    field_score_distributions = defaultdict(list)
    
    print(f"\n🔄 Testing {num_tests} candidates against {len(all_jobs)} jobs...\n")
    
    for idx, candidate in enumerate(test_candidates, 1):
        if idx % 10 == 0 or idx == 1:
            print(f"Progress: {idx}/{num_tests} ({idx*100//num_tests}%)")
        
        start_time = datetime.now()
        results = service.find_jobs_for_candidate(
            candidate_id=candidate.candidate_id,
            top_k=top_k
        )
        elapsed = (datetime.now() - start_time).total_seconds()
        all_times.append(elapsed)
        
        # Evaluate
        metrics = evaluate_matching(results, candidate, all_jobs)
        all_metrics.append(metrics)
        
        # Collect scores
        for result in results:
            score_distributions.append(result['score'])
            field_scores = result.get('field_scores', {})
            field_score_distributions['title'].append(field_scores.get('title', 0))
            field_score_distributions['skills'].append(field_scores.get('skills', 0))
            field_score_distributions['experience'].append(field_scores.get('experience', 0))
    
    # Calculate average metrics
    avg_metrics = {
        'precision@10': np.mean([m['precision@10'] for m in all_metrics]),
        'recall@10': np.mean([m['recall@10'] for m in all_metrics]),
        'f1@10': np.mean([m['f1@10'] for m in all_metrics]),
        'ndcg@10': np.mean([m['ndcg@10'] for m in all_metrics]),
        'avg_time': np.mean(all_times),
        'total_tests': num_tests
    }
    
    # Print results
    print(f"\n{'='*100}")
    print(f"KẾT QUẢ ĐÁNH GIÁ")
    print(f"{'='*100}")
    print(f"Total candidates tested: {num_tests}")
    print(f"Total jobs in database: {len(all_jobs)}")
    print(f"\n📊 METRICS:")
    print(f"  Precision@10:  {avg_metrics['precision@10']:.4f}")
    print(f"  Recall@10:     {avg_metrics['recall@10']:.4f}")
    print(f"  F1@10:         {avg_metrics['f1@10']:.4f}")
    print(f"  NDCG@10:       {avg_metrics['ndcg@10']:.4f}")
    print(f"\n⏱️  PERFORMANCE:")
    print(f"  Average time per candidate: {avg_metrics['avg_time']:.3f}s")
    print(f"  Total time: {sum(all_times):.2f}s")
    if sum(all_times) > 0:
        print(f"  Throughput: {num_tests/sum(all_times):.2f} candidates/second")
    print(f"\n📈 SCORE STATISTICS:")
    if score_distributions:
        print(f"  Overall Score: Mean={np.mean(score_distributions):.3f}, Std={np.std(score_distributions):.3f}")
        print(f"  Title Similarity: Mean={np.mean(field_score_distributions['title']):.3f}")
        print(f"  Skills Similarity: Mean={np.mean(field_score_distributions['skills']):.3f}")
        print(f"  Experience Similarity: Mean={np.mean(field_score_distributions['experience']):.3f}")
    print(f"{'='*100}\n")
    
    # Create visualizations
    if len(all_metrics) > 0:
        create_visualizations(all_metrics, all_times, score_distributions, 
                             field_score_distributions, avg_metrics)
    
    return avg_metrics


def create_visualizations(all_metrics: List[Dict], all_times: List[float],
                          score_distributions: List[float],
                          field_score_distributions: Dict[str, List[float]],
                          avg_metrics: Dict):
    """Create visualization plots."""
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Metrics Distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Two-Tower Matching Evaluation Metrics', fontsize=16, fontweight='bold')
    
    # Precision@10
    axes[0, 0].hist([m['precision@10'] for m in all_metrics], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(avg_metrics['precision@10'], color='red', linestyle='--', 
                       label=f"Mean: {avg_metrics['precision@10']:.3f}")
    axes[0, 0].set_title('Precision@10 Distribution')
    axes[0, 0].set_xlabel('Precision@10')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Recall@10
    axes[0, 1].hist([m['recall@10'] for m in all_metrics], bins=20, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].axvline(avg_metrics['recall@10'], color='red', linestyle='--',
                       label=f"Mean: {avg_metrics['recall@10']:.3f}")
    axes[0, 1].set_title('Recall@10 Distribution')
    axes[0, 1].set_xlabel('Recall@10')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # NDCG@10
    axes[1, 0].hist([m['ndcg@10'] for m in all_metrics], bins=20, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].axvline(avg_metrics['ndcg@10'], color='red', linestyle='--',
                       label=f"Mean: {avg_metrics['ndcg@10']:.3f}")
    axes[1, 0].set_title('NDCG@10 Distribution')
    axes[1, 0].set_xlabel('NDCG@10')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Latency
    axes[1, 1].hist(all_times, bins=20, edgecolor='black', alpha=0.7, color='purple')
    axes[1, 1].axvline(avg_metrics['avg_time'], color='red', linestyle='--',
                       label=f"Mean: {avg_metrics['avg_time']:.3f}s")
    axes[1, 1].set_title('Matching Latency Distribution')
    axes[1, 1].set_xlabel('Time (seconds)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    metrics_file = output_dir / f"two_tower_metrics_{timestamp}.png"
    plt.savefig(metrics_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved metrics plot: {metrics_file}")
    plt.close()
    
    # 2. Score Distributions
    if score_distributions:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Score Distributions', fontsize=16, fontweight='bold')
        
        # Overall scores
        axes[0, 0].hist(score_distributions, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(np.mean(score_distributions), color='red', linestyle='--',
                           label=f"Mean: {np.mean(score_distributions):.3f}")
        axes[0, 0].set_title('Overall Score Distribution')
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Title scores
        if field_score_distributions['title']:
            axes[0, 1].hist(field_score_distributions['title'], bins=50, edgecolor='black', alpha=0.7, color='blue')
            axes[0, 1].axvline(np.mean(field_score_distributions['title']), color='red', linestyle='--',
                               label=f"Mean: {np.mean(field_score_distributions['title']):.3f}")
            axes[0, 1].set_title('Title Similarity Distribution')
            axes[0, 1].set_xlabel('Title Similarity')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Skills scores
        if field_score_distributions['skills']:
            axes[1, 0].hist(field_score_distributions['skills'], bins=50, edgecolor='black', alpha=0.7, color='green')
            axes[1, 0].axvline(np.mean(field_score_distributions['skills']), color='red', linestyle='--',
                               label=f"Mean: {np.mean(field_score_distributions['skills']):.3f}")
            axes[1, 0].set_title('Skills Similarity Distribution')
            axes[1, 0].set_xlabel('Skills Similarity')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Experience scores
        if field_score_distributions['experience']:
            axes[1, 1].hist(field_score_distributions['experience'], bins=50, edgecolor='black', alpha=0.7, color='orange')
            axes[1, 1].axvline(np.mean(field_score_distributions['experience']), color='red', linestyle='--',
                               label=f"Mean: {np.mean(field_score_distributions['experience']):.3f}")
            axes[1, 1].set_title('Experience Similarity Distribution')
            axes[1, 1].set_xlabel('Experience Similarity')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        scores_file = output_dir / f"two_tower_scores_{timestamp}.png"
        plt.savefig(scores_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved scores plot: {scores_file}")
        plt.close()
    
    # 3. Metrics Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_names = ['Precision@10', 'Recall@10', 'F1@10', 'NDCG@10']
    metrics_values = [
        avg_metrics['precision@10'],
        avg_metrics['recall@10'],
        avg_metrics['f1@10'],
        avg_metrics['ndcg@10']
    ]
    
    bars = ax.bar(metrics_names, metrics_values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'], alpha=0.7)
    ax.set_ylim([0, 1])
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Average Evaluation Metrics', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    comparison_file = output_dir / f"two_tower_comparison_{timestamp}.png"
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {comparison_file}")
    plt.close()
    
    # 4. Field Score Comparison
    if field_score_distributions['title']:
        fig, ax = plt.subplots(figsize=(10, 6))
        field_names = ['Title', 'Skills', 'Experience']
        field_means = [
            np.mean(field_score_distributions['title']),
            np.mean(field_score_distributions['skills']),
            np.mean(field_score_distributions['experience'])
        ]
        field_stds = [
            np.std(field_score_distributions['title']),
            np.std(field_score_distributions['skills']),
            np.std(field_score_distributions['experience'])
        ]
        
        bars = ax.bar(field_names, field_means, yerr=field_stds, 
                      color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.7, capsize=5)
        ax.set_ylim([0, 1])
        ax.set_ylabel('Average Similarity', fontsize=12)
        ax.set_title('Field Similarity Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, mean, std in zip(bars, field_means, field_stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.3f}±{std:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        field_file = output_dir / f"two_tower_fields_{timestamp}.png"
        plt.savefig(field_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved field comparison plot: {field_file}")
        plt.close()
    
    print(f"\n✅ All visualizations saved to {output_dir}/")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick test Two-Tower với đánh giá và visualization")
    parser.add_argument("--num-tests", type=int, default=50, help="Số lượng candidates để test")
    parser.add_argument("--top-k", type=int, default=10, help="Top K jobs")
    
    args = parser.parse_args()
    
    db: Session = None
    try:
        db = SessionLocal()
        metrics = test_and_evaluate(db, args.num_tests, args.top_k)
        
        if metrics:
            print(f"\n✅ Test và đánh giá hoàn tất!")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n❌ ERROR: {e}")
    finally:
        if db:
            db.close()


if __name__ == '__main__':
    main()


