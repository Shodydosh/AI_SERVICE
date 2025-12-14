"""Index 25K candidates và 25K jobs, sau đó test với 5K candidates - Tối ưu với batch encoding."""
import sys
import logging
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.services.two_tower_matching_service import TwoTowerMatchingService
from src.database.two_tower_repository import TwoTowerRepository
from src.embeddings.candidate_tower_encoder import CandidateTowerEncoder, preprocess_candidate_experience
from src.embeddings.job_tower_encoder import (
    JobTowerEncoder, 
    preprocess_job_title, 
    preprocess_job_skills, 
    preprocess_job_requirements
)
from src.database.models import CandidateTwoTower, JobDescriptionTwoTower

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Setup matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

# Sample data templates
JOB_TITLES = [
    "Kế Toán Viên", "Senior Python Developer", "Digital Marketing Specialist",
    "Backend Developer", "Kế Toán Trưởng", "Machine Learning Engineer",
    "Cloud Engineer", "React Developer", "Senior Java Developer",
    "Mobile App Developer", "Data Scientist", "DevOps Engineer",
    "Frontend Developer", "Full-stack Developer", "QA Engineer",
    "Product Manager", "UI/UX Designer", "Business Analyst",
    "Sales Manager", "HR Specialist", "Content Writer"
]

SKILLS_POOL = {
    "Kế Toán": ["Excel", "Kế toán", "Báo cáo tài chính", "SAP", "QuickBooks"],
    "Python": ["Python", "Django", "Flask", "FastAPI", "PostgreSQL", "REST API"],
    "Java": ["Java", "Spring Boot", "Microservices", "MySQL", "Maven"],
    "Frontend": ["React", "TypeScript", "JavaScript", "HTML", "CSS"],
    "Marketing": ["SEO", "Google Ads", "Facebook Ads", "Content Marketing"],
    "DevOps": ["Docker", "Kubernetes", "AWS", "CI/CD", "Terraform"],
    "Mobile": ["React Native", "Flutter", "iOS", "Android"],
    "Data": ["Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch"]
}

COMPANIES = ["TechCorp", "StartupXYZ", "BigCompany", "InnovateLtd", "GlobalTech"]
LOCATIONS = ["Hà Nội", "TP. Hồ Chí Minh", "Đà Nẵng", "Cần Thơ"]


def generate_candidate_data(candidate_id: int) -> dict:
    """Generate sample candidate data."""
    title = random.choice(JOB_TITLES)
    
    if "Kế Toán" in title:
        skills_list = SKILLS_POOL["Kế Toán"]
    elif "Python" in title or "Data" in title:
        skills_list = SKILLS_POOL["Python"] + SKILLS_POOL["Data"]
    elif "Java" in title:
        skills_list = SKILLS_POOL["Java"]
    elif "Frontend" in title or "React" in title:
        skills_list = SKILLS_POOL["Frontend"]
    elif "Marketing" in title:
        skills_list = SKILLS_POOL["Marketing"]
    elif "DevOps" in title or "Cloud" in title:
        skills_list = SKILLS_POOL["DevOps"]
    elif "Mobile" in title:
        skills_list = SKILLS_POOL["Mobile"]
    else:
        skills_list = random.choice(list(SKILLS_POOL.values()))
    
    num_skills = random.randint(3, 6)
    skills = ", ".join(random.sample(skills_list, min(num_skills, len(skills_list))))
    
    years = random.randint(1, 8)
    experience = f"{years} năm kinh nghiệm {title.lower()}, thành thạo {skills.split(',')[0]}"
    
    return {
        "candidate_id": f"CAND{candidate_id:05d}",
        "name": f"Nguyễn Văn {chr(65 + (candidate_id % 26))}",
        "email": f"candidate{candidate_id}@example.com",
        "title": title,
        "skills": skills,
        "experience": experience
    }


def generate_job_data(job_id: int) -> dict:
    """Generate sample job data."""
    title = random.choice(JOB_TITLES)
    
    if "Kế Toán" in title:
        skills_list = SKILLS_POOL["Kế Toán"]
    elif "Python" in title or "Data" in title:
        skills_list = SKILLS_POOL["Python"] + SKILLS_POOL["Data"]
    elif "Java" in title:
        skills_list = SKILLS_POOL["Java"]
    elif "Frontend" in title or "React" in title:
        skills_list = SKILLS_POOL["Frontend"]
    elif "Marketing" in title:
        skills_list = SKILLS_POOL["Marketing"]
    elif "DevOps" in title or "Cloud" in title:
        skills_list = SKILLS_POOL["DevOps"]
    elif "Mobile" in title:
        skills_list = SKILLS_POOL["Mobile"]
    else:
        skills_list = random.choice(list(SKILLS_POOL.values()))
    
    num_skills = random.randint(3, 6)
    skills = ", ".join(random.sample(skills_list, min(num_skills, len(skills_list))))
    
    years = random.randint(2, 7)
    requirement = f"{years}+ năm kinh nghiệm {title.lower()}, thành thạo {skills.split(',')[0]}"
    
    return {
        "job_id": f"JOB{job_id:05d}",
        "title": title,
        "skills": skills,
        "requirement": requirement,
        "company": random.choice(COMPANIES),
        "location": random.choice(LOCATIONS)
    }


def batch_encode_candidates(encoder: CandidateTowerEncoder, candidates_data: List[dict], 
                           encoding_batch_size: int = 32) -> List[Dict]:
    """Batch encode candidates - tối ưu với batch processing."""
    all_embeddings = []
    
    # Collect all texts for each field
    titles = []
    skills = []
    experiences = []
    
    for cand in candidates_data:
        titles.append(preprocess_job_title(cand['title'] or ""))
        skills.append(preprocess_job_skills(cand['skills'] or ""))
        experiences.append(preprocess_candidate_experience(cand['experience'] or ""))
    
    # Batch encode each field
    title_embeddings = []
    skills_embeddings = []
    exp_embeddings = []
    
    # Encode titles
    for i in range(0, len(titles), encoding_batch_size):
        batch_titles = titles[i:i + encoding_batch_size]
        batch_embs = encoder.model.encode(
            batch_titles,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=encoding_batch_size
        )
        title_embeddings.extend(batch_embs.tolist())
    
    # Encode skills
    for i in range(0, len(skills), encoding_batch_size):
        batch_skills = skills[i:i + encoding_batch_size]
        batch_embs = encoder.model.encode(
            batch_skills,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=encoding_batch_size
        )
        skills_embeddings.extend(batch_embs.tolist())
    
    # Encode experiences
    for i in range(0, len(experiences), encoding_batch_size):
        batch_exps = experiences[i:i + encoding_batch_size]
        batch_embs = encoder.model.encode(
            batch_exps,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=encoding_batch_size
        )
        exp_embeddings.extend(batch_embs.tolist())
    
    # Combine into result format
    for i in range(len(candidates_data)):
        all_embeddings.append({
            'title_embedding': title_embeddings[i],
            'skills_embedding': skills_embeddings[i],
            'experience_embedding': exp_embeddings[i]
        })
    
    return all_embeddings


def batch_encode_jobs(encoder: JobTowerEncoder, jobs_data: List[dict], 
                     encoding_batch_size: int = 32) -> List[Dict]:
    """Batch encode jobs - tối ưu với batch processing."""
    all_embeddings = []
    
    # Collect all texts for each field
    titles = []
    skills = []
    requirements = []
    
    for job in jobs_data:
        titles.append(preprocess_job_title(job['title'] or ""))
        skills.append(preprocess_job_skills(job['skills'] or ""))
        requirements.append(preprocess_job_requirements(job['requirement'] or ""))
    
    # Batch encode each field
    title_embeddings = []
    skills_embeddings = []
    req_embeddings = []
    
    # Encode titles
    for i in range(0, len(titles), encoding_batch_size):
        batch_titles = titles[i:i + encoding_batch_size]
        batch_embs = encoder.model.encode(
            batch_titles,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=encoding_batch_size
        )
        title_embeddings.extend(batch_embs.tolist())
    
    # Encode skills
    for i in range(0, len(skills), encoding_batch_size):
        batch_skills = skills[i:i + encoding_batch_size]
        batch_embs = encoder.model.encode(
            batch_skills,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=encoding_batch_size
        )
        skills_embeddings.extend(batch_embs.tolist())
    
    # Encode requirements
    for i in range(0, len(requirements), encoding_batch_size):
        batch_reqs = requirements[i:i + encoding_batch_size]
        batch_embs = encoder.model.encode(
            batch_reqs,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=encoding_batch_size
        )
        req_embeddings.extend(batch_embs.tolist())
    
    # Combine into result format
    for i in range(len(jobs_data)):
        all_embeddings.append({
            'title_embedding': title_embeddings[i],
            'skills_embedding': skills_embeddings[i],
            'requirement_embedding': req_embeddings[i]
        })
    
    return all_embeddings


def index_data_optimized(db: Session, num_candidates: int = 25000, num_jobs: int = 25000, 
                        db_batch_size: int = 200, encoding_batch_size: int = 64):
    """Index candidates and jobs với batch encoding và batch commit."""
    print(f"\n{'='*100}")
    print(f"INDEXING {num_candidates} CANDIDATES VÀ {num_jobs} JOBS")
    print(f"DB Batch Size: {db_batch_size}, Encoding Batch Size: {encoding_batch_size}")
    print(f"{'='*100}\n")
    
    candidate_encoder = CandidateTowerEncoder()
    job_encoder = JobTowerEncoder()
    
    # Index candidates
    print(f"[Indexing] Indexing {num_candidates} candidates...")
    start_time = datetime.now()
    
    for batch_start in range(1, num_candidates + 1, db_batch_size):
        batch_end = min(batch_start + db_batch_size, num_candidates + 1)
        batch_ids = list(range(batch_start, batch_end))
        
        # Generate batch data
        batch_candidates = [generate_candidate_data(i) for i in batch_ids]
        
        # Batch encode
        batch_embeddings = batch_encode_candidates(candidate_encoder, batch_candidates, encoding_batch_size)
        
        # Insert to database
        for cand_data, embeddings in zip(batch_candidates, batch_embeddings):
            # Check if exists and update, otherwise create new
            existing = db.query(CandidateTwoTower).filter(
                CandidateTwoTower.candidate_id == cand_data['candidate_id']
            ).first()
            
            if existing:
                # Update existing
                existing.title = cand_data['title']
                existing.skills = cand_data['skills']
                existing.experience = cand_data['experience']
                existing.name = cand_data['name']
                existing.email = cand_data['email']
                existing.title_embedding = embeddings['title_embedding']
                existing.skills_embedding = embeddings['skills_embedding']
                existing.experience_embedding = embeddings['experience_embedding']
            else:
                # Create new
                candidate = CandidateTwoTower(
                    candidate_id=cand_data['candidate_id'],
                    title=cand_data['title'],
                    skills=cand_data['skills'],
                    experience=cand_data['experience'],
                    name=cand_data['name'],
                    email=cand_data['email'],
                    title_embedding=embeddings['title_embedding'],
                    skills_embedding=embeddings['skills_embedding'],
                    experience_embedding=embeddings['experience_embedding']
                )
                db.add(candidate)
        
        # Commit batch
        try:
            db.commit()
            elapsed = (datetime.now() - start_time).total_seconds()
            progress = (batch_end - 1) * 100 // num_candidates
            print(f"   ✓ Committed batch {batch_start}-{batch_end-1} ({progress}%) - {elapsed:.1f}s elapsed")
        except Exception as e:
            db.rollback()
            logger.error(f"Error committing batch {batch_start}-{batch_end}: {e}")
            raise
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"[OK] Indexed {num_candidates} candidates in {elapsed:.1f}s ({elapsed/num_candidates:.3f}s per candidate)\n")
    
    # Index jobs
    print(f"[Indexing] Indexing {num_jobs} jobs...")
    start_time = datetime.now()
    
    for batch_start in range(1, num_jobs + 1, db_batch_size):
        batch_end = min(batch_start + db_batch_size, num_jobs + 1)
        batch_ids = list(range(batch_start, batch_end))
        
        # Generate batch data
        batch_jobs = [generate_job_data(i) for i in batch_ids]
        
        # Batch encode
        batch_embeddings = batch_encode_jobs(job_encoder, batch_jobs, encoding_batch_size)
        
        # Insert to database
        for job_data, embeddings in zip(batch_jobs, batch_embeddings):
            # Check if exists and update, otherwise create new
            existing = db.query(JobDescriptionTwoTower).filter(
                JobDescriptionTwoTower.job_id == job_data['job_id']
            ).first()
            
            if existing:
                # Update existing
                existing.title = job_data['title']
                existing.skills = job_data['skills']
                existing.requirement = job_data['requirement']
                existing.company = job_data['company']
                existing.location = job_data['location']
                existing.title_embedding = embeddings['title_embedding']
                existing.skills_embedding = embeddings['skills_embedding']
                existing.requirement_embedding = embeddings['requirement_embedding']
            else:
                # Create new
                job = JobDescriptionTwoTower(
                    job_id=job_data['job_id'],
                    title=job_data['title'],
                    skills=job_data['skills'],
                    requirement=job_data['requirement'],
                    company=job_data['company'],
                    location=job_data['location'],
                    title_embedding=embeddings['title_embedding'],
                    skills_embedding=embeddings['skills_embedding'],
                    requirement_embedding=embeddings['requirement_embedding']
                )
                db.add(job)
        
        # Commit batch
        try:
            db.commit()
            elapsed = (datetime.now() - start_time).total_seconds()
            progress = (batch_end - 1) * 100 // num_jobs
            print(f"   [OK] Committed batch {batch_start}-{batch_end-1} ({progress}%) - {elapsed:.1f}s elapsed")
        except Exception as e:
            db.rollback()
            logger.error(f"Error committing batch {batch_start}-{batch_end}: {e}")
            raise
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"[OK] Indexed {num_jobs} jobs in {elapsed:.1f}s ({elapsed/num_jobs:.3f}s per job)\n")
    
    print(f"{'='*100}")
    print(f"[DONE] INDEXING HOAN TAT!")
    print(f"{'='*100}\n")


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


def test_and_evaluate(db: Session, num_tests: int = 5000, top_k: int = 10):
    """Test matching và đánh giá."""
    print(f"\n{'='*100}")
    print(f"TEST VA DANH GIA TWO-TOWER MATCHING")
    print(f"{'='*100}\n")
    
    service = TwoTowerMatchingService(db)
    repository = TwoTowerRepository(db)
    
    all_candidates = repository.get_all_candidates()
    all_jobs = repository.get_all_jobs()
    
    print(f"[STATUS] Database Status:")
    print(f"   Candidates: {len(all_candidates)}")
    print(f"   Jobs: {len(all_jobs)}")
    
    if len(all_candidates) == 0 or len(all_jobs) == 0:
        print("\n[WARNING] Khong co data trong database!")
        return None
    
    if len(all_candidates) < num_tests:
        num_tests = len(all_candidates)
    
    test_candidates = random.sample(all_candidates, num_tests)
    
    # Metrics storage
    all_metrics = []
    all_times = []
    score_distributions = []
    field_score_distributions = defaultdict(list)
    
    print(f"\n[Testing] Testing {num_tests} candidates against {len(all_jobs)} jobs...\n")
    
    for idx, candidate in enumerate(test_candidates, 1):
        if idx % 100 == 0 or idx == 1:
            elapsed = sum(all_times) if all_times else 0
            print(f"Progress: {idx}/{num_tests} ({idx*100//num_tests}%) - {elapsed:.1f}s elapsed")
        
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
    print(f"KET QUA DANH GIA")
    print(f"{'='*100}")
    print(f"Total candidates tested: {num_tests}")
    print(f"Total jobs in database: {len(all_jobs)}")
    print(f"\n[METRICS] METRICS:")
    print(f"  Precision@10:  {avg_metrics['precision@10']:.4f}")
    print(f"  Recall@10:     {avg_metrics['recall@10']:.4f}")
    print(f"  F1@10:         {avg_metrics['f1@10']:.4f}")
    print(f"  NDCG@10:       {avg_metrics['ndcg@10']:.4f}")
    print(f"\n[PERFORMANCE] PERFORMANCE:")
    print(f"  Average time per candidate: {avg_metrics['avg_time']:.3f}s")
    print(f"  Total time: {sum(all_times):.2f}s")
    if sum(all_times) > 0:
        print(f"  Throughput: {num_tests/sum(all_times):.2f} candidates/second")
    print(f"\n[STATISTICS] SCORE STATISTICS:")
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
    fig.suptitle('Two-Tower Matching Evaluation Metrics (25K Dataset)', fontsize=16, fontweight='bold')
    
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
    metrics_file = output_dir / f"two_tower_25k_metrics_{timestamp}.png"
    plt.savefig(metrics_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved metrics plot: {metrics_file}")
    plt.close()
    
    # 2. Score Distributions
    if score_distributions:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Score Distributions (25K Dataset)', fontsize=16, fontweight='bold')
        
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
        scores_file = output_dir / f"two_tower_25k_scores_{timestamp}.png"
        plt.savefig(scores_file, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved scores plot: {scores_file}")
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
    ax.set_title('Average Evaluation Metrics (25K Dataset)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    comparison_file = output_dir / f"two_tower_25k_comparison_{timestamp}.png"
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved comparison plot: {comparison_file}")
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
        ax.set_title('Field Similarity Comparison (25K Dataset)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, mean, std in zip(bars, field_means, field_stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.3f}±{std:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        field_file = output_dir / f"two_tower_25k_fields_{timestamp}.png"
        plt.savefig(field_file, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved field comparison plot: {field_file}")
        plt.close()
    
    print(f"\n[DONE] All visualizations saved to {output_dir}/")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Index 25K candidates và jobs, test với 5K candidates")
    parser.add_argument("--num-candidates", type=int, default=25000, help="Số lượng candidates để index")
    parser.add_argument("--num-jobs", type=int, default=25000, help="Số lượng jobs để index")
    parser.add_argument("--num-tests", type=int, default=5000, help="Số lượng candidates để test")
    parser.add_argument("--top-k", type=int, default=10, help="Top K jobs")
    parser.add_argument("--db-batch-size", type=int, default=200, help="Batch size cho database commit")
    parser.add_argument("--encoding-batch-size", type=int, default=64, help="Batch size cho encoding")
    parser.add_argument("--skip-indexing", action="store_true", help="Skip indexing, chỉ test")
    
    args = parser.parse_args()
    
    db: Session = None
    try:
        db = SessionLocal()
        
        if not args.skip_indexing:
            index_data_optimized(
                db, 
                num_candidates=args.num_candidates,
                num_jobs=args.num_jobs,
                db_batch_size=args.db_batch_size,
                encoding_batch_size=args.encoding_batch_size
            )
        
        metrics = test_and_evaluate(db, args.num_tests, args.top_k)
        
        if metrics:
            print(f"\n[DONE] Test va danh gia hoan tat!")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n[ERROR] ERROR: {e}")
    finally:
        if db:
            db.close()


if __name__ == '__main__':
    main()

