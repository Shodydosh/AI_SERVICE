"""Test Two-Tower Matching với 1000 candidates và 1000 jobs."""
import sys
import logging
import random
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.services.two_tower_matching_service import TwoTowerMatchingService
from src.database.two_tower_repository import TwoTowerRepository
from src.embeddings.candidate_tower_encoder import CandidateTowerEncoder
from src.embeddings.job_tower_encoder import JobTowerEncoder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample data templates
JOB_TITLES = [
    "Kế Toán Viên", "Senior Python Developer", "Digital Marketing Specialist",
    "Backend Developer", "Kế Toán Trưởng", "Machine Learning Engineer",
    "Cloud Engineer", "React Developer", "Senior Java Developer",
    "Mobile App Developer", "Data Scientist", "DevOps Engineer",
    "Frontend Developer", "Full-stack Developer", "QA Engineer",
    "Product Manager", "UI/UX Designer", "Business Analyst",
    "Sales Manager", "HR Specialist", "Content Writer",
    "Graphic Designer", "Network Engineer", "Security Engineer",
    "Database Administrator", "System Administrator", "Project Manager"
]

SKILLS_POOL = {
    "Kế Toán": ["Excel", "Kế toán", "Báo cáo tài chính", "SAP", "QuickBooks", "Tally"],
    "Python": ["Python", "Django", "Flask", "FastAPI", "PostgreSQL", "REST API", "Git"],
    "Java": ["Java", "Spring Boot", "Microservices", "MySQL", "Maven", "Hibernate"],
    "Frontend": ["React", "TypeScript", "JavaScript", "HTML", "CSS", "Redux"],
    "Marketing": ["Digital Marketing", "SEO", "Google Ads", "Facebook Ads", "Content Writing"],
    "Data": ["Python", "Machine Learning", "TensorFlow", "PyTorch", "SQL", "Pandas"],
    "DevOps": ["Docker", "Kubernetes", "AWS", "CI/CD", "Jenkins", "Linux"],
    "Mobile": ["React Native", "Flutter", "iOS", "Android", "Swift", "Kotlin"]
}

EXPERIENCE_TEMPLATES = [
    "{years} năm kinh nghiệm làm {role} tại công ty lớn, xử lý {task}",
    "{years} năm kinh nghiệm phát triển {tech} application, làm việc với {framework}",
    "{years} năm kinh nghiệm {domain}, quản lý {responsibility}",
    "{years} năm kinh nghiệm {role}, thành thạo {skills}",
    "{years} năm kinh nghiệm tại {industry}, chuyên về {specialization}"
]

COMPANIES = [
    "Công Ty ABC", "Tech Corp", "Marketing Agency", "Software Company",
    "Big Company", "AI Startup", "Cloud Solutions", "Web Agency",
    "Enterprise Tech", "Mobile App Studio", "Data Solutions", "IT Services"
]

LOCATIONS = ["Hà Nội", "Hồ Chí Minh", "Đà Nẵng", "Cần Thơ", "Hải Phòng"]


def generate_candidate_data(candidate_id: int) -> dict:
    """Generate sample candidate data."""
    # Random job title
    title = random.choice(JOB_TITLES)
    
    # Get relevant skills based on title
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
    
    # Random skills (3-6 skills)
    num_skills = random.randint(3, 6)
    skills = ", ".join(random.sample(skills_list, min(num_skills, len(skills_list))))
    
    # Random experience
    years = random.randint(1, 8)
    template = random.choice(EXPERIENCE_TEMPLATES)
    experience = template.format(
        years=years,
        role=title.lower(),
        task="báo cáo tài chính" if "Kế Toán" in title else "phát triển ứng dụng",
        tech=random.choice(["web", "mobile", "backend"]),
        framework=random.choice(["Django", "Flask", "React", "Spring Boot"]),
        domain=random.choice(["kế toán", "phát triển phần mềm", "marketing"]),
        responsibility=random.choice(["team", "dự án", "chiến dịch"]),
        skills=skills,
        industry=random.choice(["công nghệ", "tài chính", "thương mại"]),
        specialization=random.choice(["backend", "frontend", "full-stack"])
    )
    
    return {
        "candidate_id": f"CAND{candidate_id:04d}",
        "name": f"Nguyễn Văn {chr(65 + (candidate_id % 26))}",
        "email": f"candidate{candidate_id}@example.com",
        "title": title,
        "skills": skills,
        "experience": experience
    }


def generate_job_data(job_id: int) -> dict:
    """Generate sample job data."""
    # Random job title
    title = random.choice(JOB_TITLES)
    
    # Get relevant skills
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
    
    # Random skills (3-6 skills)
    num_skills = random.randint(3, 6)
    skills = ", ".join(random.sample(skills_list, min(num_skills, len(skills_list))))
    
    # Random requirement
    years = random.randint(2, 7)
    requirement = f"{years}+ năm kinh nghiệm {title.lower()}, thành thạo {skills.split(',')[0]}"
    
    return {
        "job_id": f"JOB{job_id:04d}",
        "title": title,
        "skills": skills,
        "requirement": requirement,
        "company": random.choice(COMPANIES),
        "location": random.choice(LOCATIONS)
    }


def index_data(db: Session, num_candidates: int = 1000, num_jobs: int = 1000):
    """Index candidates and jobs vào database."""
    print(f"\n{'='*100}")
    print(f"INDEXING {num_candidates} CANDIDATES VÀ {num_jobs} JOBS")
    print(f"{'='*100}\n")
    
    repository = TwoTowerRepository(db)
    candidate_encoder = CandidateTowerEncoder()
    job_encoder = JobTowerEncoder()
    
    # Clear existing data (optional)
    print("Clearing existing data...")
    try:
        db.query(repository.__class__.__bases__[0].__subclasses__()[0]).delete()
        db.commit()
    except:
        pass
    
    # Index candidates
    print(f"\n📝 Indexing {num_candidates} candidates...")
    start_time = datetime.now()
    
    for i in range(1, num_candidates + 1):
        if i % 100 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"   Progress: {i}/{num_candidates} ({i*100//num_candidates}%) - {elapsed:.1f}s")
        
        cand_data = generate_candidate_data(i)
        embeddings = candidate_encoder.encode_candidate(
            title=cand_data['title'],
            skills=cand_data['skills'],
            experience=cand_data['experience']
        )
        repository.create_candidate(
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
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"✓ Indexed {num_candidates} candidates in {elapsed:.1f}s ({elapsed/num_candidates:.3f}s per candidate)")
    
    # Index jobs
    print(f"\n📝 Indexing {num_jobs} jobs...")
    start_time = datetime.now()
    
    for i in range(1, num_jobs + 1):
        if i % 100 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"   Progress: {i}/{num_jobs} ({i*100//num_jobs}%) - {elapsed:.1f}s")
        
        job_data = generate_job_data(i)
        embeddings = job_encoder.encode_job(
            title=job_data['title'],
            skills=job_data['skills'],
            requirements=job_data['requirement']
        )
        repository.create_job(
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
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"✓ Indexed {num_jobs} jobs in {elapsed:.1f}s ({elapsed/num_jobs:.3f}s per job)")
    
    print(f"\n✅ Total indexing time: {(datetime.now() - start_time).total_seconds():.1f}s")


def test_matching(db: Session, num_tests: int = 10, top_k: int = 10):
    """Test matching với một số candidates."""
    print(f"\n{'='*100}")
    print(f"TEST MATCHING - {num_tests} CANDIDATES, TOP {top_k} JOBS")
    print(f"{'='*100}\n")
    
    service = TwoTowerMatchingService(db)
    repository = service.repository
    
    # Get random candidates
    all_candidates = repository.get_all_candidates()
    if len(all_candidates) < num_tests:
        num_tests = len(all_candidates)
    
    test_candidates = random.sample(all_candidates, num_tests)
    
    total_time = 0
    
    for idx, candidate in enumerate(test_candidates, 1):
        print(f"\n{'='*100}")
        print(f"TEST {idx}/{num_tests}: Candidate {candidate.candidate_id}")
        print(f"{'='*100}")
        print(f"Name: {candidate.name}")
        print(f"Title: {candidate.title}")
        print(f"Skills: {candidate.skills[:80]}...")
        
        start_time = datetime.now()
        results = service.find_jobs_for_candidate(
            candidate_id=candidate.candidate_id,
            top_k=top_k
        )
        elapsed = (datetime.now() - start_time).total_seconds()
        total_time += elapsed
        
        print(f"\n⏱️  Matching time: {elapsed:.2f}s")
        print(f"\n📊 Top {len(results)} Jobs:")
        
        for rank, result in enumerate(results[:5], 1):  # Show top 5
            print(f"\n  [{rank}] {result['title']} (Score: {result['score']:.4f})")
            print(f"      Company: {result.get('company', 'N/A')}")
            print(f"      Location: {result.get('location', 'N/A')}")
            field_scores = result.get('field_scores', {})
            print(f"      Field Scores: Title={field_scores.get('title', 0):.3f}, "
                  f"Skills={field_scores.get('skills', 0):.3f}, "
                  f"Exp={field_scores.get('experience', 0):.3f}")
        
        if len(results) > 5:
            print(f"\n  ... and {len(results) - 5} more jobs")
    
    avg_time = total_time / num_tests
    print(f"\n{'='*100}")
    print(f"SUMMARY")
    print(f"{'='*100}")
    print(f"Total candidates tested: {num_tests}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per candidate: {avg_time:.2f}s")
    print(f"Jobs per candidate: {top_k}")
    print(f"{'='*100}\n")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Two-Tower Matching với 1000 records")
    parser.add_argument("--num-candidates", type=int, default=1000, help="Số lượng candidates")
    parser.add_argument("--num-jobs", type=int, default=1000, help="Số lượng jobs")
    parser.add_argument("--num-tests", type=int, default=10, help="Số lượng candidates để test")
    parser.add_argument("--top-k", type=int, default=10, help="Top K jobs")
    parser.add_argument("--skip-index", action="store_true", help="Skip indexing, chỉ test matching")
    
    args = parser.parse_args()
    
    db: Session = None
    try:
        db = SessionLocal()
        
        if not args.skip_index:
            # Index data
            index_data(db, args.num_candidates, args.num_jobs)
        else:
            print("Skipping indexing (--skip-index flag set)")
        
        # Test matching
        test_matching(db, args.num_tests, args.top_k)
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n❌ ERROR: {e}")
    finally:
        if db:
            db.close()


if __name__ == '__main__':
    main()


