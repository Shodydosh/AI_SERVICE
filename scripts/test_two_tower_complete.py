"""Complete test script cho Two-Tower Matching Service với sample data."""
import sys
import logging
from pathlib import Path

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

# Sample data - 10 candidates
SAMPLE_CANDIDATES = [
    {
        "candidate_id": "CAND001",
        "name": "Nguyễn Văn A",
        "email": "nguyenvana@example.com",
        "title": "Nhân Viên Kế Toán",
        "skills": "Excel, Kế toán, Báo cáo tài chính, SAP, QuickBooks",
        "experience": "5 năm kinh nghiệm làm kế toán tại công ty lớn, xử lý báo cáo tài chính hàng tháng, quản lý sổ sách kế toán"
    },
    {
        "candidate_id": "CAND002",
        "name": "Trần Thị B",
        "email": "tranthib@example.com",
        "title": "Lập Trình Viên Python",
        "skills": "Python, Django, Flask, PostgreSQL, REST API, Git",
        "experience": "3 năm kinh nghiệm phát triển web application với Python, làm việc với Django framework, xây dựng REST API"
    },
    {
        "candidate_id": "CAND003",
        "name": "Lê Văn C",
        "email": "levanc@example.com",
        "title": "Nhân Viên Marketing",
        "skills": "Digital Marketing, SEO, Google Ads, Facebook Ads, Content Writing",
        "experience": "4 năm kinh nghiệm marketing online, quản lý chiến dịch quảng cáo Google và Facebook, tối ưu SEO"
    },
    {
        "candidate_id": "CAND004",
        "name": "Phạm Thị D",
        "email": "phamthid@example.com",
        "title": "Data Scientist",
        "skills": "Python, Machine Learning, TensorFlow, PyTorch, SQL, Pandas",
        "experience": "2 năm kinh nghiệm làm Data Science, xây dựng models dự đoán, phân tích dữ liệu lớn với Python"
    },
    {
        "candidate_id": "CAND005",
        "name": "Hoàng Văn E",
        "email": "hoangvane@example.com",
        "title": "DevOps Engineer",
        "skills": "Docker, Kubernetes, AWS, CI/CD, Jenkins, Linux",
        "experience": "3 năm kinh nghiệm DevOps, setup CI/CD pipeline, quản lý infrastructure trên AWS và Kubernetes"
    },
    {
        "candidate_id": "CAND006",
        "name": "Vũ Thị F",
        "email": "vuthif@example.com",
        "title": "Frontend Developer",
        "skills": "React, TypeScript, JavaScript, HTML, CSS, Redux",
        "experience": "2 năm kinh nghiệm phát triển frontend với React, xây dựng responsive UI/UX, làm việc với TypeScript"
    },
    {
        "candidate_id": "CAND007",
        "name": "Đỗ Văn G",
        "email": "dovang@example.com",
        "title": "Java Developer",
        "skills": "Java, Spring Boot, Microservices, MySQL, Maven",
        "experience": "4 năm kinh nghiệm phát triển backend với Java Spring Boot, xây dựng microservices, RESTful APIs"
    },
    {
        "candidate_id": "CAND008",
        "name": "Bùi Thị H",
        "email": "buithih@example.com",
        "title": "QA Engineer",
        "skills": "Selenium, Cypress, JUnit, Test Automation, API Testing",
        "experience": "3 năm kinh nghiệm QA, viết test cases, automation testing với Selenium và Cypress"
    },
    {
        "candidate_id": "CAND009",
        "name": "Ngô Văn I",
        "email": "ngovani@example.com",
        "title": "Mobile Developer",
        "skills": "React Native, Flutter, iOS, Android, Swift, Kotlin",
        "experience": "2 năm kinh nghiệm phát triển mobile app với React Native và Flutter, publish apps lên App Store và Play Store"
    },
    {
        "candidate_id": "CAND010",
        "name": "Trịnh Thị K",
        "email": "trinhthik@example.com",
        "title": "UI/UX Designer",
        "skills": "Figma, Adobe XD, Sketch, User Research, Prototyping",
        "experience": "3 năm kinh nghiệm thiết kế UI/UX, tạo wireframes và prototypes, user research và testing"
    }
]

# Sample jobs - 10 jobs
SAMPLE_JOBS = [
    {
        "job_id": "JOB001",
        "title": "Kế Toán Viên",
        "skills": "Excel, Kế toán, Báo cáo tài chính, SAP",
        "requirement": "3-5 năm kinh nghiệm kế toán, thành thạo Excel và phần mềm kế toán",
        "company": "Công Ty ABC",
        "location": "Hà Nội"
    },
    {
        "job_id": "JOB002",
        "title": "Senior Python Developer",
        "skills": "Python, Django, FastAPI, PostgreSQL, REST API",
        "requirement": "3+ năm kinh nghiệm phát triển backend với Python, thành thạo Django hoặc FastAPI",
        "company": "Tech Corp",
        "location": "Hồ Chí Minh"
    },
    {
        "job_id": "JOB003",
        "title": "Digital Marketing Specialist",
        "skills": "Digital Marketing, SEO, Google Ads, Facebook Ads",
        "requirement": "2+ năm kinh nghiệm marketing online, thành thạo Google Ads và Facebook Ads",
        "company": "Marketing Agency",
        "location": "Hà Nội"
    },
    {
        "job_id": "JOB004",
        "title": "Backend Developer",
        "skills": "Java, Spring Boot, Microservices, PostgreSQL",
        "requirement": "2+ năm kinh nghiệm backend development với Java",
        "company": "Software Company",
        "location": "Đà Nẵng"
    },
    {
        "job_id": "JOB005",
        "title": "Kế Toán Trưởng",
        "skills": "Kế toán, Báo cáo tài chính, Quản lý team",
        "requirement": "5+ năm kinh nghiệm kế toán, có kinh nghiệm quản lý team",
        "company": "Big Company",
        "location": "Hồ Chí Minh"
    },
    {
        "job_id": "JOB006",
        "title": "Machine Learning Engineer",
        "skills": "Python, Machine Learning, TensorFlow, PyTorch, Deep Learning",
        "requirement": "2+ năm kinh nghiệm ML, xây dựng và deploy ML models, thành thạo TensorFlow hoặc PyTorch",
        "company": "AI Startup",
        "location": "Hồ Chí Minh"
    },
    {
        "job_id": "JOB007",
        "title": "Cloud Engineer",
        "skills": "AWS, Azure, Docker, Kubernetes, Terraform",
        "requirement": "3+ năm kinh nghiệm cloud infrastructure, thành thạo AWS hoặc Azure, CI/CD pipelines",
        "company": "Cloud Solutions",
        "location": "Hà Nội"
    },
    {
        "job_id": "JOB008",
        "title": "React Developer",
        "skills": "React, TypeScript, JavaScript, HTML, CSS",
        "requirement": "2+ năm kinh nghiệm React, TypeScript, xây dựng responsive web applications",
        "company": "Web Agency",
        "location": "Đà Nẵng"
    },
    {
        "job_id": "JOB009",
        "title": "Senior Java Developer",
        "skills": "Java, Spring Boot, Microservices, Kafka, Redis",
        "requirement": "4+ năm kinh nghiệm Java, Spring Boot, xây dựng distributed systems",
        "company": "Enterprise Tech",
        "location": "Hồ Chí Minh"
    },
    {
        "job_id": "JOB010",
        "title": "Mobile App Developer",
        "skills": "React Native, Flutter, iOS, Android",
        "requirement": "2+ năm kinh nghiệm mobile development với React Native hoặc Flutter",
        "company": "Mobile App Studio",
        "location": "Hà Nội"
    }
]


def print_header(title: str, width: int = 100):
    """Print formatted header."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_step(step_num: int, description: str):
    """Print step description."""
    print(f"\n[STEP {step_num}] {description}")
    print("-" * 100)


def explain_flow():
    """Giải thích luồng hoạt động."""
    print_header("🔄 LUỒNG HOẠT ĐỘNG TWO-TOWER MATCHING (3 EMBEDDINGS)")
    
    print("\n📊 KIẾN TRÚC:")
    print("   ┌─────────────────────────────────────────────────────────────┐")
    print("   │                    INPUT DATA                               │")
    print("   │  Candidate: [title, skills, experience]                     │")
    print("   │  Job:       [title, skills, requirement]                    │")
    print("   └─────────────────────────────────────────────────────────────┘")
    print("                            ↓")
    print("   ┌─────────────────────────────────────────────────────────────┐")
    print("   │              ENCODE THÀNH 3 EMBEDDINGS                      │")
    print("   │  Candidate:                                                 │")
    print("   │    - title_embedding (768-dim)                              │")
    print("   │    - skills_embedding (768-dim)                             │")
    print("   │    - experience_embedding (768-dim)                         │")
    print("   │  Job:                                                       │")
    print("   │    - title_embedding (768-dim)                              │")
    print("   │    - skills_embedding (768-dim)                             │")
    print("   │    - requirement_embedding (768-dim)                        │")
    print("   └─────────────────────────────────────────────────────────────┘")
    print("                            ↓")
    print("   ┌─────────────────────────────────────────────────────────────┐")
    print("   │          TÍNH SIMILARITY CHO TỪNG FIELD                     │")
    print("   │   1. title_sim = cosine(candidate.title, job.title)         │")
    print("   │   2. skills_sim = cosine(candidate.skills, job.skills)      │")
    print("   │   3. exp_sim = cosine(candidate.exp, job.requirement)       │")
    print("   └─────────────────────────────────────────────────────────────┘")
    print("                            ↓")
    print("   ┌─────────────────────────────────────────────────────────────┐")
    print("   │              KẾT HỢP SCORES                                 │")
    print("   │  Final Score = 0.4×title + 0.4×skills + 0.2×experience      │")
    print("   └─────────────────────────────────────────────────────────────┘")
    print("                            ↓")
    print("   ┌─────────────────────────────────────────────────────────────┐")
    print("   │                    TOP-K RESULTS                            │")
    print("   │  Sắp xếp theo Final Score và trả về top-k                   │")
    print("   └─────────────────────────────────────────────────────────────┘")
    
    print("\n\n📝 CHI TIẾT CÁC BƯỚC:")
    
    print_step(1, "KHỞI TẠO SERVICE")
    print("""
    1.1. Load CandidateTowerEncoder:
         - Model: VoVanPhuc/sup-SimCSE-VietNamese-phobert-base
         - Dimension: 768
         - Hỗ trợ Vietnamese tokenization
    
    1.2. Load JobTowerEncoder:
         - Model: VoVanPhuc/sup-SimCSE-VietNamese-phobert-base
         - Dimension: 768
    
    1.3. Set weights mặc định:
         - title: 0.4 (40%)
         - skills: 0.4 (40%)
         - experience: 0.2 (20%)
    """)
    
    print_step(2, "INDEX DATA (Lần đầu)")
    print("""
    2.1. Với mỗi candidate:
         - Encode title → title_embedding (768-dim)
         - Encode skills → skills_embedding (768-dim)
         - Encode experience → experience_embedding (768-dim)
         - Lưu vào database (candidate_two_tower table)
    
    2.2. Với mỗi job:
         - Encode title → title_embedding (768-dim)
         - Encode skills → skills_embedding (768-dim)
         - Encode requirement → requirement_embedding (768-dim)
         - Lưu vào database (job_description_two_tower table)
    """)
    
    print_step(3, "TÌM JOBS CHO CANDIDATE")
    print("""
    3.1. Lấy candidate từ database
    3.2. Encode candidate thành 3 embeddings (nếu chưa có trong DB)
    3.3. Lấy tất cả jobs từ database
    3.4. Với mỗi job:
         a. Load embeddings từ DB (nếu có) hoặc encode mới
         b. Tính similarity cho từng field:
            - title_sim = dot(candidate.title_emb, job.title_emb)
            - skills_sim = dot(candidate.skills_emb, job.skills_emb)
            - exp_sim = dot(candidate.exp_emb, job.req_emb)
         c. Tính combined score:
            score = 0.4 × title_sim + 0.4 × skills_sim + 0.2 × exp_sim
    3.5. Sắp xếp jobs theo score giảm dần
    3.6. Trả về top-k với field_scores breakdown
    """)


def index_sample_data(db: Session):
    """Index sample candidates and jobs vào database."""
    print_step(0, "INDEX SAMPLE DATA")
    
    repository = TwoTowerRepository(db)
    candidate_encoder = CandidateTowerEncoder()
    job_encoder = JobTowerEncoder()
    
    # Index candidates
    print("\n📝 Indexing candidates...")
    for cand_data in SAMPLE_CANDIDATES:
        print(f"   - Indexing candidate: {cand_data['candidate_id']} - {cand_data['name']}")
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
    print(f"✓ Indexed {len(SAMPLE_CANDIDATES)} candidates")
    
    # Index jobs
    print("\n📝 Indexing jobs...")
    for job_data in SAMPLE_JOBS:
        print(f"   - Indexing job: {job_data['job_id']} - {job_data['title']}")
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
    print(f"✓ Indexed {len(SAMPLE_JOBS)} jobs")


def test_matching(db: Session, candidate_id: str, top_k: int = 5):
    """Test matching với candidate cụ thể."""
    print_header(f"🎯 TEST MATCHING - Candidate: {candidate_id}")
    
    service = TwoTowerMatchingService(db)
    
    print_step(1, "Tìm jobs cho candidate")
    print(f"   Candidate ID: {candidate_id}")
    print(f"   Top K: {top_k}")
    
    # Get candidate info
    candidate = service.repository.get_candidate(candidate_id)
    if not candidate:
        print(f"\n❌ Candidate {candidate_id} not found!")
        return
    
    print(f"\n📋 Candidate Info:")
    print(f"   Name: {candidate.name}")
    print(f"   Title: {candidate.title}")
    print(f"   Skills: {candidate.skills[:100]}...")
    print(f"   Experience: {candidate.experience[:100]}...")
    
    print("\n🔄 Đang xử lý...")
    print("   [1/4] Encoding candidate thành 3 embeddings...")
    print("   [2/4] Loading jobs từ database...")
    print("   [3/4] Encoding jobs và tính similarity...")
    print("   [4/4] Tính combined scores và sắp xếp...")
    
    results = service.find_jobs_for_candidate(candidate_id, top_k=top_k)
    
    print(f"\n✅ Tìm thấy {len(results)} matching jobs\n")
    
    for idx, result in enumerate(results, 1):
        print(f"{'='*100}")
        print(f"🎯 RANK #{idx}")
        print(f"{'='*100}")
        print(f"Job ID: {result['job_id']}")
        print(f"Title: {result.get('title', 'N/A')}")
        print(f"Company: {result.get('company', 'N/A')}")
        print(f"Location: {result.get('location', 'N/A')}")
        print(f"\n📊 SCORES:")
        print(f"   ┌─────────────────────────────────────────────────────┐")
        print(f"   │ Overall Score: {result['score']:.4f}                    │")
        print(f"   ├─────────────────────────────────────────────────────┤")
        field_scores = result.get('field_scores', {})
        title_score = field_scores.get('title', 0)
        skills_score = field_scores.get('skills', 0)
        exp_score = field_scores.get('experience', 0)
        
        title_contrib = 0.4 * title_score
        skills_contrib = 0.4 * skills_score
        exp_contrib = 0.2 * exp_score
        
        print(f"   │ Title Match:      {title_score:.4f} (weight: 0.4) = {title_contrib:.4f}  │")
        print(f"   │ Skills Match:     {skills_score:.4f} (weight: 0.4) = {skills_contrib:.4f}  │")
        print(f"   │ Experience Match: {exp_score:.4f} (weight: 0.2) = {exp_contrib:.4f}  │")
        print(f"   └─────────────────────────────────────────────────────┘")
        print(f"\n💡 Breakdown:")
        print(f"   Final = 0.4×{title_score:.4f} + 0.4×{skills_score:.4f} + 0.2×{exp_score:.4f}")
        print(f"        = {title_contrib:.4f} + {skills_contrib:.4f} + {exp_contrib:.4f}")
        print(f"        = {result['score']:.4f}")
        print()


def main():
    """Main function."""
    print_header("🧪 COMPLETE TEST TWO-TOWER MATCHING SERVICE")
    
    # Explain flow
    explain_flow()
    
    # Test
    db: Session = None
    try:
        db = SessionLocal()
        
        print_header("🚀 BẮT ĐẦU TEST")
        
        # Index sample data
        index_sample_data(db)
        
        # Test matching for each candidate (top 10)
        for cand in SAMPLE_CANDIDATES:
            test_matching(db, cand['candidate_id'], top_k=10)
            print("\n")
        
        print_header("✅ TEST HOÀN TẤT")
        print("\n✓ Tất cả tests đã chạy thành công!")
        print("\n📊 Tóm tắt:")
        print(f"   - Đã index {len(SAMPLE_CANDIDATES)} candidates")
        print(f"   - Đã index {len(SAMPLE_JOBS)} jobs")
        print(f"   - Đã test matching cho {len(SAMPLE_CANDIDATES)} candidates")
        print(f"   - Top 10 recommendations cho mỗi candidate")
        
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
        print(f"\n❌ ERROR: {e}")
    finally:
        if db:
            db.close()


if __name__ == '__main__':
    main()

