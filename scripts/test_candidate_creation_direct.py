"""Test script để test candidate creation trực tiếp (không qua HTTP)."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.services.multi_field_embedding_service import MultiFieldEmbeddingService
from src.database.multi_field_repository import MultiFieldEmbeddingRepository
from src.services.multi_filter_matching_service import MultiFilterMatchingService
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 10 Sample candidates
SAMPLE_CANDIDATES = [
    {
        "candidate_id": "TEST_001",
        "title": "Nhân Viên Kế Toán",
        "skills": "Excel, Kế toán, Báo cáo tài chính, SAP, QuickBooks",
        "experience": "5 năm kinh nghiệm làm kế toán tại công ty lớn, xử lý báo cáo tài chính hàng tháng, quản lý sổ sách kế toán",
        "name": "Nguyễn Văn A",
        "email": "nguyenvana@example.com"
    },
    {
        "candidate_id": "TEST_002",
        "title": "Lập Trình Viên Python",
        "skills": "Python, Django, Flask, PostgreSQL, REST API, Git",
        "experience": "3 năm kinh nghiệm phát triển web application với Python, làm việc với Django framework, xây dựng REST API",
        "name": "Trần Thị B",
        "email": "tranthib@example.com"
    },
    {
        "candidate_id": "TEST_003",
        "title": "Nhân Viên Marketing",
        "skills": "Digital Marketing, SEO, Google Ads, Facebook Ads, Content Writing",
        "experience": "4 năm kinh nghiệm marketing online, quản lý chiến dịch quảng cáo Google và Facebook, tối ưu SEO",
        "name": "Lê Văn C",
        "email": "levanc@example.com"
    },
    {
        "candidate_id": "TEST_004",
        "title": "Kỹ Sư Phần Mềm",
        "skills": "Java, Spring Boot, Microservices, Docker, Kubernetes, AWS",
        "experience": "6 năm kinh nghiệm phát triển phần mềm enterprise, xây dựng hệ thống microservices, làm việc với cloud AWS",
        "name": "Phạm Thị D",
        "email": "phamthid@example.com"
    },
    {
        "candidate_id": "TEST_005",
        "title": "Nhân Viên Nhân Sự",
        "skills": "Tuyển dụng, Quản lý nhân sự, HRIS, Đào tạo, Quan hệ lao động",
        "experience": "4 năm kinh nghiệm quản lý nhân sự, tuyển dụng nhân viên, tổ chức đào tạo, xử lý các vấn đề quan hệ lao động",
        "name": "Hoàng Văn E",
        "email": "hoangvane@example.com"
    },
    {
        "candidate_id": "TEST_006",
        "title": "Data Analyst",
        "skills": "SQL, Python, Pandas, Tableau, Power BI, Excel, Data Visualization",
        "experience": "3 năm kinh nghiệm phân tích dữ liệu, xây dựng dashboard với Tableau và Power BI, viết query SQL phức tạp",
        "name": "Vũ Thị F",
        "email": "vuthif@example.com"
    },
    {
        "candidate_id": "TEST_007",
        "title": "Frontend Developer",
        "skills": "React, JavaScript, TypeScript, HTML, CSS, Redux, Next.js",
        "experience": "4 năm kinh nghiệm phát triển frontend, làm việc với React và TypeScript, xây dựng responsive web applications",
        "name": "Đặng Văn G",
        "email": "dangvang@example.com"
    },
    {
        "candidate_id": "TEST_008",
        "title": "Kế Toán Trưởng",
        "skills": "Kế toán tài chính, Báo cáo tài chính, Thuế, Kiểm toán, Quản lý tài chính",
        "experience": "8 năm kinh nghiệm kế toán, 3 năm làm kế toán trưởng, quản lý đội ngũ kế toán, lập báo cáo tài chính",
        "name": "Bùi Thị H",
        "email": "buithih@example.com"
    },
    {
        "candidate_id": "TEST_009",
        "title": "DevOps Engineer",
        "skills": "Docker, Kubernetes, CI/CD, Jenkins, GitLab, AWS, Terraform, Ansible",
        "experience": "5 năm kinh nghiệm DevOps, xây dựng CI/CD pipeline, quản lý infrastructure trên cloud, tự động hóa deployment",
        "name": "Ngô Văn I",
        "email": "ngovani@example.com"
    },
    {
        "candidate_id": "TEST_010",
        "title": "Product Manager",
        "skills": "Product Management, Agile, Scrum, User Research, Product Strategy, Analytics",
        "experience": "6 năm kinh nghiệm quản lý sản phẩm, làm việc với Agile/Scrum, nghiên cứu người dùng, xây dựng chiến lược sản phẩm",
        "name": "Đỗ Thị K",
        "email": "dothik@example.com"
    }
]


# Global embedding service để tránh load model nhiều lần
_embedding_service = None
_repository = None

def get_embedding_service(db: Session):
    """Get or create embedding service (singleton)."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = MultiFieldEmbeddingService(db)
    return _embedding_service

def get_repository(db: Session):
    """Get or create repository (singleton)."""
    global _repository
    if _repository is None:
        _repository = MultiFieldEmbeddingRepository(db)
    return _repository

def test_create_single_candidate(db: Session, candidate: dict):
    """Test tạo một candidate."""
    print(f"\n{'='*80}")
    print(f"TEST: Create Candidate - {candidate['candidate_id']}")
    print(f"{'='*80}")
    
    try:
        start_time = time.time()
        
        # Initialize services (reuse if possible)
        embedding_service = get_embedding_service(db)
        generator = embedding_service.generator
        repository = get_repository(db)
        
        # Validate
        if not candidate.get("experience") or not candidate["experience"].strip():
            print(f"❌ Error: Experience is required")
            return False
        
        # Generate embeddings
        print(f"   Generating embeddings...")
        embeddings = generator.generate_candidate_embeddings(
            title=candidate.get("title"),
            skills=candidate.get("skills"),
            experience=candidate["experience"]
        )
        
        # Check if candidate exists
        existing = repository.get_candidate_multi_embedding(candidate["candidate_id"])
        is_update = existing is not None
        
        # Save to database
        print(f"   Saving to database...")
        candidate_record = repository.create_candidate_multi_embedding(
            candidate_id=candidate["candidate_id"],
            title=candidate.get("title"),
            skills=candidate.get("skills"),
            experience=candidate["experience"],
            title_embedding=embeddings['title_embedding'],
            skills_embedding=embeddings['skills_embedding'],
            experience_embedding=embeddings['experience_embedding'],
            name=candidate.get("name"),
            email=candidate.get("email"),
            replace_existing=True
        )
        
        elapsed_time = time.time() - start_time
        
        status_msg = "updated" if is_update else "created"
        print(f"✅ Success: Candidate {status_msg}")
        print(f"   Candidate ID: {candidate_record.candidate_id}")
        print(f"   Name: {candidate_record.name}")
        print(f"   Embeddings Generated: ✅")
        print(f"   Time: {elapsed_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# Global matching service để tránh rebuild FAISS nhiều lần
_matching_service = None

def get_matching_service(db: Session):
    """Get or create matching service (singleton)."""
    global _matching_service
    if _matching_service is None:
        _matching_service = MultiFilterMatchingService(db, use_faiss=True)
    return _matching_service

def test_get_recommendations(db: Session, candidate_id: str):
    """Test lấy recommendations cho candidate."""
    print(f"\nTEST: Get Recommendations for {candidate_id}")
    
    try:
        matching_service = get_matching_service(db)
        
        start_time = time.time()
        recommendations = matching_service.find_jobs_for_candidate(
            candidate_id=candidate_id,
            top_k=10
        )
        elapsed_time = time.time() - start_time
        
        if recommendations:
            print(f"✅ Found {len(recommendations)} recommendations ({elapsed_time:.2f}s)")
            print("   Top 3 jobs:")
            for i, job in enumerate(recommendations[:3], 1):
                score = job.get("similarity_score", 0)
                print(f"   {i}. {job.get('title')} - Score: {score:.4f}")
            return True
        else:
            print(f"⚠️  No recommendations found")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("\n" + "="*80)
    print("🧪 TEST CANDIDATE CREATION - 10 SAMPLES (Direct)")
    print("="*80)
    
    db: Session = SessionLocal()
    
    try:
        results = {
            "created": [],
            "recommendations": []
        }
        
        # Test 1: Create candidates
        print("\n" + "="*80)
        print("TEST 1: Create Candidates")
        print("="*80)
        
        for candidate in SAMPLE_CANDIDATES:
            success = test_create_single_candidate(db, candidate)
            results["created"].append(success)
            time.sleep(0.5)  # Small delay
        
        # Test 2: Get recommendations
        print("\n" + "="*80)
        print("TEST 2: Get Recommendations")
        print("="*80)
        
        for candidate in SAMPLE_CANDIDATES[:5]:  # Test first 5
            success = test_get_recommendations(db, candidate["candidate_id"])
            results["recommendations"].append(success)
            time.sleep(0.5)
        
        # Summary
        print("\n" + "="*80)
        print("📊 TEST SUMMARY")
        print("="*80)
        
        created_count = sum(results["created"])
        recommendations_count = sum(results["recommendations"])
        
        print(f"Created: {created_count}/{len(SAMPLE_CANDIDATES)} ✅")
        print(f"Recommendations: {recommendations_count}/{5} ✅")
        
        total_tests = len(SAMPLE_CANDIDATES) + 5
        passed_tests = created_count + recommendations_count
        
        print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("\n🎉 All tests passed!")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} test(s) failed")
        
    finally:
        db.close()


if __name__ == "__main__":
    main()

