"""Test script cho Two-Tower Matching Service với 3 embeddings."""
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.services.two_tower_matching_service import TwoTowerMatchingService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str, width: int = 100):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_step(step_num: int, description: str):
    """Print a step description."""
    print(f"\n[STEP {step_num}] {description}")
    print("-" * 100)


def explain_two_tower_flow():
    """Giải thích luồng hoạt động của Two-Tower với 3 embeddings."""
    print_section("LUỒNG HOẠT ĐỘNG CỦA TWO-TOWER MATCHING SERVICE (3 EMBEDDINGS)")
    
    print("\n📋 TỔNG QUAN:")
    print("   Two-Tower Service sử dụng 3 embeddings riêng biệt cho mỗi candidate/job:")
    print("   - Candidate: title, skills, experience")
    print("   - Job: title, skills, requirement")
    
    print("\n🔄 LUỒNG HOẠT ĐỘNG:")
    
    print_step(1, "KHỞI TẠO SERVICE")
    print("""
    - Khởi tạo CandidateTowerEncoder: Encode candidates thành 3 embeddings
    - Khởi tạo JobTowerEncoder: Encode jobs thành 3 embeddings
    - Set weights mặc định: {'title': 0.4, 'skills': 0.4, 'experience': 0.2}
    """)
    
    print_step(2, "TÌM JOBS CHO CANDIDATE")
    print("""
    2.1. Lấy candidate từ database (candidate_id)
    2.2. Encode candidate thành 3 embeddings:
         - title_embedding: Từ candidate.title
         - skills_embedding: Từ candidate.skills
         - experience_embedding: Từ candidate.experience
    2.3. Lấy tất cả jobs từ database
    2.4. Với mỗi job:
         a. Encode job thành 3 embeddings:
            - title_embedding: Từ job.title
            - skills_embedding: Từ job.skills
            - requirement_embedding: Từ job.requirement
         b. Tính cosine similarity cho từng field:
            - title_sim = dot(candidate.title_emb, job.title_emb)
            - skills_sim = dot(candidate.skills_emb, job.skills_emb)
            - exp_sim = dot(candidate.experience_emb, job.requirement_emb)
         c. Tính combined score:
            score = weight_title * title_sim + weight_skills * skills_sim + weight_exp * exp_sim
    2.5. Sắp xếp jobs theo combined score (giảm dần)
    2.6. Trả về top-k jobs với field_scores breakdown
    """)
    
    print_step(3, "TÍNH SIMILARITY CHO TỪNG FIELD")
    print("""
    3.1. Title Similarity:
         - So sánh candidate.title với job.title
         - Cosine similarity giữa 2 title embeddings
         - Trọng số: 0.4 (40%)
    
    3.2. Skills Similarity:
         - So sánh candidate.skills với job.skills
         - Cosine similarity giữa 2 skills embeddings
         - Trọng số: 0.4 (40%)
    
    3.3. Experience Similarity:
         - So sánh candidate.experience với job.requirement
         - Cosine similarity giữa experience embedding và requirement embedding
         - Trọng số: 0.2 (20%)
    """)
    
    print_step(4, "KẾT HỢP SCORES")
    print("""
    Final Score = (0.4 × title_sim) + (0.4 × skills_sim) + (0.2 × exp_sim)
    
    Ví dụ:
    - title_sim = 0.90
    - skills_sim = 0.78
    - exp_sim = 0.72
    - Final = (0.4 × 0.90) + (0.4 × 0.78) + (0.2 × 0.72) = 0.36 + 0.312 + 0.144 = 0.816
    """)
    
    print_step(5, "KẾT QUẢ TRẢ VỀ")
    print("""
    Mỗi kết quả bao gồm:
    {
        'job_id': '...',
        'title': '...',
        'company': '...',
        'location': '...',
        'score': 0.816,  # Combined weighted score
        'field_scores': {
            'title': 0.90,      # Title similarity
            'skills': 0.78,     # Skills similarity
            'experience': 0.72  # Experience vs Requirement similarity
        }
    }
    """)


def test_service(candidate_id: str = None, top_k: int = 5):
    """Test Two-Tower Matching Service."""
    print_section("TEST TWO-TOWER MATCHING SERVICE")
    
    db: Session = None
    try:
        # Connect to database
        print_step(1, "Kết nối database")
        db = SessionLocal()
        print("✓ Database connected")
        
        # Initialize service
        print_step(2, "Khởi tạo TwoTowerMatchingService")
        service = TwoTowerMatchingService(db)
        print("✓ Service initialized")
        print(f"  - Candidate Encoder: {service.candidate_encoder.actual_model_name}")
        print(f"  - Job Encoder: {service.job_encoder.actual_model_name}")
        print(f"  - Weights: {service.weights}")
        
        # Get available candidates
        print_step(3, "Lấy danh sách candidates từ database")
        repository = service.repository
        all_candidates = repository.get_all_candidates()
        print(f"✓ Found {len(all_candidates)} candidates in database")
        
        if not all_candidates:
            print("\n⚠️  Không có candidates trong database!")
            print("   Vui lòng chạy script để index candidates trước:")
            print("   python scripts/process_multi_field_embeddings.py --candidate-file ...")
            return
        
        # Select candidate
        if candidate_id:
            candidate = repository.get_candidate(candidate_id)
            if not candidate:
                print(f"\n⚠️  Candidate {candidate_id} không tồn tại!")
                print(f"   Sử dụng candidate đầu tiên thay thế")
                candidate = all_candidates[0]
        else:
            candidate = all_candidates[0]
        
        print(f"\n✓ Selected candidate:")
        print(f"  - ID: {candidate.candidate_id}")
        print(f"  - Name: {candidate.name or 'N/A'}")
        print(f"  - Title: {candidate.title or 'N/A'}")
        print(f"  - Skills: {(candidate.skills or 'N/A')[:100]}...")
        
        # Get jobs
        print_step(4, "Lấy danh sách jobs từ database")
        all_jobs = repository.get_all_jobs()
        print(f"✓ Found {len(all_jobs)} jobs in database")
        
        if not all_jobs:
            print("\n⚠️  Không có jobs trong database!")
            print("   Vui lòng chạy script để index jobs trước:")
            print("   python scripts/process_multi_field_embeddings.py --jd-file ...")
            return
        
        # Find jobs for candidate
        print_step(5, f"Tìm top {top_k} jobs cho candidate {candidate.candidate_id}")
        print("\n🔄 Đang xử lý...")
        print("   - Encode candidate thành 3 embeddings...")
        print("   - Encode tất cả jobs thành 3 embeddings...")
        print("   - Tính similarity cho từng field...")
        print("   - Kết hợp scores...")
        
        results = service.find_jobs_for_candidate(
            candidate_id=candidate.candidate_id,
            top_k=top_k
        )
        
        # Display results
        print_step(6, f"KẾT QUẢ - Top {len(results)} Jobs")
        print("\n" + "=" * 100)
        
        for idx, result in enumerate(results, 1):
            print(f"\n[{idx}] Job ID: {result['job_id']}")
            print(f"    Title: {result.get('title', 'N/A')}")
            print(f"    Company: {result.get('company', 'N/A')}")
            print(f"    Location: {result.get('location', 'N/A')}")
            print(f"    ────────────────────────────────────────────────────────────────")
            print(f"    📊 SCORES:")
            print(f"       Overall Score: {result['score']:.4f}")
            field_scores = result.get('field_scores', {})
            print(f"       ├─ Title Match:      {field_scores.get('title', 0):.4f} (weight: 0.4)")
            print(f"       ├─ Skills Match:     {field_scores.get('skills', 0):.4f} (weight: 0.4)")
            print(f"       └─ Experience Match: {field_scores.get('experience', 0):.4f} (weight: 0.2)")
            print(f"    ────────────────────────────────────────────────────────────────")
            
            # Calculate breakdown
            title_contrib = 0.4 * field_scores.get('title', 0)
            skills_contrib = 0.4 * field_scores.get('skills', 0)
            exp_contrib = 0.2 * field_scores.get('experience', 0)
            print(f"    💡 Breakdown:")
            print(f"       Title:      {title_contrib:.4f} = 0.4 × {field_scores.get('title', 0):.4f}")
            print(f"       Skills:     {skills_contrib:.4f} = 0.4 × {field_scores.get('skills', 0):.4f}")
            print(f"       Experience: {exp_contrib:.4f} = 0.2 × {field_scores.get('experience', 0):.4f}")
            print(f"       ────────────────────────────────────────────────────────────────")
            print(f"       Total:       {result['score']:.4f}")
        
        print("\n" + "=" * 100)
        print("✓ Test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
        print(f"\n❌ ERROR: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Kiểm tra database connection")
        print("   2. Đảm bảo có data trong database (candidates và jobs)")
        print("   3. Kiểm tra logs để biết chi tiết lỗi")
    finally:
        if db:
            db.close()
            print("\n✓ Database connection closed")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Two-Tower Matching Service với 3 embeddings")
    parser.add_argument("--candidate-id", type=str, help="Candidate ID để test (optional)")
    parser.add_argument("--top-k", type=int, default=5, help="Số lượng jobs top-k (default: 5)")
    parser.add_argument("--explain-only", action="store_true", help="Chỉ giải thích luồng, không chạy test")
    
    args = parser.parse_args()
    
    # Explain flow
    explain_two_tower_flow()
    
    # Run test if not explain-only
    if not args.explain_only:
        print("\n\n")
        test_service(candidate_id=args.candidate_id, top_k=args.top_k)
    else:
        print("\n\n" + "=" * 100)
        print("  Chỉ giải thích luồng hoạt động (--explain-only)")
        print("  Để chạy test, bỏ flag --explain-only")
        print("=" * 100)


if __name__ == '__main__':
    main()


