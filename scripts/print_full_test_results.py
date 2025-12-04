"""In đầy đủ kết quả test với recommendations chi tiết."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.services.multi_filter_matching_service import MultiFilterMatchingService

# Test candidates
TEST_CANDIDATES = [
    {"id": "TEST_001", "title": "Nhân Viên Kế Toán", "name": "Nguyễn Văn A"},
    {"id": "TEST_002", "title": "Lập Trình Viên Python", "name": "Trần Thị B"},
    {"id": "TEST_003", "title": "Nhân Viên Marketing", "name": "Lê Văn C"},
    {"id": "TEST_004", "title": "Kỹ Sư Phần Mềm", "name": "Phạm Thị D"},
    {"id": "TEST_005", "title": "Nhân Viên Nhân Sự", "name": "Hoàng Văn E"},
]

def print_full_results():
    """In đầy đủ kết quả test."""
    print("\n" + "="*100)
    print("📊 KẾT QUẢ TEST ĐẦY ĐỦ - 5 TEST CANDIDATES")
    print("="*100)
    
    db: Session = SessionLocal()
    
    try:
        matching_service = MultiFilterMatchingService(db, use_faiss=True)
        
        for candidate in TEST_CANDIDATES:
            print("\n" + "="*100)
            print(f"🎯 CANDIDATE: {candidate['id']} - {candidate['title']}")
            print(f"   Name: {candidate['name']}")
            print("="*100)
            
            try:
                recommendations = matching_service.find_jobs_for_candidate(
                    candidate_id=candidate["id"],
                    top_k=10
                )
                
                if not recommendations:
                    print("❌ No recommendations found\n")
                    continue
                
                print(f"\n✅ Found {len(recommendations)} recommendations\n")
                
                for i, job in enumerate(recommendations, 1):
                    print(f"{i:2d}. {job.get('title', 'N/A')}")
                    print(f"     Job ID: {job.get('job_id', 'N/A')}")
                    print(f"     Company: {job.get('company', 'N/A')}")
                    print(f"     Location: {job.get('location', 'N/A')}")
                    
                    similarity_score = job.get('similarity_score', 0)
                    field_similarities = job.get('field_similarities', {})
                    
                    print(f"     Overall Score: {similarity_score:.4f}")
                    
                    if field_similarities:
                        title_score = field_similarities.get('title', 0)
                        skills_score = field_similarities.get('skills', 0)
                        exp_score = field_similarities.get('experience', 0)
                        
                        print(f"     Field Scores: Title={title_score:.4f}, Skills={skills_score:.4f}, Exp={exp_score:.4f}")
                    
                    print()
                
                # Summary
                top_score = recommendations[0].get('similarity_score', 0) if recommendations else 0
                print(f"📈 Summary: {len(recommendations)} jobs | Top Score: {top_score:.4f}")
                print("-"*100)
                
            except Exception as e:
                print(f"❌ Error: {e}\n")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*100)
        print("✅ HOÀN THÀNH")
        print("="*100)
        print(f"\n📁 Visualization saved to: visualizations/test_candidates_tsne.png")
        print(f"   (t-SNE plot showing embeddings of 5 candidates vs 200 sample jobs)\n")
        
    finally:
        db.close()


if __name__ == "__main__":
    print_full_results()

