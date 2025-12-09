"""Script để in chi tiết recommendations cho các test candidates."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.services.multi_filter_matching_service import MultiFilterMatchingService
import json

# Test candidates
TEST_CANDIDATES = [
    {"id": "TEST_001", "title": "Nhân Viên Kế Toán"},
    {"id": "TEST_002", "title": "Lập Trình Viên Python"},
    {"id": "TEST_003", "title": "Nhân Viên Marketing"},
    {"id": "TEST_004", "title": "Kỹ Sư Phần Mềm"},
    {"id": "TEST_005", "title": "Nhân Viên Nhân Sự"},
]

def print_recommendations(db: Session, candidate_id: str, candidate_title: str):
    """In chi tiết recommendations cho một candidate."""
    print("\n" + "="*100)
    print(f"CANDIDATE: {candidate_id} - {candidate_title}")
    print("="*100)
    
    try:
        matching_service = MultiFilterMatchingService(db, use_faiss=True)
        
        recommendations = matching_service.find_jobs_for_candidate(
            candidate_id=candidate_id,
            top_k=10
        )
        
        if not recommendations:
            print("❌ No recommendations found")
            return
        
        print(f"\n✅ Found {len(recommendations)} recommendations\n")
        
        for i, job in enumerate(recommendations, 1):
            print(f"{i}. {job.get('title', 'N/A')}")
            print(f"   Job ID: {job.get('job_id', 'N/A')}")
            print(f"   Company: {job.get('company', 'N/A')}")
            print(f"   Location: {job.get('location', 'N/A')}")
            
            # Similarity scores
            similarity_score = job.get('similarity_score', 0)
            field_similarities = job.get('field_similarities', {})
            
            print(f"   Overall Score: {similarity_score:.4f}")
            
            if field_similarities:
                print(f"   Field Scores:")
                if 'title' in field_similarities:
                    print(f"     - Title: {field_similarities['title']:.4f}")
                if 'skills' in field_similarities:
                    print(f"     - Skills: {field_similarities['skills']:.4f}")
                if 'experience' in field_similarities:
                    print(f"     - Experience: {field_similarities['experience']:.4f}")
            
            print()
        
        # Summary
        print("-"*100)
        print(f"Summary: {len(recommendations)} jobs, Top Score: {recommendations[0].get('similarity_score', 0):.4f}")
        print("="*100)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function."""
    print("\n" + "="*100)
    print("📋 CHI TIẾT RECOMMENDATIONS CHO 5 TEST CANDIDATES")
    print("="*100)
    
    db: Session = SessionLocal()
    
    try:
        for candidate in TEST_CANDIDATES:
            print_recommendations(
                db=db,
                candidate_id=candidate["id"],
                candidate_title=candidate["title"]
            )
    
    finally:
        db.close()
    
    print("\n✅ Hoàn thành!\n")


if __name__ == "__main__":
    main()


