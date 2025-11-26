"""Check embedding generation progress."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.connection import SessionLocal
from src.database.repository import EmbeddingRepository

def check_progress():
    db = SessionLocal()
    try:
        repo = EmbeddingRepository(db)
        jd_count = len(repo.get_all_jd_embeddings())
        cand_count = len(repo.get_all_candidate_embeddings())
        
        print("=" * 80)
        print("EMBEDDING GENERATION PROGRESS")
        print("=" * 80)
        print(f"\nJob Descriptions: {jd_count:,} / 14,634 ({jd_count*100//14634 if jd_count > 0 else 0}%)")
        print(f"Candidates: {cand_count:,} / 44,953 ({cand_count*100//44953 if cand_count > 0 else 0}%)")
        
        if cand_count >= 5:
            print("\n✓ Ready to test with candidates!")
        else:
            print("\n⏳ Still processing candidates...")
            print("   (JDs are processed first, then candidates)")
        
        print("=" * 80)
    finally:
        db.close()

if __name__ == "__main__":
    check_progress()

