"""Check database status for multi-field embeddings."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.database.multi_field_repository import MultiFieldEmbeddingRepository

def main():
    print("=" * 80)
    print("DATABASE STATUS CHECK")
    print("=" * 80)
    
    db: Session = next(get_db())
    try:
        repo = MultiFieldEmbeddingRepository(db)
        
        job_count = repo.count_job_multi_embeddings()
        candidate_count = repo.count_candidate_multi_embeddings()
        
        print(f"\nJobs in database: {job_count}")
        print(f"Candidates in database: {candidate_count}")
        
        if candidate_count > 0:
            print("\nFirst 10 candidate IDs:")
            candidates = repo.get_all_candidate_multi_embeddings()
            for i, c in enumerate(candidates[:10], 1):
                print(f"  {i}. {c.candidate_id}")
        else:
            print("\n⚠️  No candidates in database. Please process candidate dataset first.")
        
        if job_count > 0:
            print("\nFirst 10 job IDs:")
            jobs = repo.get_all_job_multi_embeddings()
            for i, j in enumerate(jobs[:10], 1):
                print(f"  {i}. {j.job_id}")
        else:
            print("\n⚠️  No jobs in database. Please process job dataset first.")
        
        print("\n" + "=" * 80)
        
        if job_count == 0 or candidate_count == 0:
            print("\nTo process data, run:")
            print("  python scripts/process_multi_field_embeddings.py --jd-file data/raw/job_data.csv")
            print("  python scripts/process_multi_field_embeddings.py --candidate-file data/raw/candidates_dataset.csv")
        
    except Exception as e:
        print(f"\n❌ Error checking database: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    main()



