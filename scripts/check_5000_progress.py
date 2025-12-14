"""Check progress cho 5K dataset."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.connection import SessionLocal
from sqlalchemy import text

db = SessionLocal()
try:
    # Count candidates
    result = db.execute(text("SELECT COUNT(*) FROM candidate_two_tower"))
    num_candidates = result.scalar()
    
    # Count jobs
    result = db.execute(text("SELECT COUNT(*) FROM job_description_two_tower"))
    num_jobs = result.scalar()
    
    TARGET_CANDIDATES = 5000
    TARGET_JOBS = 5000
    
    print(f"\n{'='*80}")
    print(f"📊 PROGRESS - 5K DATASET")
    print(f"{'='*80}\n")
    
    print(f"Candidates: {num_candidates:,} / {TARGET_CANDIDATES:,} ({num_candidates*100//TARGET_CANDIDATES if num_candidates <= TARGET_CANDIDATES else 100}%)")
    print(f"Jobs:       {num_jobs:,} / {TARGET_JOBS:,} ({num_jobs*100//TARGET_JOBS if num_jobs <= TARGET_JOBS else 100}%)")
    print(f"Total:      {num_candidates + num_jobs:,} / {TARGET_CANDIDATES + TARGET_JOBS:,}")
    
    if num_candidates >= TARGET_CANDIDATES and num_jobs >= TARGET_JOBS:
        print(f"\n✅ INDEXING HOÀN TẤT! Sẵn sàng cho testing 500 candidates.")
    else:
        remaining_candidates = TARGET_CANDIDATES - num_candidates
        remaining_jobs = TARGET_JOBS - num_jobs
        print(f"\n⏳ Đang indexing...")
        if remaining_candidates > 0:
            print(f"   Còn lại: {remaining_candidates:,} candidates")
        if remaining_jobs > 0:
            print(f"   Còn lại: {remaining_jobs:,} jobs")
    
    print(f"{'='*80}\n")
finally:
    db.close()


