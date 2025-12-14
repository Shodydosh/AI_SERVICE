"""Ước tính thời gian cho 5K dataset."""
import sys
from pathlib import Path
from datetime import datetime, timedelta

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
    TARGET_TESTS = 500
    
    # Tốc độ ước tính với batch encoding
    ESTIMATED_RATE = 10  # records/second
    
    remaining_candidates = max(0, TARGET_CANDIDATES - num_candidates)
    remaining_jobs = max(0, TARGET_JOBS - num_jobs)
    
    print(f"\n{'='*80}")
    print(f"⏱️  ƯỚC TÍNH THỜI GIAN - 5K DATASET")
    print(f"{'='*80}\n")
    
    print(f"📊 TIẾN ĐỘ:")
    print(f"   Candidates: {num_candidates:,} / {TARGET_CANDIDATES:,}")
    print(f"   Jobs:       {num_jobs:,} / {TARGET_JOBS:,}")
    
    # Thời gian indexing
    candidate_time = remaining_candidates / ESTIMATED_RATE if remaining_candidates > 0 else 0
    job_time = remaining_jobs / ESTIMATED_RATE if remaining_jobs > 0 else 0
    
    # Thời gian testing (0.5s per candidate với 5K jobs)
    test_time = TARGET_TESTS * 0.5 if (num_candidates >= TARGET_CANDIDATES and num_jobs >= TARGET_JOBS) else 0
    
    total_time = candidate_time + job_time + test_time
    
    print(f"\n⏳ THỜI GIAN CÒN LẠI:")
    if remaining_candidates > 0:
        print(f"   Indexing Candidates: ~{candidate_time/60:.1f} phút")
    if remaining_jobs > 0:
        print(f"   Indexing Jobs:       ~{job_time/60:.1f} phút")
    if test_time > 0:
        print(f"   Testing (500):       ~{test_time/60:.1f} phút")
    
    print(f"\n⏱️  TỔNG: ~{total_time/60:.1f} phút ({total_time:.0f} giây)")
    
    if total_time > 0:
        completion_time = datetime.now() + timedelta(seconds=total_time)
        print(f"🕐 Dự kiến hoàn thành: {completion_time.strftime('%H:%M:%S')}")
    
    print(f"{'='*80}\n")
    
finally:
    db.close()


