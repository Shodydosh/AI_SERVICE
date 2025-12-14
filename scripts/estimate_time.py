"""Ước tính thời gian còn lại cho indexing và testing."""
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
    
    # Target
    TARGET_CANDIDATES = 25000
    TARGET_JOBS = 25000
    TARGET_TESTS = 5000
    
    print(f"\n{'='*80}")
    print(f"⏱️  ƯỚC TÍNH THỜI GIAN CÒN LẠI")
    print(f"{'='*80}\n")
    
    print(f"📊 TIẾN ĐỘ HIỆN TẠI:")
    print(f"   Candidates: {num_candidates:,} / {TARGET_CANDIDATES:,} ({num_candidates*100//TARGET_CANDIDATES if num_candidates <= TARGET_CANDIDATES else 100}%)")
    print(f"   Jobs:       {num_jobs:,} / {TARGET_JOBS:,} ({num_jobs*100//TARGET_JOBS if num_jobs <= TARGET_JOBS else 100}%)")
    
    # Tính toán dựa trên tốc độ ước tính
    # Với batch encoding (64/batch) và batch commit (200/batch):
    # - Encoding: ~0.5s per batch (64 texts) = ~0.008s per text
    # - Database: ~0.1s per commit (200 records)
    # - Tổng: ~0.01s per record (bao gồm cả 3 embeddings)
    
    # Tốc độ ước tính (records/second)
    ESTIMATED_RATE = 10  # records/second với batch encoding
    
    remaining_candidates = max(0, TARGET_CANDIDATES - num_candidates)
    remaining_jobs = max(0, TARGET_JOBS - num_jobs)
    
    # Thời gian còn lại cho indexing
    if remaining_candidates > 0:
        candidate_time = remaining_candidates / ESTIMATED_RATE
        print(f"\n⏳ INDEXING CANDIDATES:")
        print(f"   Còn lại: {remaining_candidates:,} candidates")
        print(f"   Ước tính: {candidate_time/60:.1f} phút ({candidate_time:.0f} giây)")
    else:
        print(f"\n✅ INDEXING CANDIDATES: Hoàn tất!")
        candidate_time = 0
    
    if remaining_jobs > 0:
        job_time = remaining_jobs / ESTIMATED_RATE
        print(f"\n⏳ INDEXING JOBS:")
        print(f"   Còn lại: {remaining_jobs:,} jobs")
        print(f"   Ước tính: {job_time/60:.1f} phút ({job_time:.0f} giây)")
    else:
        print(f"\n✅ INDEXING JOBS: Hoàn tất!")
        job_time = 0
    
    # Thời gian testing
    if num_candidates >= TARGET_CANDIDATES and num_jobs >= TARGET_JOBS:
        # Testing: ~0.5s per candidate (encode + similarity với 25K jobs)
        test_time = TARGET_TESTS * 0.5
        print(f"\n⏳ TESTING:")
        print(f"   Số lượng: {TARGET_TESTS:,} candidates")
        print(f"   Ước tính: {test_time/60:.1f} phút ({test_time:.0f} giây)")
    else:
        print(f"\n⏸️  TESTING: Chưa bắt đầu (đợi indexing hoàn tất)")
        test_time = 0
    
    # Tổng thời gian
    total_time = candidate_time + job_time + test_time
    if total_time > 0:
        print(f"\n{'='*80}")
        print(f"⏱️  TỔNG THỜI GIAN CÒN LẠI:")
        print(f"   {total_time/60:.1f} phút ({total_time:.0f} giây)")
        print(f"   ~{total_time/3600:.2f} giờ")
        
        # Ước tính thời gian hoàn thành
        completion_time = datetime.now() + timedelta(seconds=total_time)
        print(f"\n🕐 DỰ KIẾN HOÀN THÀNH:")
        print(f"   {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"\n✅ TẤT CẢ ĐÃ HOÀN TẤT!")
    
    print(f"{'='*80}\n")
    
finally:
    db.close()


