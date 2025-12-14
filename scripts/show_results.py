"""Hiển thị kết quả từ database và visualizations."""
import sys
from pathlib import Path
from datetime import datetime

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
    
    # Get latest test results if any
    print(f"\n{'='*80}")
    print(f"📊 KẾT QUẢ INDEXING VÀ TESTING")
    print(f"{'='*80}\n")
    
    print(f"📈 DATABASE STATUS:")
    print(f"   Candidates indexed: {num_candidates:,} / 25,000 ({num_candidates*100//25000 if num_candidates <= 25000 else 100}%)")
    print(f"   Jobs indexed:       {num_jobs:,} / 25,000 ({num_jobs*100//25000 if num_jobs <= 25000 else 100}%)")
    print(f"   Total:               {num_candidates + num_jobs:,} / 50,000")
    
    # Check visualization files
    viz_dir = Path("visualizations")
    if viz_dir.exists():
        viz_files = list(viz_dir.glob("two_tower_25k_*.png"))
        if viz_files:
            print(f"\n📊 VISUALIZATIONS:")
            for viz_file in sorted(viz_files):
                mtime = datetime.fromtimestamp(viz_file.stat().st_mtime)
                print(f"   ✓ {viz_file.name} ({mtime.strftime('%Y-%m-%d %H:%M:%S')})")
    
    if num_candidates < 25000 or num_jobs < 25000:
        print(f"\n⏳ STATUS: Đang indexing...")
        print(f"   Ước tính thời gian còn lại: ~20-30 phút")
    elif num_candidates >= 25000 and num_jobs >= 25000:
        print(f"\n✅ STATUS: Indexing hoàn tất!")
        print(f"   Sẵn sàng cho testing 5,000 candidates")
    
    print(f"\n{'='*80}\n")
    
finally:
    db.close()


