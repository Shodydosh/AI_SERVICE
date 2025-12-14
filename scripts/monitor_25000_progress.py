"""Monitor progress của indexing 25K candidates và jobs."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.connection import SessionLocal
from src.database.two_tower_repository import TwoTowerRepository

db = SessionLocal()
try:
    repo = TwoTowerRepository(db)
    num_candidates = len(repo.get_all_candidates())
    num_jobs = len(repo.get_all_jobs())
    
    print(f"\n{'='*80}")
    print(f"📊 INDEXING PROGRESS - 25K DATASET")
    print(f"{'='*80}")
    print(f"\n✅ Candidates indexed: {num_candidates:,} / 25,000 ({num_candidates*100//25000}%)")
    print(f"✅ Jobs indexed:       {num_jobs:,} / 25,000 ({num_jobs*100//25000}%)")
    print(f"\n📈 Total progress:     {num_candidates + num_jobs:,} / 50,000 ({(num_candidates + num_jobs)*100//50000}%)")
    
    if num_candidates >= 25000 and num_jobs >= 25000:
        print(f"\n🎉 INDEXING HOÀN TẤT! Sẵn sàng cho testing.")
    elif num_candidates < 25000:
        remaining_candidates = 25000 - num_candidates
        print(f"\n⏳ Đang index candidates... Còn lại: {remaining_candidates:,} candidates")
    elif num_jobs < 25000:
        remaining_jobs = 25000 - num_jobs
        print(f"\n⏳ Đang index jobs... Còn lại: {remaining_jobs:,} jobs")
    
    print(f"{'='*80}\n")
finally:
    db.close()

