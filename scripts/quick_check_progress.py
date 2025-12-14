"""Quick check progress - simpler version."""
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
    
    print(f"\n{'='*60}")
    print(f"📊 PROGRESS CHECK")
    print(f"{'='*60}")
    print(f"Candidates: {num_candidates:,} / 25,000 ({num_candidates*100//25000 if num_candidates <= 25000 else 100}%)")
    print(f"Jobs:       {num_jobs:,} / 25,000 ({num_jobs*100//25000 if num_jobs <= 25000 else 100}%)")
    print(f"Total:      {num_candidates + num_jobs:,} / 50,000")
    print(f"{'='*60}\n")
finally:
    db.close()

