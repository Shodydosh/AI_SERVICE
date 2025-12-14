"""Check indexing progress."""
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
    
    print(f"\n{'='*60}")
    print(f"INDEXING PROGRESS")
    print(f"{'='*60}")
    print(f"Candidates indexed: {num_candidates}")
    print(f"Jobs indexed: {num_jobs}")
    print(f"Total: {num_candidates + num_jobs} records")
    print(f"{'='*60}\n")
finally:
    db.close()

