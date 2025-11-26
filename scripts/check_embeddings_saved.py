"""Script to check if embeddings are saved correctly in database."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.database.evaluation_models import EmbeddingEvaluationJD, EmbeddingEvaluationCandidate

db: Session = SessionLocal()

print("=" * 80)
print("CHECKING EMBEDDINGS IN DATABASE")
print("=" * 80)

# Check JD embeddings
for method_id in range(1, 6):
    count = db.query(EmbeddingEvaluationJD).filter(
        EmbeddingEvaluationJD.method_id == method_id
    ).count()
    
    if count > 0:
        sample = db.query(EmbeddingEvaluationJD).filter(
            EmbeddingEvaluationJD.method_id == method_id
        ).first()
        print(f"\nMethod {method_id} ({sample.method_name}):")
        print(f"  JD embeddings: {count}")
        print(f"  Sample job_id: {sample.job_id}")
        print(f"  Embedding dimension: {len(sample.embedding)}")
        print(f"  Embedding non-zero: {sum(1 for x in sample.embedding if x != 0)}")

# Check Candidate embeddings
for method_id in range(1, 6):
    count = db.query(EmbeddingEvaluationCandidate).filter(
        EmbeddingEvaluationCandidate.method_id == method_id
    ).count()
    
    if count > 0:
        sample = db.query(EmbeddingEvaluationCandidate).filter(
            EmbeddingEvaluationCandidate.method_id == method_id
        ).first()
        print(f"\nMethod {method_id} ({sample.method_name}):")
        print(f"  Candidate embeddings: {count}")
        print(f"  Sample candidate_id: {sample.candidate_id}")
        print(f"  Embedding dimension: {len(sample.embedding)}")
        print(f"  Embedding non-zero: {sum(1 for x in sample.embedding if x != 0)}")
    else:
        print(f"\nMethod {method_id}: No candidate embeddings found")

db.close()

