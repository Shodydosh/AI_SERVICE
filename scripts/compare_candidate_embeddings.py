"""Compare candidate embeddings to understand why they're similar."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.data_processing.candidate_processor import CandidateProcessor
from src.embeddings.weighted_embedding import WeightedEmbeddingGenerator
from src.database.connection import SessionLocal
from src.database.repository import EmbeddingRepository

def compare_candidates():
    print("=" * 80)
    print("COMPARING CANDIDATE EMBEDDINGS")
    print("=" * 80)
    
    # Load data
    df = pd.read_csv('data/processed/candidates_dataset.csv', low_memory=False)
    processor = CandidateProcessor()
    processor.load_from_csv('data/processed/candidates_dataset.csv')
    
    # Get embeddings from database
    db = SessionLocal()
    repo = EmbeddingRepository(db)
    
    candidate_ids = ['15001', '15002']
    
    for candidate_id in candidate_ids:
        print(f"\n{'='*80}")
        print(f"CANDIDATE {candidate_id}")
        print(f"{'='*80}")
        
        # Get from database
        candidate_emb = repo.get_candidate_embedding(candidate_id)
        if candidate_emb:
            print(f"\nFrom Database:")
            try:
                print(f"  Name: {candidate_emb.name}")
            except:
                print(f"  Name: [encoding error]")
            print(f"  Skills: {'Yes' if candidate_emb.skills else 'No'}")
            print(f"  Experience: {'Yes' if candidate_emb.experience else 'No'}")
            print(f"  Education: {'Yes' if candidate_emb.education else 'No'}")
            print(f"  Summary: {'Yes' if candidate_emb.summary else 'No'}")
            print(f"  Embedding norm: {np.linalg.norm(candidate_emb.embedding):.4f}")
        
        # Get from CSV
        row = df[df['cv_id'] == int(candidate_id)].iloc[0] if len(df[df['cv_id'] == int(candidate_id)]) > 0 else None
        if row is not None:
            print(f"\nFrom CSV - Fields Available:")
            field_texts = processor.get_field_texts(row)
            print(f"  Fields: {list(field_texts.keys())}")
            for field in field_texts.keys():
                print(f"    {field}: {len(field_texts[field])} chars")
            
            # Show dynamic weights
            weights = WeightedEmbeddingGenerator.get_dynamic_weights(
                field_texts, 
                WeightedEmbeddingGenerator.DEFAULT_CANDIDATE_WEIGHTS
            )
            print(f"\nDynamic Weights:")
            for field, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                if field in field_texts:
                    print(f"  {field}: {weight}")
    
    # Compare embeddings
    print(f"\n{'='*80}")
    print("EMBEDDING SIMILARITY")
    print(f"{'='*80}")
    
    emb1 = repo.get_candidate_embedding('15001')
    emb2 = repo.get_candidate_embedding('15002')
    
    if emb1 and emb2:
        # Calculate cosine similarity
        vec1 = np.array(emb1.embedding)
        vec2 = np.array(emb2.embedding)
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        print(f"\nCosine Similarity between Candidate 15001 and 15002: {similarity:.4f}")
        print(f"This means their embeddings are {similarity*100:.1f}% similar!")
        
        if similarity > 0.9:
            print("\n⚠️  WARNING: Embeddings are too similar (>90%)!")
            print("   This explains why they match with the same jobs.")
            print("   Possible causes:")
            print("   1. Both candidates have similar or missing data")
            print("   2. The fields used for embedding are too generic")
            print("   3. Dynamic weights are not differentiating enough")
    
    db.close()

if __name__ == "__main__":
    compare_candidates()

