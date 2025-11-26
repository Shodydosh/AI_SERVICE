"""Check detailed candidate data to understand why embeddings are similar."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.data_processing.candidate_processor import CandidateProcessor
from src.embeddings.weighted_embedding import WeightedEmbeddingGenerator

def check_candidates():
    df = pd.read_csv('data/processed/candidates_dataset.csv')
    processor = CandidateProcessor()
    processor.load_from_csv('data/processed/candidates_dataset.csv')
    
    print("=" * 80)
    print("CANDIDATE DATA COMPARISON")
    print("=" * 80)
    
    # Check first 2 candidates
    for idx in [0, 1]:
        print("\n" + "=" * 80)
        print(f"CANDIDATE {idx+1}")
        print("=" * 80)
        row = df.iloc[idx]
        
        print(f"\nBasic Info:")
        print(f"  cv_id: {row.get('cv_id', 'N/A')}")
        print(f"  user_name: {row.get('user_name', 'N/A')}")
        
        print(f"\nKey Fields for Embedding:")
        key_fields = ['skills', 'experience', 'work_experience', 'desired_job_translated', 
                     'summary', 'education', 'industry', 'resume_text']
        for col in key_fields:
            if col in df.columns:
                value = row.get(col, 'N/A')
                if pd.notna(value) and str(value).strip() and str(value).lower() not in ['nan', 'none', 'null', '']:
                    try:
                        print(f"  {col}: {str(value)[:200]}")
                    except:
                        print(f"  {col}: [encoding error]")
        
        print(f"\nFields Used for Embedding:")
        field_texts = processor.get_field_texts(row)
        print(f"  Available fields: {list(field_texts.keys())}")
        for field, text in field_texts.items():
            print(f"  {field}: {text[:200]}...")
        
        # Show what would be embedded
        combined_text = processor.get_combined_text(row)
        print(f"\nCombined Text (first 500 chars):")
        print(f"  {combined_text[:500]}...")
        
        # Show dynamic weights
        weights = WeightedEmbeddingGenerator.get_dynamic_weights(
            field_texts, 
            WeightedEmbeddingGenerator.DEFAULT_CANDIDATE_WEIGHTS
        )
        print(f"\nDynamic Weights Applied:")
        for field, weight in weights.items():
            if field in field_texts:
                print(f"  {field}: {weight}")

if __name__ == "__main__":
    check_candidates()

