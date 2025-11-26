"""Check actual candidate data to understand why matching is too generic."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

def check_candidates():
    df = pd.read_csv('data/processed/candidates_dataset.csv')
    
    print("=" * 80)
    print("CANDIDATE DATA ANALYSIS")
    print("=" * 80)
    print(f"\nTotal columns: {len(df.columns)}")
    print(f"Columns: {df.columns.tolist()[:30]}")
    
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
        print(f"  skills: {str(row.get('skills', 'N/A'))[:300]}")
        print(f"  experience: {str(row.get('experience', 'N/A'))[:300]}")
        print(f"  desired_job_translated: {str(row.get('desired_job_translated', 'N/A'))[:300]}")
        print(f"  summary: {str(row.get('summary', 'N/A'))[:300]}")
        print(f"  education: {str(row.get('education', 'N/A'))[:300]}")
        print(f"  industry: {str(row.get('industry', 'N/A'))[:300]}")
        print(f"  resume_text: {str(row.get('resume_text', 'N/A'))[:300] if pd.notna(row.get('resume_text')) else 'N/A'}")

if __name__ == "__main__":
    check_candidates()

