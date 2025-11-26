"""Find candidates with more complete data for testing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

def find_candidates_with_data():
    df = pd.read_csv('data/processed/candidates_dataset.csv', low_memory=False)
    
    print("=" * 80)
    print("FINDING CANDIDATES WITH COMPLETE DATA")
    print("=" * 80)
    
    # Score candidates based on available fields
    def score_candidate(row):
        score = 0
        if pd.notna(row.get('skills')) and str(row.get('skills', '')).strip().lower() not in ['nan', 'none', 'null', '']:
            score += 3
        if pd.notna(row.get('experience')) and str(row.get('experience', '')).strip().lower() not in ['nan', 'none', 'null', '']:
            score += 2
        if pd.notna(row.get('work_experience')) and str(row.get('work_experience', '')).strip().lower() not in ['nan', 'none', 'null', '']:
            score += 2
        if pd.notna(row.get('summary')) and str(row.get('summary', '')).strip().lower() not in ['nan', 'none', 'null', '']:
            score += 2
        if pd.notna(row.get('education')) and str(row.get('education', '')).strip().lower() not in ['nan', 'none', 'null', '']:
            score += 1
        if pd.notna(row.get('desired_job_translated')) and str(row.get('desired_job_translated', '')).strip().lower() not in ['nan', 'none', 'null', '']:
            score += 1
        return score
    
    # Score all candidates
    scores = []
    for idx, row in df.iterrows():
        cv_id = row.get('cv_id', 'N/A')
        score = score_candidate(row)
        scores.append({
            'cv_id': cv_id,
            'score': score,
            'has_skills': pd.notna(row.get('skills')) and str(row.get('skills', '')).strip().lower() not in ['nan', 'none', 'null', ''],
            'has_experience': pd.notna(row.get('experience')) and str(row.get('experience', '')).strip().lower() not in ['nan', 'none', 'null', ''],
            'has_summary': pd.notna(row.get('summary')) and str(row.get('summary', '')).strip().lower() not in ['nan', 'none', 'null', ''],
            'has_education': pd.notna(row.get('education')) and str(row.get('education', '')).strip().lower() not in ['nan', 'none', 'null', '']
        })
    
    scores_df = pd.DataFrame(scores)
    scores_df = scores_df.sort_values('score', ascending=False)
    
    print(f"\nTop 10 candidates with most complete data:")
    print("-" * 80)
    for idx, row in scores_df.head(10).iterrows():
        print(f"  CV ID: {row['cv_id']}, Score: {row['score']}")
        print(f"    Skills: {row['has_skills']}, Experience: {row['has_experience']}, Summary: {row['has_summary']}, Education: {row['has_education']}")
    
    # Get 3 random candidates with good scores (score >= 3)
    good_candidates = scores_df[scores_df['score'] >= 3]
    if len(good_candidates) >= 3:
        random_candidates = good_candidates.sample(n=3, random_state=42)
        print(f"\n{'='*80}")
        print("3 RANDOM CANDIDATES WITH GOOD DATA (for testing):")
        print(f"{'='*80}")
        for idx, row in random_candidates.iterrows():
            print(f"  CV ID: {row['cv_id']}, Score: {row['score']}")
    else:
        print(f"\n⚠️  Not enough candidates with score >= 3. Found: {len(good_candidates)}")
        if len(good_candidates) > 0:
            print("Using all available:")
            for idx, row in good_candidates.iterrows():
                print(f"  CV ID: {row['cv_id']}, Score: {row['score']}")

if __name__ == "__main__":
    find_candidates_with_data()

