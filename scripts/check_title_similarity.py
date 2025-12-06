"""Check title similarity in ground truth pairs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.utils.rule_matcher import RuleMatcher

df = pd.read_csv('ground_truth_500_pairs.csv')
rule_matcher = RuleMatcher()

print("=" * 80)
print("CHECKING TITLE SIMILARITY IN HIGH/MEDIUM PAIRS")
print("=" * 80)

for sim_type in ['high', 'medium']:
    pairs = df[df['similarity_type'] == sim_type]
    print(f"\n{sim_type.upper()} similarity pairs ({len(pairs)} total):")
    print("-" * 80)
    
    mismatched = []
    for idx, row in pairs.head(30).iterrows():
        candidate_title = row['candidate_title']
        job_title = row['job_title']
        
        # Calculate actual similarity
        char_overlap = rule_matcher.calculate_char_overlap(candidate_title, job_title)
        token_overlap = rule_matcher.calculate_token_overlap(candidate_title, job_title)
        max_overlap = max(char_overlap, token_overlap)
        
        # Check if titles are actually similar
        is_similar = max_overlap >= 0.3  # At least 30% overlap
        
        if not is_similar:
            mismatched.append({
                'pair_id': row['pair_id'],
                'candidate': candidate_title[:50],
                'job': job_title[:50],
                'overlap': f"{max_overlap:.2%}"
            })
    
    if mismatched:
        print(f"\n⚠️  Found {len(mismatched)} pairs with LOW title similarity:")
        for m in mismatched[:10]:
            print(f"  Pair {m['pair_id']}: Overlap={m['overlap']}")
            print(f"    Candidate: {m['candidate']}")
            print(f"    Job:       {m['job']}")
    else:
        print("✓ All checked pairs have reasonable title similarity")

