"""Generate 500 Candidate-Job pairs for ground truth labeling."""
import sys
from pathlib import Path
import csv
import random
import re
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.utils.rule_matcher import RuleMatcher
from src.utils.vietnamese_translator import VietnameseTranslator
import numpy as np

# Try to import repositories
try:
    from src.database.two_tower_repository import TwoTowerRepository
    TWO_TOWER_AVAILABLE = True
except:
    TWO_TOWER_AVAILABLE = False

try:
    from src.database.models import (
        JobDescriptionMultiEmbedding,
        CandidateMultiEmbedding
    )
    MULTI_FIELD_AVAILABLE = True
except:
    MULTI_FIELD_AVAILABLE = False

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Initialize translator for Vietnamese to English preprocessing
translator = VietnameseTranslator()


def extract_keywords(text: str) -> List[str]:
    """Extract keywords from text."""
    if not text:
        return []
    # Normalize and split
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Split and filter
    words = [w.strip() for w in text.split() if len(w.strip()) > 2]
    return words


def count_keyword_overlap(text1: str, text2: str) -> Tuple[int, int]:
    """Count keyword overlap between two texts. Returns (overlap_count, total_keywords)."""
    keywords1 = set(extract_keywords(text1))
    keywords2 = set(extract_keywords(text2))
    
    if not keywords1 or not keywords2:
        return 0, max(len(keywords1), len(keywords2))
    
    overlap = len(keywords1 & keywords2)
    total = len(keywords1 | keywords2)
    
    return overlap, total


def calculate_title_similarity(candidate_title: str, job_title: str) -> Dict[str, float]:
    """Calculate title similarity metrics."""
    if not candidate_title or not job_title:
        return {'overlap_ratio': 0.0, 'keyword_overlap_ratio': 0.0, 'keyword_count': 0}
    
    rule_matcher = RuleMatcher()
    
    # Character and token overlap
    char_overlap = rule_matcher.calculate_char_overlap(candidate_title, job_title)
    token_overlap = rule_matcher.calculate_token_overlap(candidate_title, job_title)
    max_overlap = max(char_overlap, token_overlap)
    
    # Keyword overlap
    keywords1 = set(extract_keywords(candidate_title))
    keywords2 = set(extract_keywords(job_title))
    
    if keywords1 and keywords2:
        keyword_overlap = len(keywords1 & keywords2)
        keyword_total = len(keywords1 | keywords2)
        keyword_ratio = keyword_overlap / keyword_total if keyword_total > 0 else 0.0
        # Check if >= 3/5 keywords match
        keyword_match_ratio = keyword_overlap / min(len(keywords1), len(keywords2)) if min(len(keywords1), len(keywords2)) > 0 else 0.0
    else:
        keyword_overlap = 0
        keyword_ratio = 0.0
        keyword_match_ratio = 0.0
    
    return {
        'overlap_ratio': max_overlap,
        'char_overlap': char_overlap,
        'token_overlap': token_overlap,
        'keyword_overlap': keyword_overlap,
        'keyword_ratio': keyword_ratio,
        'keyword_match_ratio': keyword_match_ratio,
        'keyword_count': len(keywords1)
    }


def count_skill_matches(candidate_skills: List[str], job_requirements: Optional[str]) -> int:
    """Count how many candidate skills match job requirements."""
    if not candidate_skills or not job_requirements:
        return 0
    
    rule_matcher = RuleMatcher()
    job_text_normalized = rule_matcher.normalize_text(job_requirements)
    
    matched_count = 0
    for skill in candidate_skills:
        skill_normalized = rule_matcher.normalize_skill(skill)
        skill_variations = rule_matcher.get_skill_variations(skill)
        
        # Check exact match
        if skill_normalized in job_text_normalized:
            matched_count += 1
            continue
        
        # Check variations
        matched = False
        for variation in skill_variations:
            variation_normalized = rule_matcher.normalize_text(variation)
            if variation_normalized in job_text_normalized:
                matched_count += 1
                matched = True
                break
        
        if matched:
            continue
        
        # Check partial match
        skill_words = skill_normalized.split()
        if len(skill_words) > 1:
            matched_words = sum(1 for word in skill_words if word in job_text_normalized)
            if matched_words >= min(2, len(skill_words)):
                matched_count += 1
    
    return matched_count


def preprocess_text(text: str) -> str:
    """
    Preprocess text: translate Vietnamese to English if needed.
    Preserves proper nouns, company names, certificates, products, abbreviations.
    """
    if not text:
        return text
    return translator.translate(text)


def preprocess_skills(skills: List[str]) -> List[str]:
    """Preprocess skills list: translate each skill if needed."""
    if not skills:
        return skills
    return [translator.translate(skill) for skill in skills]


def classify_similarity(
    candidate_title: str,
    candidate_skills: List[str],
    job_title: str,
    job_requirements: Optional[str]
) -> Tuple[str, float]:
    """
    Classify similarity as 'high', 'medium', or 'random'.
    Returns (similarity_type, predicted_similarity_score).
    
    Improved logic: Title similarity is REQUIRED for high/medium.
    
    Note: Input data should already be preprocessed (translated from Vietnamese to English).
    """
    # Calculate title similarity
    title_sim = calculate_title_similarity(candidate_title, job_title)
    
    # Count skill matches
    skill_matches = count_skill_matches(candidate_skills, job_requirements)
    
    # Rule for HIGH similarity:
    # - Title overlap >= 50% OR >= 3 keywords match (STRICT requirement)
    # - AND >= 2 skills match
    # Title similarity is MANDATORY for HIGH
    title_high = (
        title_sim['overlap_ratio'] >= 0.5 or
        (title_sim['keyword_overlap'] >= 3 and title_sim['keyword_count'] >= 5) or
        (title_sim['keyword_match_ratio'] >= 0.6 and title_sim['keyword_count'] >= 5)
    )
    skills_high = skill_matches >= 2
    
    if title_high and skills_high:
        predicted_score = 0.8 + random.uniform(0, 0.15)  # 0.8-0.95
        return 'high', predicted_score
    
    # Rule for MEDIUM similarity:
    # - Title overlap >= 35% OR >= 2 keywords match (STRICT requirement - increased from 30%)
    # - AND >= 1 skill match
    # Title similarity is MANDATORY for MEDIUM (no exceptions)
    title_medium = (
        title_sim['overlap_ratio'] >= 0.35 or
        (title_sim['keyword_overlap'] >= 2 and title_sim['keyword_count'] >= 3) or
        (title_sim['keyword_match_ratio'] >= 0.4 and title_sim['keyword_count'] >= 3)
    )
    skills_medium = skill_matches >= 1
    
    # Medium requires BOTH title relevance AND skill match
    if title_medium and skills_medium:
        predicted_score = 0.5 + random.uniform(0, 0.25)  # 0.5-0.75
        return 'medium', predicted_score
    
    # Otherwise: RANDOM (low similarity)
    predicted_score = random.uniform(0.1, 0.4)  # 0.1-0.4
    return 'random', predicted_score


def generate_ground_truth_pairs(
    candidates: List,
    jobs: List,
    target_pairs: int = 500
) -> List[Dict]:
    """
    Generate ground truth pairs with balanced distribution.
    
    Strategy:
    - For each candidate, try to find 1 high, 1 medium, 1 random pair
    - If not enough candidates, repeat with different candidates
    - Ensure no duplicate pairs
    """
    rule_matcher = RuleMatcher()
    pairs = []
    used_pairs: Set[Tuple[str, str]] = set()  # (candidate_id, job_id)
    
    # Shuffle for randomness
    candidates_shuffled = candidates.copy()
    jobs_shuffled = jobs.copy()
    random.shuffle(candidates_shuffled)
    random.shuffle(jobs_shuffled)
    
    # Track distribution
    type_counts = {'high': 0, 'medium': 0, 'random': 0}
    target_per_type = target_pairs // 3  # ~167 each
    
    print(f"Generating {target_pairs} pairs...")
    print(f"Target distribution: ~{target_per_type} pairs per type")
    print(f"Total candidates: {len(candidates)}, Total jobs: {len(jobs)}\n")
    
    # First pass: Try to get balanced pairs
    for candidate in candidates_shuffled:
        if len(pairs) >= target_pairs:
            break
        
        candidate_id = candidate.candidate_id
        # Preprocess: translate Vietnamese to English before processing
        candidate_title = preprocess_text(candidate.title or "")
        candidate_skills_raw = rule_matcher.extract_skills_from_text(candidate.skills or "")
        candidate_skills = preprocess_skills(candidate_skills_raw)
        
        # Try to find 1 high, 1 medium, 1 random pair for this candidate
        for sim_type in ['high', 'medium', 'random']:
            if len(pairs) >= target_pairs:
                break
            
            # Check if we already have enough of this type
            if type_counts[sim_type] >= target_per_type + 20:  # Allow some flexibility
                continue
            
            found = False
            attempts = 0
            max_attempts = min(100, len(jobs))
            
            while not found and attempts < max_attempts:
                # For random, truly random selection
                if sim_type == 'random':
                    job = random.choice(jobs_shuffled)
                else:
                    # For high/medium, try jobs in order
                    job_idx = (attempts + random.randint(0, len(jobs_shuffled) - 1)) % len(jobs_shuffled)
                    job = jobs_shuffled[job_idx]
                
                job_id = job.job_id
                pair_key = (candidate_id, job_id)
                
                # Skip if already used
                if pair_key in used_pairs:
                    attempts += 1
                    continue
                
                # Preprocess job data: translate Vietnamese to English
                job_title_translated = preprocess_text(job.title or "")
                job_requirements_translated = preprocess_text(job.requirement or "") if job.requirement else None
                
                # Classify similarity (will also preprocess internally, but we do it here for consistency)
                classified_type, predicted_score = classify_similarity(
                    candidate_title,
                    candidate_skills,
                    job_title_translated,
                    job_requirements_translated
                )
                
                # Accept if matches desired type
                if classified_type == sim_type:
                    pairs.append({
                        'pair_id': len(pairs) + 1,
                        'similarity_type': sim_type,
                        'candidate_id': candidate_id,
                        'job_id': job_id,
                        'candidate_title': candidate_title,  # Already translated
                        'job_title': job_title_translated,  # Translated
                        'candidate_skills': ', '.join(candidate_skills) if candidate_skills else "",  # Already translated
                        'job_requirements': job_requirements_translated or "",  # Translated
                        'predicted_similarity': round(predicted_score, 3),
                        'human_label': ''  # Empty for human labeling
                    })
                    used_pairs.add(pair_key)
                    type_counts[sim_type] += 1
                    found = True
                
                attempts += 1
    
    # Second pass: Fill remaining slots with any available pairs
    print(f"\nAfter first pass: {len(pairs)} pairs generated")
    print(f"Distribution: High={type_counts['high']}, Medium={type_counts['medium']}, Random={type_counts['random']}")
    
    if len(pairs) < target_pairs:
        print(f"\nFilling remaining {target_pairs - len(pairs)} pairs...")
        
        remaining = target_pairs - len(pairs)
        attempts = 0
        max_total_attempts = len(candidates) * len(jobs) * 2
        
        while len(pairs) < target_pairs and attempts < max_total_attempts:
            candidate = random.choice(candidates_shuffled)
            job = random.choice(jobs_shuffled)
            
            candidate_id = candidate.candidate_id
            job_id = job.job_id
            pair_key = (candidate_id, job_id)
            
            if pair_key in used_pairs:
                attempts += 1
                continue
            
            # Preprocess: translate Vietnamese to English before processing
            candidate_title = preprocess_text(candidate.title or "")
            candidate_skills_raw = rule_matcher.extract_skills_from_text(candidate.skills or "")
            candidate_skills = preprocess_skills(candidate_skills_raw)
            
            # Preprocess job data: translate Vietnamese to English
            job_title_translated = preprocess_text(job.title or "")
            job_requirements_translated = preprocess_text(job.requirement or "") if job.requirement else None
            
            classified_type, predicted_score = classify_similarity(
                candidate_title,
                candidate_skills,
                job_title_translated,
                job_requirements_translated
            )
            
            pairs.append({
                'pair_id': len(pairs) + 1,
                'similarity_type': classified_type,
                'candidate_id': candidate_id,
                'job_id': job_id,
                'candidate_title': candidate_title,  # Already translated
                'job_title': job_title_translated,  # Translated
                'candidate_skills': ', '.join(candidate_skills) if candidate_skills else "",  # Already translated
                'job_requirements': job_requirements_translated or "",  # Translated
                'predicted_similarity': round(predicted_score, 3),
                'human_label': ''
            })
            used_pairs.add(pair_key)
            type_counts[classified_type] += 1
            attempts += 1
    
    print(f"\nFinal: {len(pairs)} pairs generated")
    print(f"Final distribution: High={type_counts['high']}, Medium={type_counts['medium']}, Random={type_counts['random']}")
    
    return pairs


def main():
    """Main function to generate ground truth CSV."""
    print("=" * 80)
    print("GENERATE GROUND TRUTH 500 PAIRS")
    print("=" * 80)
    
    # Get database connection
    db: Session = next(get_db())
    
    # Try Two-Tower first, fallback to Multi-Field
    all_candidates = []
    all_jobs = []
    
    print("\nLoading data from database...")
    
    use_two_tower = False
    if TWO_TOWER_AVAILABLE:
        try:
            repository = TwoTowerRepository(db)
            all_candidates = repository.get_all_candidates()
            all_jobs = repository.get_all_jobs()
            if all_candidates and all_jobs:
                print("Using Two-Tower tables")
                use_two_tower = True
        except Exception as e:
            print(f"Two-Tower tables not available: {e}")
            db.rollback()  # Rollback failed transaction
    
    if not use_two_tower and MULTI_FIELD_AVAILABLE:
        try:
            # Get fresh connection if previous transaction failed
            if db.in_transaction():
                db.rollback()
            all_candidates = db.query(CandidateMultiEmbedding).all()
            all_jobs = db.query(JobDescriptionMultiEmbedding).all()
            if all_candidates and all_jobs:
                print("Using Multi-Field tables")
        except Exception as e:
            print(f"Multi-Field tables not available: {e}")
            db.rollback()
    
    if not all_candidates:
        print("\n❌ ERROR: No candidates found in any table!")
        print("Please process candidate dataset first.")
        return
    
    if not all_jobs:
        print("\n❌ ERROR: No jobs found in any table!")
        print("Please process job dataset first.")
        return
    
    print(f"Found {len(all_candidates)} candidates")
    print(f"Found {len(all_jobs)} jobs")
    
    if len(all_candidates) == 0:
        print("\n❌ ERROR: No candidates found in database!")
        print("Please process candidate dataset first.")
        return
    
    if len(all_jobs) == 0:
        print("\n❌ ERROR: No jobs found in database!")
        print("Please process job dataset first.")
        return
    
    # Filter candidates and jobs that have required fields
    valid_candidates = [
        c for c in all_candidates
        if c.title and (c.skills or c.experience)
    ]
    valid_jobs = [
        j for j in all_jobs
        if j.title and (j.requirement or j.skills)
    ]
    
    print(f"\nValid candidates (with title and skills/experience): {len(valid_candidates)}")
    print(f"Valid jobs (with title and requirement/skills): {len(valid_jobs)}")
    
    if len(valid_candidates) < 50:
        print(f"\n⚠️  WARNING: Only {len(valid_candidates)} valid candidates. May have difficulty generating 500 unique pairs.")
    
    if len(valid_jobs) < 100:
        print(f"\n⚠️  WARNING: Only {len(valid_jobs)} valid jobs. May have difficulty generating 500 unique pairs.")
    
    # Generate pairs
    pairs = generate_ground_truth_pairs(valid_candidates, valid_jobs, target_pairs=500)
    
    if len(pairs) < 500:
        print(f"\n⚠️  WARNING: Only generated {len(pairs)} pairs (target: 500)")
        print("This may be due to insufficient data or difficulty finding unique pairs.")
    
    # Write to CSV
    output_file = Path("ground_truth_500_pairs.csv")
    print(f"\nWriting to {output_file}...")
    
    fieldnames = [
        'pair_id',
        'similarity_type',
        'candidate_id',
        'job_id',
        'candidate_title',
        'job_title',
        'candidate_skills',
        'job_requirements',
        'predicted_similarity',
        'human_label'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)
    
    print(f"✅ Successfully created {output_file} with {len(pairs)} pairs")
    
    # Print summary statistics
    type_counts = defaultdict(int)
    for pair in pairs:
        type_counts[pair['similarity_type']] += 1
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total pairs: {len(pairs)}")
    print(f"High similarity: {type_counts['high']}")
    print(f"Medium similarity: {type_counts['medium']}")
    print(f"Random similarity: {type_counts['random']}")
    print(f"\nFile saved: {output_file.absolute()}")
    print("=" * 80)


if __name__ == '__main__':
    main()

