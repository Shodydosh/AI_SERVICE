"""Debug test script to print full candidate and job information for Two-Tower + Rule Matching."""
import sys
import json
import warnings
import logging
import io
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
import numpy as np

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    import os
    # Set console code page to UTF-8
    try:
        os.system('chcp 65001 >nul 2>&1')
    except:
        pass
    # Reconfigure stdout/stderr with UTF-8
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.utils.rule_matcher import RuleMatcher
from two_tower.model import TwoTowerModel

# Try to import Two-Tower repository, fallback to Multi-Field
try:
    from src.database.two_tower_repository import TwoTowerRepository
    TWO_TOWER_AVAILABLE = True
except:
    TWO_TOWER_AVAILABLE = False

from src.database.multi_field_repository import MultiFieldEmbeddingRepository


def load_two_tower_model(model_path: str = "outputs_improved/best_model_improved.pt", device: str = 'cpu'):
    """Load Two-Tower model with fallback handling."""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        model = TwoTowerModel(
            candidate_model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
            job_model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
            output_dim=768
        )
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    model.eval()
    return model


def build_candidate_text(candidate) -> str:
    """Build candidate text from database record."""
    parts = []
    if candidate.title:
        parts.append(f"Title: {candidate.title}")
    if candidate.skills:
        parts.append(f"Skills: {candidate.skills}")
    if candidate.experience:
        parts.append(f"Experience: {candidate.experience}")
    return " | ".join(parts) if parts else ""


def build_job_text(job) -> str:
    """Build job text from database record."""
    parts = []
    if job.title:
        parts.append(f"Title: {job.title}")
    if job.skills:
        parts.append(f"Skills: {job.skills}")
    if job.requirement:
        parts.append(f"Requirements: {job.requirement}")
    return " | ".join(parts) if parts else ""


def safe_print(text, max_length=500, use_ascii_fallback=True):
    """Safely print text with encoding handling and truncation."""
    if text is None:
        return 'N/A'
    text_str = str(text)
    
    # Truncate if too long
    if len(text_str) > max_length:
        truncated = text_str[:max_length] + f"... (truncated, total length: {len(text_str)})"
    else:
        truncated = text_str
    
    # Try to encode/decode to check if it's safe for console
    if use_ascii_fallback and sys.platform == 'win32':
        try:
            # Try encoding to see if it will work
            truncated.encode('utf-8', errors='strict')
            return truncated
        except UnicodeEncodeError:
            # If encoding fails, return ASCII-safe version
            return truncated.encode('ascii', errors='replace').decode('ascii')
    
    return truncated


def print_candidate_info(candidate, rule_matcher: RuleMatcher, output_print=print):
    """Print full candidate information."""
    output_print("\n" + "=" * 80)
    output_print("=== CANDIDATE INFO ===")
    output_print("=" * 80)
    output_print(f"ID: {candidate.candidate_id}")
    
    # Use normalized version for display to avoid encoding issues
    name_display = safe_print(candidate.name, max_length=100, use_ascii_fallback=False)
    output_print(f"Name: {name_display}")
    output_print(f"Email: {safe_print(candidate.email, max_length=100, use_ascii_fallback=False)}")
    
    title_raw = safe_print(candidate.title, max_length=200, use_ascii_fallback=False)
    output_print(f"\nTitle (raw): {title_raw}")
    title_normalized = rule_matcher.normalize_text(candidate.title or '')
    output_print(f"Title Normalized: {title_normalized}")
    
    skills_raw = safe_print(candidate.skills, max_length=300, use_ascii_fallback=False)
    output_print(f"\nSkills (raw): {skills_raw}")
    if candidate.skills:
        try:
            skills_list = rule_matcher.extract_skills_from_text(candidate.skills)
            normalized_skills = [rule_matcher.normalize_skill(s) for s in skills_list]
            output_print(f"Skills (normalized list): {normalized_skills}")
        except Exception as e:
            output_print(f"Skills (normalized list): [Error extracting: {e}]")
    else:
        output_print("Skills (normalized list): []")
    
    experience = safe_print(candidate.experience, max_length=400, use_ascii_fallback=False)
    output_print(f"\nExperience: {experience}")
    education = safe_print(getattr(candidate, 'education', None), max_length=200, use_ascii_fallback=False)
    output_print(f"Education: {education}")
    
    # Raw JSON - use ensure_ascii=False but handle encoding
    candidate_dict = {
        'candidate_id': str(candidate.candidate_id),
        'name': candidate.name,
        'email': candidate.email,
        'title': candidate.title,
        'skills': candidate.skills,
        'experience': candidate.experience,
        'education': getattr(candidate, 'education', None)
    }
    output_print(f"\nRaw JSON (first 500 chars):")
    try:
        json_str = json.dumps(candidate_dict, indent=2, ensure_ascii=False)
        # Show first part only
        preview = json_str[:500]
        output_print(preview)
        if len(json_str) > 500:
            output_print(f"... (JSON truncated, total length: {len(json_str)})")
    except Exception as e:
        output_print(f"[Error serializing JSON: {e}]")


def print_job_info(job, rule_matcher: RuleMatcher, output_print=print):
    """Print full job information."""
    output_print("\n" + "=" * 80)
    output_print("=== JOB INFO ===")
    output_print("=" * 80)
    output_print(f"ID: {job.job_id}")
    
    title_raw = safe_print(job.title, max_length=200, use_ascii_fallback=False)
    output_print(f"Title (raw): {title_raw}")
    title_normalized = rule_matcher.normalize_text(job.title or '')
    output_print(f"Title Normalized: {title_normalized}")
    
    requirement_raw = safe_print(job.requirement, max_length=400, use_ascii_fallback=False)
    output_print(f"\nRequirements (raw): {requirement_raw}")
    if job.requirement:
        try:
            req_tokens = rule_matcher.tokenize(job.requirement)
            output_print(f"Requirements Tokens: {req_tokens[:20]}...")  # Limit to 20 tokens
        except Exception as e:
            output_print(f"Requirements Tokens: [Error tokenizing: {e}]")
    else:
        output_print("Requirements Tokens: []")
    
    output_print(f"\nDescription: {safe_print(getattr(job, 'description', None), max_length=200, use_ascii_fallback=False)}")
    output_print(f"Company: {safe_print(job.company, max_length=200, use_ascii_fallback=False)}")
    output_print(f"Location: {safe_print(job.location, max_length=200, use_ascii_fallback=False)}")
    output_print(f"Level: {safe_print(getattr(job, 'level', None), max_length=100, use_ascii_fallback=False)}")
    output_print(f"Tags: {safe_print(getattr(job, 'tags', None), max_length=200, use_ascii_fallback=False)}")
    
    # Raw JSON - show preview only
    job_dict = {
        'job_id': str(job.job_id),
        'title': job.title,
        'company': job.company,
        'location': job.location,
        'skills': job.skills,
        'requirement': job.requirement,
        'description': getattr(job, 'description', None),
        'level': getattr(job, 'level', None),
        'tags': getattr(job, 'tags', None)
    }
    output_print(f"\nRaw JSON (first 500 chars):")
    try:
        json_str = json.dumps(job_dict, indent=2, ensure_ascii=False)
        # Show first part only
        preview = json_str[:500]
        output_print(preview)
        if len(json_str) > 500:
            output_print(f"... (JSON truncated, total length: {len(json_str)})")
    except Exception as e:
        output_print(f"[Error serializing JSON: {e}]")


def print_match_metrics(
    candidate,
    job,
    two_tower_similarity: float,
    rule_result: Dict[str, Any],
    rule_matcher: RuleMatcher,
    output_print=print
):
    """Print full matching metrics."""
    output_print("\n" + "=" * 80)
    output_print("=== MATCH METRICS ===")
    output_print("=" * 80)
    
    output_print(f"\nTwo-Tower Similarity: {two_tower_similarity:.4f}")
    
    # Rule 1 details (new explainable format)
    rule1 = rule_result['rule1']
    rule1_debug = rule1.get('debug', {})
    output_print(f"\nRule 1 - Title Score:")
    output_print(f"  Status: {rule1.get('status', 'UNKNOWN')}")
    output_print(f"  Score: {rule1.get('score', 0):.4f}")
    output_print(f"  Threshold: {rule1.get('threshold', 0):.4f}")
    output_print(f"  Reasons:")
    for reason in rule1.get('reasons', []):
        output_print(f"    - {reason}")
    output_print(f"\n  Debug Metrics:")
    output_print(f"    Token Jaccard: {rule1_debug.get('token_jaccard', 0):.4f}")
    output_print(f"    Sequence Similarity: {rule1_debug.get('sequence_similarity', 0):.4f}")
    output_print(f"    TF-IDF Similarity: {rule1_debug.get('tfidf_similarity', 0):.4f}")
    output_print(f"    Semantic Similarity: {rule1_debug.get('semantic_similarity', 0):.4f}")
    
    # Token analysis
    token_analysis = rule1_debug.get('token_analysis', {})
    if token_analysis:
        output_print(f"\n  Token Analysis:")
        output_print(f"    Candidate tokens: {token_analysis.get('candidate_tokens', [])[:10]}")
        output_print(f"    Job tokens: {token_analysis.get('job_tokens', [])[:10]}")
        output_print(f"    Matched tokens: {token_analysis.get('matched_tokens', [])[:10]}")
        output_print(f"    Candidate-only: {token_analysis.get('candidate_only_tokens', [])[:5]}")
        output_print(f"    Job-only: {token_analysis.get('job_only_tokens', [])[:5]}")
    
    # Rule 2 details (new explainable format)
    rule2 = rule_result['rule2']
    rule2_debug = rule2.get('debug', {})
    output_print(f"\nRule 2 - Skill Score Breakdown:")
    output_print(f"  Status: {rule2.get('status', 'UNKNOWN')}")
    output_print(f"  Score: {rule2.get('score', 0):.4f}")
    output_print(f"  Threshold: {rule2.get('threshold', 0):.4f}")
    output_print(f"  Reasons:")
    for reason in rule2.get('reasons', []):
        output_print(f"    - {reason}")
    
    output_print(f"\n  Match Breakdown:")
    output_print(f"    Exact Matches: {rule2_debug.get('exact_matches', [])}")
    output_print(f"    Synonym Matches: {rule2_debug.get('synonym_matches', [])}")
    output_print(f"    Partial Matches: {rule2_debug.get('partial_matches', [])}")
    output_print(f"    Regex/Pattern Matches: {rule2_debug.get('regex_matches', [])}")
    output_print(f"    Category Matches: {rule2_debug.get('category_matches', [])}")
    
    # Skill contributions
    skill_contributions = rule2_debug.get('skill_contributions', [])
    if skill_contributions:
        output_print(f"\n  Skill Contributions:")
        for contrib in skill_contributions[:5]:  # Show top 5
            output_print(f"    - {contrib.get('skill', 'N/A')}: {contrib.get('type', 'N/A')} (+{contrib.get('score', 0):.2f})")
            if contrib.get('details'):
                output_print(f"      Details: {contrib.get('details', '')[:80]}")
    
    output_print(f"\nFinal Decision: {rule_result.get('final_status', 'UNKNOWN')}")
    output_print(f"Reason: {rule_result.get('reason', 'N/A')}")


def print_debug_high_similarity_low_title(
    candidate,
    job,
    two_tower_similarity: float,
    title_score: float,
    rule_matcher: RuleMatcher,
    output_print=print
):
    """Print debug information for high similarity but low title overlap."""
    output_print("\n" + "!" * 80)
    output_print("-- DEBUG: High similarity but low title overlap --")
    output_print("!" * 80)
    
    candidate_title = candidate.title or ""
    job_title = job.title or ""
    
    # Char diff
    cand_chars = set(rule_matcher.normalize_text(candidate_title).replace(' ', ''))
    job_chars = set(rule_matcher.normalize_text(job_title).replace(' ', ''))
    char_intersection = cand_chars & job_chars
    char_union = cand_chars | job_chars
    char_diff = sorted((char_union - char_intersection))
    
    output_print(f"\nChar Analysis:")
    output_print(f"  Candidate chars: {len(cand_chars)} unique")
    output_print(f"  Job chars: {len(job_chars)} unique")
    output_print(f"  Common chars: {len(char_intersection)}")
    output_print(f"  Different chars: {char_diff[:20]}")
    output_print(f"  Common chars list: {sorted(char_intersection)[:20]}")
    
    # Token diff
    cand_tokens = set(rule_matcher.tokenize(candidate_title))
    job_tokens = set(rule_matcher.tokenize(job_title))
    token_intersection = cand_tokens & job_tokens
    token_union = cand_tokens | job_tokens
    
    output_print(f"\nToken Analysis:")
    output_print(f"  Candidate tokens: {cand_tokens}")
    output_print(f"  Job tokens: {job_tokens}")
    output_print(f"  Common tokens: {token_intersection}")
    output_print(f"  Different tokens: {sorted(token_union - token_intersection)}")
    
    # Embedding distance breakdown (if available)
    output_print(f"\nEmbedding Analysis:")
    output_print(f"  Two-Tower Similarity: {two_tower_similarity:.4f}")
    output_print(f"  Title Score: {title_score:.4f}")
    output_print(f"  Difference: {two_tower_similarity - title_score:.4f}")
    output_print(f"  This suggests similarity comes from skills/experience, not title match")


def compute_all_matches(
    db: Session,
    model: TwoTowerModel,
    rule_matcher: RuleMatcher,
    max_pairs: int = 100
) -> List[Dict[str, Any]]:
    """Compute matches for all candidate-job pairs."""
    # Try Two-Tower repository first, fallback to Multi-Field
    if TWO_TOWER_AVAILABLE:
        try:
            repository = TwoTowerRepository(db)
            all_candidates = repository.get_all_candidates()
            all_jobs = repository.get_all_jobs()
        except Exception as e:
            # Rollback failed transaction
            db.rollback()
            # Fallback to Multi-Field repository
            repository = MultiFieldEmbeddingRepository(db)
            all_candidates = repository.get_all_candidate_multi_embeddings()
            all_jobs = repository.get_all_job_multi_embeddings()
    else:
        # Use Multi-Field repository
        repository = MultiFieldEmbeddingRepository(db)
        all_candidates = repository.get_all_candidate_multi_embeddings()
        all_jobs = repository.get_all_job_multi_embeddings()
    
    if not all_candidates or not all_jobs:
        print("No candidates or jobs found in database!")
        return []
    
    print(f"\nComputing matches for {len(all_candidates)} candidates x {len(all_jobs)} jobs...")
    print(f"Limiting to {max_pairs} pairs for performance...")
    
    # Sample pairs
    np.random.seed(42)
    if len(all_candidates) > max_pairs:
        candidate_indices = np.random.choice(len(all_candidates), max_pairs, replace=False)
        candidates = [all_candidates[i] for i in candidate_indices]
    else:
        candidates = all_candidates
    
    if len(all_jobs) > max_pairs:
        job_indices = np.random.choice(len(all_jobs), max_pairs, replace=False)
        jobs = [all_jobs[i] for i in job_indices]
    else:
        jobs = all_jobs
    
    matches = []
    
    with torch.no_grad():
        for i, candidate in enumerate(candidates):
            if (i + 1) % 10 == 0:
                print(f"  Processing candidate {i+1}/{len(candidates)}...")
            
            candidate_text = build_candidate_text(candidate)
            if not candidate_text:
                continue
            
            candidate_emb = model.encode_candidates([candidate_text])[0]
            
            for job in jobs:
                job_text = build_job_text(job)
                if not job_text:
                    continue
                
                job_emb = model.encode_jobs([job_text])[0]
                
                # Compute cosine similarity
                similarity = float(torch.sum(candidate_emb * job_emb).item())
                
                # Rule matching
                candidate_skills = rule_matcher.extract_skills_from_text(candidate.skills or "")
                rule_result = rule_matcher.evaluate_match(
                    candidate_title=candidate.title or "",
                    candidate_skills=candidate_skills,
                    job_title=job.title or "",
                    job_requirements=job.requirement,
                    job_description=getattr(job, 'description', None)
                )
                
                matches.append({
                    'candidate': candidate,
                    'job': job,
                    'similarity': similarity,
                    'rule_result': rule_result,
                    'candidate_text': candidate_text,
                    'job_text': job_text
                })
    
    return matches


def test_debug_samples(output_file: str = None):
    """
    Main test function to print debug samples.
    
    Args:
        output_file: Optional path to output file. If provided, writes to file with UTF-8 encoding.
                     If None, prints to console.
    """
    # Setup output
    if output_file:
        output_fp = open(output_file, 'w', encoding='utf-8')
        def output_print(*args, **kwargs):
            print(*args, **kwargs, file=output_fp)
            print(*args, **kwargs)  # Also print to console
    else:
        output_fp = None
        output_print = print
    
    output_print("=" * 80)
    output_print("TWO-TOWER + RULE MATCHING DEBUG SAMPLES")
    output_print("=" * 80)
    output_print("\nModel Configuration:")
    output_print(f"  Preferred Model: VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
    output_print(f"  Fallback Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    if output_file:
        output_print(f"\n[INFO] Output will be saved to: {output_file}")
    
    try:
        # Load model
        output_print("\n1. Loading Two-Tower model...")
        model = load_two_tower_model()
        output_print("[OK] Model loaded")
        
        # Print model information
        candidate_model = getattr(model, 'candidate_model_name', 'Unknown')
        job_model = getattr(model, 'job_model_name', 'Unknown')
        output_print(f"   Candidate Tower Model: {candidate_model}")
        output_print(f"   Job Tower Model: {job_model}")
        
        # Check if fallback is being used
        preferred = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
        fallback = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        if candidate_model != preferred or job_model != preferred:
            output_print(f"\n   [WARNING] Using fallback model!")
            output_print(f"   [WARNING] Preferred model '{preferred}' was not available")
            output_print(f"   [WARNING] Current model: {candidate_model}")
    except Exception as e:
        output_print(f"[ERROR] Error loading model: {e}")
        if output_fp:
            output_fp.close()
        return
    
    # Initialize rule matcher
    output_print("\n2. Initializing Rule Matcher...")
    rule_matcher = RuleMatcher()
    output_print("[OK] Rule Matcher initialized")
    
    # Print semantic model info if available
    if hasattr(rule_matcher, 'semantic_model_name') and rule_matcher.semantic_model_name:
        output_print(f"   Semantic Model: {rule_matcher.semantic_model_name}")
        preferred_semantic = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
        if rule_matcher.semantic_model_name != preferred_semantic:
            output_print(f"   [WARNING] Semantic model using fallback: {rule_matcher.semantic_model_name}")
    
    # Load data and compute matches
    output_print("\n3. Loading data and computing matches...")
    db = SessionLocal()
    try:
        matches = compute_all_matches(db, model, rule_matcher, max_pairs=50)
        output_print(f"[OK] Computed {len(matches)} matches")
    except Exception as e:
        output_print(f"[ERROR] Error computing matches: {e}")
        import traceback
        traceback.print_exc()
        db.close()
        if output_fp:
            output_fp.close()
        return
    finally:
        db.close()
    
    if not matches:
        output_print("No matches found!")
        if output_fp:
            output_fp.close()
        return
    
    # Sort by similarity
    matches.sort(key=lambda x: x['similarity'], reverse=True)
    
    # A. Top 5 Highest Score Matches
    output_print("\n" + "=" * 80)
    output_print("A. TOP 5 HIGHEST SCORE MATCHES")
    output_print("=" * 80)
    
    for i, match in enumerate(matches[:5], 1):
        output_print(f"\n{'='*80}")
        output_print(f"=== TOP MATCH #{i} ===")
        output_print(f"{'='*80}")
        
        print_candidate_info(match['candidate'], rule_matcher, output_print)
        print_job_info(match['job'], rule_matcher, output_print)
        print_match_metrics(
            match['candidate'],
            match['job'],
            match['similarity'],
            match['rule_result'],
            rule_matcher,
            output_print
        )
        
        # Debug if needed
        if match['similarity'] > 0.85 and match['rule_result'].get('final_title_score', 0) < 0.4:
            print_debug_high_similarity_low_title(
                match['candidate'],
                match['job'],
                match['similarity'],
                match['rule_result'].get('final_title_score', 0),
                rule_matcher,
                output_print
            )
    
    # B. Low-score but Similar-title cases
    output_print("\n" + "=" * 80)
    output_print("B. LOW-SCORE BUT SIMILAR-TITLE CASES")
    output_print("=" * 80)
    output_print("(To detect false-negatives: similar title but low similarity)")
    
    # Filter: title score >= 0.6 but similarity < 0.5
    low_score_similar_title = [
        m for m in matches
        if m['rule_result'].get('final_title_score', 0) >= 0.6 and m['similarity'] < 0.5
    ]
    low_score_similar_title.sort(key=lambda x: x['rule_result'].get('final_title_score', 0), reverse=True)
    
    for i, match in enumerate(low_score_similar_title[:3], 1):
        output_print(f"\n{'='*80}")
        output_print(f"=== LOW-SCORE SIMILAR-TITLE CASE #{i} ===")
        output_print(f"{'='*80}")
        
        print_candidate_info(match['candidate'], rule_matcher, output_print)
        print_job_info(match['job'], rule_matcher, output_print)
        print_match_metrics(
            match['candidate'],
            match['job'],
            match['similarity'],
            match['rule_result'],
            rule_matcher,
            output_print
        )
    
    # C. Bad-match suspicion (high similarity but low title/skills match)
    output_print("\n" + "=" * 80)
    output_print("C. BAD-MATCH SUSPICION")
    output_print("=" * 80)
    output_print("(To detect false-positives: high similarity but title/skills don't match)")
    
    # Filter: similarity > 0.7 but title < 0.5 and skill < 1.0
    bad_matches = [
        m for m in matches
        if m['similarity'] > 0.7
        and m['rule_result'].get('final_title_score', 0) < 0.5
        and m['rule_result'].get('skill_score', 0) < 1.0
    ]
    bad_matches.sort(key=lambda x: x['similarity'], reverse=True)
    
    for i, match in enumerate(bad_matches[:2], 1):
        output_print(f"\n{'='*80}")
        output_print(f"=== BAD-MATCH SUSPICION #{i} ===")
        output_print(f"{'='*80}")
        
        print_candidate_info(match['candidate'], rule_matcher, output_print)
        print_job_info(match['job'], rule_matcher, output_print)
        print_match_metrics(
            match['candidate'],
            match['job'],
            match['similarity'],
            match['rule_result'],
            rule_matcher,
            output_print
        )
        
        # Always print debug for bad matches
        print_debug_high_similarity_low_title(
            match['candidate'],
            match['job'],
            match['similarity'],
            match['rule_result'].get('final_title_score', 0),
            rule_matcher,
            output_print
        )
    
    output_print("\n" + "=" * 80)
    output_print("DEBUG SAMPLES COMPLETE")
    output_print("=" * 80)
    
    # Close file if opened
    if output_fp:
        output_fp.close()
        output_print(f"\n[INFO] Output saved to: {output_file}")


if __name__ == '__main__':
    import sys
    # Allow output file as command line argument
    output_file = sys.argv[1] if len(sys.argv) > 1 else None
    test_debug_samples(output_file=output_file)

