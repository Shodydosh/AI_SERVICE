"""Script to recommend top 10 jobs for each candidate and print results for manual review."""
import sys
import json
import warnings
import logging
from pathlib import Path
from typing import List, Dict, Any
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import và setup UTF-8 logging
from src.utils.logging_utf8 import setup_utf8_logging

# Setup logging với UTF-8 (chỉ console, không file)
setup_utf8_logging(level=logging.INFO, log_file=None)

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

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
    """Load Two-Tower model."""
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


def safe_print(text, max_length=500):
    """Safely print text with encoding handling and truncation."""
    if text is None:
        return 'N/A'
    text_str = str(text)
    # Ensure UTF-8 encoding
    if isinstance(text_str, bytes):
        try:
            text_str = text_str.decode('utf-8')
        except:
            text_str = text_str.decode('utf-8', errors='replace')
    if len(text_str) > max_length:
        return text_str[:max_length] + f"... (truncated, total length: {len(text_str)})"
    return text_str


def print_candidate_summary(candidate, rule_matcher: RuleMatcher, output_print=print):
    """Print candidate summary."""
    output_print(f"\n{'='*100}")
    output_print(f"CANDIDATE ID: {candidate.candidate_id}")
    output_print(f"{'='*100}")
    output_print(f"Name: {safe_print(candidate.name, max_length=100)}")
    output_print(f"Title: {safe_print(candidate.title, max_length=200)}")
    output_print(f"Title Normalized: {rule_matcher.normalize_text(candidate.title or '')}")
    
    if candidate.skills:
        try:
            skills_list = rule_matcher.extract_skills_from_text(candidate.skills)
            output_print(f"Skills ({len(skills_list)}): {', '.join(skills_list[:10])}")
        except:
            output_print(f"Skills: {safe_print(candidate.skills, max_length=300)}")
    
    experience = safe_print(candidate.experience, max_length=300)
    output_print(f"Experience: {experience}")


def print_job_recommendation(job, similarity: float, rule_result: Dict[str, Any], rank: int, rule_matcher: RuleMatcher, output_print=print):
    """Print job recommendation with details."""
    output_print(f"\n{'-'*100}")
    output_print(f"RECOMMENDATION #{rank} | Similarity: {similarity:.4f} | Final Decision: {rule_result.get('final_status', 'UNKNOWN')}")
    output_print(f"{'-'*100}")
    output_print(f"Job ID: {job.job_id}")
    output_print(f"Title: {safe_print(job.title, max_length=200)}")
    output_print(f"Title Normalized: {rule_matcher.normalize_text(job.title or '')}")
    output_print(f"Company: {safe_print(job.company, max_length=200)}")
    output_print(f"Location: {safe_print(job.location, max_length=200)}")
    
    # Matching scores
    rule1 = rule_result.get('rule1', {})
    rule2 = rule_result.get('rule2', {})
    
    output_print(f"\n📊 Matching Scores:")
    output_print(f"  Two-Tower Similarity: {similarity:.4f}")
    output_print(f"  Rule 1 - Title Score: {rule1.get('score', 0):.4f} (Status: {rule1.get('status', 'UNKNOWN')}, Threshold: {rule1.get('threshold', 0):.4f})")
    output_print(f"  Rule 2 - Skill Score: {rule2.get('score', 0):.4f} (Status: {rule2.get('status', 'UNKNOWN')}, Threshold: {rule2.get('threshold', 0):.4f})")
    output_print(f"  Final Decision: {rule_result.get('final_status', 'UNKNOWN')}")
    output_print(f"  Reason: {rule_result.get('reason', 'N/A')}")
    
    # Rule 1 details
    rule1_debug = rule1.get('debug', {})
    if rule1_debug:
        output_print(f"\n  Rule 1 Details:")
        output_print(f"    Semantic Similarity: {rule1_debug.get('semantic_similarity', 0):.4f}")
        output_print(f"    Token Jaccard: {rule1_debug.get('token_jaccard', 0):.4f}")
        output_print(f"    Sequence Similarity: {rule1_debug.get('sequence_similarity', 0):.4f}")
    
    # Rule 2 details
    rule2_debug = rule2.get('debug', {})
    if rule2_debug:
        matched_count = rule2_debug.get('match_count', 0)
        total_skills = rule2_debug.get('total_candidate_skills', 0)
        output_print(f"\n  Rule 2 Details:")
        output_print(f"    Matched Skills: {matched_count}/{total_skills}")
        if rule2_debug.get('exact_matches'):
            output_print(f"    Exact Matches: {len(rule2_debug['exact_matches'])}")
        if rule2_debug.get('synonym_matches'):
            output_print(f"    Synonym/Translation Matches: {len(rule2_debug['synonym_matches'])}")
            # Show first few
            for match in rule2_debug['synonym_matches'][:3]:
                output_print(f"      - {safe_print(match, max_length=80)}")
    
    # Job requirements preview
    if job.requirement:
        req_preview = safe_print(job.requirement, max_length=400)
        output_print(f"\n  Job Requirements: {req_preview}")


def recommend_jobs_for_candidates(
    db: Session,
    model: TwoTowerModel,
    rule_matcher: RuleMatcher,
    max_candidates: int = 10,
    top_k: int = 10,
    output_file: str = None
):
    """
    Recommend top K jobs for each candidate.
    
    Args:
        db: Database session
        model: Two-Tower model
        rule_matcher: Rule matcher instance
        max_candidates: Maximum number of candidates to process
        top_k: Number of top jobs to recommend per candidate
        output_file: Optional output file path
    """
    # Setup output with proper UTF-8 encoding
    if output_file:
        output_fp = open(output_file, 'w', encoding='utf-8', errors='replace')
        def output_print(*args, **kwargs):
            # Format arguments with UTF-8 safe encoding
            formatted_args = []
            for arg in args:
                if isinstance(arg, str):
                    # Ensure UTF-8
                    try:
                        arg.encode('utf-8')
                    except:
                        arg = arg.encode('utf-8', errors='replace').decode('utf-8')
                formatted_args.append(arg)
            # Write to file
            print(*formatted_args, **kwargs, file=output_fp, flush=True)
            # Also print to console (with encoding handling)
            try:
                print(*formatted_args, **kwargs, flush=True)
            except UnicodeEncodeError:
                # Fallback: encode to UTF-8 bytes then decode
                for arg in formatted_args:
                    try:
                        print(arg, end=' ', flush=True)
                    except:
                        print(repr(arg), end=' ', flush=True)
                print()  # Newline
    else:
        output_fp = None
        def output_print(*args, **kwargs):
            # Safe console print with UTF-8
            formatted_args = []
            for arg in args:
                if isinstance(arg, str):
                    try:
                        arg.encode('utf-8')
                    except:
                        arg = arg.encode('utf-8', errors='replace').decode('utf-8')
                formatted_args.append(arg)
            try:
                print(*formatted_args, **kwargs, flush=True)
            except UnicodeEncodeError:
                for arg in formatted_args:
                    try:
                        print(arg, end=' ', flush=True)
                    except:
                        print(repr(arg), end=' ', flush=True)
                print()
    
    output_print("=" * 100)
    output_print("JOB RECOMMENDATIONS FOR CANDIDATES")
    output_print("=" * 100)
    output_print(f"\nConfiguration:")
    output_print(f"  Max Candidates: {max_candidates}")
    output_print(f"  Top K Jobs per Candidate: {top_k}")
    output_print(f"  Model: VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
    if output_file:
        output_print(f"  Output File: {output_file}")
    
    # Load data
    output_print("\n" + "=" * 100)
    output_print("Loading data from database...")
    
    # Try Two-Tower repository first, fallback to Multi-Field
    repository = None
    all_candidates = None
    all_jobs = None
    use_two_tower = TWO_TOWER_AVAILABLE
    
    if use_two_tower:
        try:
            repository = TwoTowerRepository(db)
            all_candidates = repository.get_all_candidates()
            all_jobs = repository.get_all_jobs()
            output_print("  Using Two-Tower repository")
        except Exception as e:
            output_print(f"  Two-Tower repository failed: {e}")
            db.rollback()  # Rollback failed transaction
            output_print("  Falling back to Multi-Field repository")
            use_two_tower = False
    
    if not use_two_tower or all_candidates is None:
        try:
            repository = MultiFieldEmbeddingRepository(db)
            all_candidates = repository.get_all_candidate_multi_embeddings()
            all_jobs = repository.get_all_job_multi_embeddings()
            output_print("  Using Multi-Field repository")
        except Exception as e:
            db.rollback()  # Rollback failed transaction
            output_print(f"  ERROR: Multi-Field repository also failed: {e}")
            raise
    
    if not all_candidates or not all_jobs:
        output_print("ERROR: No candidates or jobs found in database!")
        if output_fp:
            output_fp.close()
        return
    
    output_print(f"  Loaded {len(all_candidates)} candidates")
    output_print(f"  Loaded {len(all_jobs)} jobs")
    
    # Limit candidates
    if len(all_candidates) > max_candidates:
        import random
        random.seed(42)
        candidates = random.sample(all_candidates, max_candidates)
        output_print(f"  Sampling {max_candidates} candidates for processing")
    else:
        candidates = all_candidates
    
    # Pre-compute job embeddings
    output_print(f"\nPre-computing job embeddings...")
    job_texts = []
    job_ids = []
    for job in all_jobs:
        job_text = build_job_text(job)
        if job_text:
            job_texts.append(job_text)
            job_ids.append(job.job_id)
    
    with torch.no_grad():
        job_embeddings = model.encode_jobs(job_texts)
        job_embeddings = job_embeddings.cpu().numpy()
    
    output_print(f"  Computed embeddings for {len(job_embeddings)} jobs")
    
    # Process each candidate
    output_print(f"\n{'='*100}")
    output_print("PROCESSING CANDIDATES")
    output_print(f"{'='*100}")
    
    all_recommendations = []
    
    with torch.no_grad():
        for candidate_idx, candidate in enumerate(candidates, 1):
            output_print(f"\n\n{'#'*100}")
            output_print(f"PROCESSING CANDIDATE {candidate_idx}/{len(candidates)}")
            output_print(f"{'#'*100}")
            
            candidate_text = build_candidate_text(candidate)
            if not candidate_text:
                output_print("  SKIP: Candidate has no text data")
                continue
            
            # Encode candidate
            candidate_emb = model.encode_candidates([candidate_text])[0]
            candidate_emb_np = candidate_emb.cpu().numpy()
            
            # Compute similarities with all jobs
            similarities = np.dot(job_embeddings, candidate_emb_np)
            
            # Get top K jobs
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Print candidate info
            print_candidate_summary(candidate, rule_matcher, output_print)
            
            # Process top K jobs
            candidate_recommendations = []
            for rank, job_idx in enumerate(top_indices, 1):
                job = all_jobs[job_idx]
                similarity = float(similarities[job_idx])
                
                # Rule matching
                candidate_skills = rule_matcher.extract_skills_from_text(candidate.skills or "")
                rule_result = rule_matcher.evaluate_match(
                    candidate_title=candidate.title or "",
                    candidate_skills=candidate_skills,
                    job_title=job.title or "",
                    job_requirements=job.requirement,
                    job_description=getattr(job, 'description', None)
                )
                
                # Print recommendation
                print_job_recommendation(job, similarity, rule_result, rank, rule_matcher, output_print)
                
                candidate_recommendations.append({
                    'job_id': job.job_id,
                    'similarity': similarity,
                    'rule_result': rule_result
                })
            
            all_recommendations.append({
                'candidate_id': candidate.candidate_id,
                'recommendations': candidate_recommendations
            })
    
    # Summary
    output_print(f"\n\n{'='*100}")
    output_print("SUMMARY")
    output_print(f"{'='*100}")
    output_print(f"Processed {len(all_recommendations)} candidates")
    output_print(f"Recommended {top_k} jobs per candidate")
    
    # Statistics
    ok_count = 0
    ng_count = 0
    for rec in all_recommendations:
        for job_rec in rec['recommendations']:
            if job_rec['rule_result'].get('final_status') == 'OK':
                ok_count += 1
            else:
                ng_count += 1
    
    output_print(f"\nRecommendation Statistics:")
    output_print(f"  Total Recommendations: {len(all_recommendations) * top_k}")
    output_print(f"  OK (Pass Rules): {ok_count}")
    output_print(f"  NG (Fail Rules): {ng_count}")
    output_print(f"  OK Rate: {ok_count / (ok_count + ng_count) * 100:.1f}%")
    
    if output_fp:
        output_fp.close()
        output_print(f"\n[INFO] Results saved to: {output_file}")
    
    return all_recommendations


def main():
    """Main function."""
    import argparse
    parser = argparse.ArgumentParser(description='Recommend jobs for candidates')
    parser.add_argument('--max-candidates', type=int, default=10, help='Maximum number of candidates to process')
    parser.add_argument('--top-k', type=int, default=10, help='Number of top jobs to recommend per candidate')
    parser.add_argument('--output', type=str, default='job_recommendations.txt', help='Output file path')
    parser.add_argument('--model-path', type=str, default='outputs_improved/best_model_improved.pt', help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    print("=" * 100)
    print("JOB RECOMMENDATION SYSTEM")
    print("=" * 100)
    
    # Load model
    print("\n1. Loading Two-Tower model...")
    try:
        model = load_two_tower_model(args.model_path)
        print(f"[OK] Model loaded from: {args.model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    # Initialize rule matcher
    print("\n2. Initializing Rule Matcher...")
    rule_matcher = RuleMatcher()
    print("[OK] Rule Matcher initialized")
    
    # Load database and process
    print("\n3. Processing candidates...")
    db = SessionLocal()
    try:
        recommendations = recommend_jobs_for_candidates(
            db=db,
            model=model,
            rule_matcher=rule_matcher,
            max_candidates=args.max_candidates,
            top_k=args.top_k,
            output_file=args.output
        )
        print(f"\n[OK] Completed! Processed {len(recommendations)} candidates")
    except Exception as e:
        print(f"[ERROR] Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == '__main__':
    main()

