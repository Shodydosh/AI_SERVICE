"""Test Two-Tower model using precomputed embeddings from database."""
import sys
import io
import os
import warnings
import logging
from pathlib import Path
from typing import List, Dict, Any
import torch
import numpy as np

# Fix Windows console encoding for Vietnamese
if sys.platform == 'win32':
    try:
        os.system('chcp 65001 >nul 2>&1')
    except:
        pass
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding='utf-8',
            errors='replace',
            line_buffering=True
        )
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer,
            encoding='utf-8',
            errors='replace',
            line_buffering=True
        )

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
from src.database.multi_field_repository import MultiFieldEmbeddingRepository

logger = logging.getLogger(__name__)


def safe_print(text, max_length=500):
    """Safely print text with UTF-8 encoding."""
    if text is None:
        return 'N/A'
    text_str = str(text)
    if isinstance(text_str, bytes):
        try:
            text_str = text_str.decode('utf-8')
        except:
            text_str = text_str.decode('utf-8', errors='replace')
    if len(text_str) > max_length:
        return text_str[:max_length] + f"... (truncated, total: {len(text_str)})"
    return text_str


def print_utf8(*args, **kwargs):
    """Print with UTF-8 encoding support."""
    try:
        print(*args, **kwargs, flush=True)
    except UnicodeEncodeError:
        for arg in args:
            try:
                if isinstance(arg, str):
                    arg.encode('utf-8')
                print(arg, end=' ', flush=True)
            except:
                print(repr(arg), end=' ', flush=True)
        print()


def compute_combined_embedding(embeddings: Dict[str, List[float]]) -> np.ndarray:
    """
    Compute combined embedding from multi-field embeddings.
    Simple average of title, skills, and experience/requirement embeddings.
    """
    title_emb = np.array(embeddings.get('title_embedding', []))
    skills_emb = np.array(embeddings.get('skills_embedding', []))
    exp_emb = np.array(embeddings.get('experience_embedding', []) or 
                      embeddings.get('requirement_embedding', []))
    
    if len(title_emb) == 0:
        return None
    
    # Average the three embeddings
    embs = [title_emb]
    if len(skills_emb) > 0:
        embs.append(skills_emb)
    if len(exp_emb) > 0:
        embs.append(exp_emb)
    
    combined = np.mean(embs, axis=0)
    # Normalize
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm
    return combined


def test_two_tower_precomputed(
    max_candidates: int = 5,
    top_k: int = 10,
    output_file: str = "two_tower_precomputed_test.txt"
):
    """
    Test Two-Tower matching using precomputed embeddings from database.
    
    Args:
        max_candidates: Số lượng candidates để test
        top_k: Số lượng jobs đề xuất cho mỗi candidate
        output_file: File output (UTF-8 encoded)
    """
    print_utf8("=" * 100)
    print_utf8("TEST TWO-TOWER MATCHING VỚI PRECOMPUTED EMBEDDINGS")
    print_utf8("=" * 100)
    
    # Setup output file
    output_fp = open(output_file, 'w', encoding='utf-8', errors='replace')
    
    def output_print(*args, **kwargs):
        """Print to both file and console with UTF-8."""
        formatted = []
        for arg in args:
            if isinstance(arg, str):
                try:
                    arg.encode('utf-8')
                except:
                    arg = arg.encode('utf-8', errors='replace').decode('utf-8')
            formatted.append(arg)
        print(*formatted, **kwargs, file=output_fp, flush=True)
        print_utf8(*formatted, **kwargs)
    
    output_print(f"\nConfiguration:")
    output_print(f"  Max Candidates: {max_candidates}")
    output_print(f"  Top K Jobs: {top_k}")
    output_print(f"  Output File: {output_file}")
    output_print(f"  Using: Precomputed embeddings from database")
    
    # Initialize rule matcher
    output_print("\n1. Initializing Rule Matcher...")
    rule_matcher = RuleMatcher()
    output_print("[OK] Rule Matcher initialized")
    
    # Load data from database
    output_print("\n2. Loading precomputed embeddings from database...")
    db = SessionLocal()
    try:
        repository = MultiFieldEmbeddingRepository(db)
        all_candidates = repository.get_all_candidate_multi_embeddings()
        all_jobs = repository.get_all_job_multi_embeddings()
        
        if not all_candidates or not all_jobs:
            output_print("[ERROR] No candidates or jobs found!")
            output_fp.close()
            return
        
        output_print(f"  Loaded {len(all_candidates)} candidates with embeddings")
        output_print(f"  Loaded {len(all_jobs)} jobs with embeddings")
        
        # Sample candidates
        if len(all_candidates) > max_candidates:
            import random
            random.seed(42)
            candidates = random.sample(all_candidates, max_candidates)
            output_print(f"  Sampling {max_candidates} candidates")
        else:
            candidates = all_candidates
        
        # Pre-compute job embeddings from database
        output_print(f"\n3. Loading job embeddings from database...")
        job_embeddings = []
        job_records = []
        job_ids = []
        
        for job in all_jobs:
            # Get embeddings from database
            job_emb = {
                'title_embedding': job.title_embedding,
                'skills_embedding': job.skills_embedding,
                'requirement_embedding': job.requirement_embedding
            }
            
            # Compute combined embedding
            combined_emb = compute_combined_embedding(job_emb)
            if combined_emb is not None:
                job_embeddings.append(combined_emb)
                job_records.append(job)
                job_ids.append(job.job_id)
        
        job_embeddings = np.array(job_embeddings)  # [num_jobs, embedding_dim]
        output_print(f"  Loaded {len(job_embeddings)} job embeddings from database")
        output_print(f"  Embedding dimension: {job_embeddings.shape[1]}")
        
        # Process each candidate
        output_print(f"\n{'='*100}")
        output_print("PROCESSING CANDIDATES")
        output_print(f"{'='*100}")
        
        for candidate_idx, candidate in enumerate(candidates, 1):
            output_print(f"\n\n{'#'*100}")
            output_print(f"CANDIDATE {candidate_idx}/{len(candidates)}")
            output_print(f"{'#'*100}")
            
            # Get candidate embeddings from database
            candidate_emb = {
                'title_embedding': candidate.title_embedding,
                'skills_embedding': candidate.skills_embedding,
                'experience_embedding': candidate.experience_embedding
            }
            
            # Compute combined embedding
            combined_candidate_emb = compute_combined_embedding(candidate_emb)
            if combined_candidate_emb is None:
                output_print("  SKIP: No embeddings found in database")
                continue
            
            # Compute cosine similarity with all jobs
            similarities = np.dot(job_embeddings, combined_candidate_emb)  # [num_jobs]
            
            # Get top K
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Print candidate info
            output_print(f"\n{'='*80}")
            output_print(f"CANDIDATE INFO")
            output_print(f"{'='*80}")
            output_print(f"ID: {candidate.candidate_id}")
            output_print(f"Name: {safe_print(candidate.name)}")
            output_print(f"Title: {safe_print(candidate.title)}")
            if candidate.skills:
                try:
                    skills_list = rule_matcher.extract_skills_from_text(candidate.skills)
                    output_print(f"Skills ({len(skills_list)}): {', '.join(skills_list[:10])}")
                except:
                    output_print(f"Skills: {safe_print(candidate.skills, max_length=300)}")
            output_print(f"Experience: {safe_print(candidate.experience, max_length=400)}")
            output_print(f"\nEmbedding Info:")
            output_print(f"  Title embedding: {len(candidate.title_embedding) if candidate.title_embedding else 0} dims")
            output_print(f"  Skills embedding: {len(candidate.skills_embedding) if candidate.skills_embedding else 0} dims")
            output_print(f"  Experience embedding: {len(candidate.experience_embedding) if candidate.experience_embedding else 0} dims")
            
            # Print top recommendations
            output_print(f"\n{'='*80}")
            output_print(f"TOP {top_k} JOB RECOMMENDATIONS (Using Precomputed Embeddings)")
            output_print(f"{'='*80}")
            
            for rank, job_idx in enumerate(top_indices, 1):
                job = job_records[job_idx]
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
                output_print(f"\n{'-'*80}")
                output_print(f"RECOMMENDATION #{rank}")
                output_print(f"{'-'*80}")
                output_print(f"Job ID: {job.job_id}")
                output_print(f"Title: {safe_print(job.title)}")
                output_print(f"Company: {safe_print(job.company)}")
                output_print(f"Location: {safe_print(job.location)}")
                output_print(f"\nMatching Scores:")
                output_print(f"  Two-Tower Similarity (Precomputed): {similarity:.4f}")
                output_print(f"  Rule 1 - Title Score: {rule_result.get('rule1', {}).get('score', 0):.4f} "
                           f"(Status: {rule_result.get('rule1', {}).get('status', 'UNKNOWN')})")
                output_print(f"  Rule 2 - Skill Score: {rule_result.get('rule2', {}).get('score', 0):.4f} "
                           f"(Status: {rule_result.get('rule2', {}).get('status', 'UNKNOWN')})")
                output_print(f"  Final Decision: {rule_result.get('final_status', 'UNKNOWN')}")
                output_print(f"  Reason: {rule_result.get('reason', 'N/A')}")
                
                # Embedding info
                output_print(f"\nEmbedding Info:")
                output_print(f"  Title embedding: {len(job.title_embedding) if job.title_embedding else 0} dims")
                output_print(f"  Skills embedding: {len(job.skills_embedding) if job.skills_embedding else 0} dims")
                output_print(f"  Requirement embedding: {len(job.requirement_embedding) if job.requirement_embedding else 0} dims")
                
                if job.requirement:
                    output_print(f"\nJob Requirements: {safe_print(job.requirement, max_length=400)}")
        
        # Statistics
        output_print(f"\n\n{'='*100}")
        output_print("STATISTICS")
        output_print(f"{'='*100}")
        
        # Compute average similarity
        all_similarities = []
        for candidate in candidates:
            candidate_emb = {
                'title_embedding': candidate.title_embedding,
                'skills_embedding': candidate.skills_embedding,
                'experience_embedding': candidate.experience_embedding
            }
            combined_emb = compute_combined_embedding(candidate_emb)
            if combined_emb is not None:
                similarities = np.dot(job_embeddings, combined_emb)
                all_similarities.extend(similarities.tolist())
        
        if all_similarities:
            output_print(f"  Total similarity computations: {len(all_similarities)}")
            output_print(f"  Average similarity: {np.mean(all_similarities):.4f}")
            output_print(f"  Max similarity: {np.max(all_similarities):.4f}")
            output_print(f"  Min similarity: {np.min(all_similarities):.4f}")
            output_print(f"  Median similarity: {np.median(all_similarities):.4f}")
        
        output_print(f"\n{'='*100}")
        output_print("TEST COMPLETED")
        output_print(f"{'='*100}")
        output_print(f"Processed {len(candidates)} candidates")
        output_print(f"Recommended {top_k} jobs per candidate")
        output_print(f"Results saved to: {output_file}")
        output_print(f"\nNote: All embeddings were loaded from database (precomputed)")
        
    except Exception as e:
        output_print(f"[ERROR] Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()
        output_fp.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test Two-Tower with precomputed embeddings')
    parser.add_argument('--max-candidates', type=int, default=5, help='Max candidates to test')
    parser.add_argument('--top-k', type=int, default=10, help='Top K jobs per candidate')
    parser.add_argument('--output', type=str, default='two_tower_precomputed_test.txt', help='Output file')
    
    args = parser.parse_args()
    
    test_two_tower_precomputed(
        max_candidates=args.max_candidates,
        top_k=args.top_k,
        output_file=args.output
    )

