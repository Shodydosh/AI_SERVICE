"""Test Two-Tower model với encoding tiếng Việt đúng."""
import sys
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


def safe_print(text, max_length=500):
    """Safely print text with UTF-8 encoding."""
    if text is None:
        return 'N/A'
    text_str = str(text)
    # Ensure UTF-8
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
        # Try normal print
        print(*args, **kwargs, flush=True)
    except UnicodeEncodeError:
        # Fallback: print each arg separately
        for arg in args:
            try:
                if isinstance(arg, str):
                    # Ensure it's valid UTF-8
                    arg.encode('utf-8')
                print(arg, end=' ', flush=True)
            except:
                # Last resort: print repr
                print(repr(arg), end=' ', flush=True)
        print()  # Newline


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


def test_two_tower_matching(
    max_candidates: int = 5,
    top_k: int = 10,
    output_file: str = "two_tower_test_output.txt"
):
    """
    Test Two-Tower matching với encoding tiếng Việt đúng.
    
    Args:
        max_candidates: Số lượng candidates để test
        top_k: Số lượng jobs đề xuất cho mỗi candidate
        output_file: File output (UTF-8 encoded)
    """
    print_utf8("=" * 100)
    print_utf8("TEST TWO-TOWER MATCHING VỚI TIẾNG VIỆT")
    print_utf8("=" * 100)
    
    # Setup output file
    output_fp = open(output_file, 'w', encoding='utf-8', errors='replace')
    
    def output_print(*args, **kwargs):
        """Print to both file and console with UTF-8."""
        # Format for file
        formatted = []
        for arg in args:
            if isinstance(arg, str):
                try:
                    arg.encode('utf-8')
                except:
                    arg = arg.encode('utf-8', errors='replace').decode('utf-8')
            formatted.append(arg)
        print(*formatted, **kwargs, file=output_fp, flush=True)
        # Print to console
        print_utf8(*formatted, **kwargs)
    
    output_print(f"\nConfiguration:")
    output_print(f"  Max Candidates: {max_candidates}")
    output_print(f"  Top K Jobs: {top_k}")
    output_print(f"  Output File: {output_file}")
    
    # Load model
    output_print("\n1. Loading Two-Tower model...")
    try:
        model = load_two_tower_model()
        output_print("[OK] Model loaded")
    except Exception as e:
        output_print(f"[ERROR] Failed to load model: {e}")
        output_fp.close()
        return
    
    # Initialize rule matcher
    output_print("\n2. Initializing Rule Matcher...")
    rule_matcher = RuleMatcher()
    output_print("[OK] Rule Matcher initialized")
    
    # Load data
    output_print("\n3. Loading data from database...")
    db = SessionLocal()
    try:
        use_two_tower = TWO_TOWER_AVAILABLE
        repository = None
        all_candidates = None
        all_jobs = None
        
        if use_two_tower:
            try:
                repository = TwoTowerRepository(db)
                all_candidates = repository.get_all_candidates()
                all_jobs = repository.get_all_jobs()
                output_print("  Using Two-Tower repository")
            except Exception as e:
                output_print(f"  Two-Tower repository failed: {e}")
                db.rollback()
                use_two_tower = False
        
        if not use_two_tower or all_candidates is None:
            repository = MultiFieldEmbeddingRepository(db)
            all_candidates = repository.get_all_candidate_multi_embeddings()
            all_jobs = repository.get_all_job_multi_embeddings()
            output_print("  Using Multi-Field repository")
        
        if not all_candidates or not all_jobs:
            output_print("[ERROR] No candidates or jobs found!")
            output_fp.close()
            return
        
        output_print(f"  Loaded {len(all_candidates)} candidates")
        output_print(f"  Loaded {len(all_jobs)} jobs")
        
        # Sample candidates
        if len(all_candidates) > max_candidates:
            import random
            random.seed(42)
            candidates = random.sample(all_candidates, max_candidates)
            output_print(f"  Sampling {max_candidates} candidates")
        else:
            candidates = all_candidates
        
        # Pre-compute job embeddings
        output_print(f"\n4. Pre-computing job embeddings...")
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
        
        with torch.no_grad():
            for candidate_idx, candidate in enumerate(candidates, 1):
                output_print(f"\n\n{'#'*100}")
                output_print(f"CANDIDATE {candidate_idx}/{len(candidates)}")
                output_print(f"{'#'*100}")
                
                candidate_text = build_candidate_text(candidate)
                if not candidate_text:
                    output_print("  SKIP: No text data")
                    continue
                
                # Encode candidate
                candidate_emb = model.encode_candidates([candidate_text])[0]
                candidate_emb_np = candidate_emb.cpu().numpy()
                
                # Compute similarities
                similarities = np.dot(job_embeddings, candidate_emb_np)
                
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
                
                # Print top recommendations
                output_print(f"\n{'='*80}")
                output_print(f"TOP {top_k} JOB RECOMMENDATIONS")
                output_print(f"{'='*80}")
                
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
                    output_print(f"\n{'-'*80}")
                    output_print(f"RECOMMENDATION #{rank}")
                    output_print(f"{'-'*80}")
                    output_print(f"Job ID: {job.job_id}")
                    output_print(f"Title: {safe_print(job.title)}")
                    output_print(f"Company: {safe_print(job.company)}")
                    output_print(f"Location: {safe_print(job.location)}")
                    output_print(f"\nMatching Scores:")
                    output_print(f"  Two-Tower Similarity: {similarity:.4f}")
                    output_print(f"  Rule 1 - Title Score: {rule_result.get('rule1', {}).get('score', 0):.4f} "
                               f"(Status: {rule_result.get('rule1', {}).get('status', 'UNKNOWN')})")
                    output_print(f"  Rule 2 - Skill Score: {rule_result.get('rule2', {}).get('score', 0):.4f} "
                               f"(Status: {rule_result.get('rule2', {}).get('status', 'UNKNOWN')})")
                    output_print(f"  Final Decision: {rule_result.get('final_status', 'UNKNOWN')}")
                    output_print(f"  Reason: {rule_result.get('reason', 'N/A')}")
                    
                    if job.requirement:
                        output_print(f"\nJob Requirements: {safe_print(job.requirement, max_length=400)}")
        
        output_print(f"\n\n{'='*100}")
        output_print("TEST COMPLETED")
        output_print(f"{'='*100}")
        output_print(f"Processed {len(candidates)} candidates")
        output_print(f"Recommended {top_k} jobs per candidate")
        output_print(f"Results saved to: {output_file}")
        
    except Exception as e:
        output_print(f"[ERROR] Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()
        output_fp.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test Two-Tower with Vietnamese encoding')
    parser.add_argument('--max-candidates', type=int, default=5, help='Max candidates to test')
    parser.add_argument('--top-k', type=int, default=10, help='Top K jobs per candidate')
    parser.add_argument('--output', type=str, default='two_tower_test_output.txt', help='Output file')
    
    args = parser.parse_args()
    
    test_two_tower_matching(
        max_candidates=args.max_candidates,
        top_k=args.top_k,
        output_file=args.output
    )

