"""Test script for Two-Tower model - chỉ sử dụng precomputed embeddings từ database."""
import sys
import warnings
import logging
from pathlib import Path
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

from two_tower.model import TwoTowerModel

# Try to import database modules
try:
    from sqlalchemy.orm import Session
    from src.database.connection import SessionLocal
    from src.database.multi_field_repository import MultiFieldEmbeddingRepository
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
except Exception as e:
    DATABASE_AVAILABLE = False
    print(f"[WARNING] Database modules available but connection may fail: {e}")


def compute_combined_embedding(embeddings: dict) -> np.ndarray:
    """Compute combined embedding from multi-field embeddings."""
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


def test_two_tower_with_database(max_candidates: int = 5, top_k: int = 3):
    """Test Two-Tower model với precomputed embeddings từ database."""
    print("=" * 100)
    print("TEST TWO-TOWER MODEL (VỚI PRECOMPUTED EMBEDDINGS TỪ DATABASE)")
    print("=" * 100)
    
    # Show database config
    try:
        from config.settings import settings
        db_config = settings.get_database_config()
        print(f"\nDatabase Configuration:")
        print(f"  Host: {db_config['host']}")
        print(f"  Port: {db_config['port']}")
        print(f"  Database: {db_config['database']}")
        print(f"  User: {db_config['username']}")
    except Exception as e:
        print(f"[WARNING] Could not load database config: {e}")
    
    db = None
    try:
        print(f"\n[INFO] Attempting to connect to database...")
        db = SessionLocal()
        print(f"[OK] Database connection successful")
        
        repository = MultiFieldEmbeddingRepository(db)
        
        # Load candidates và jobs từ database
        all_candidates = repository.get_all_candidate_multi_embeddings()
        all_jobs = repository.get_all_job_multi_embeddings()
        
        if not all_candidates or not all_jobs:
            print("[WARNING] No candidates or jobs found in database")
            return None
        
        print(f"\nLoaded from database:")
        print(f"  Candidates: {len(all_candidates)}")
        print(f"  Jobs: {len(all_jobs)}")
        
        # Sample candidates
        if len(all_candidates) > max_candidates:
            import random
            random.seed(42)
            candidates = random.sample(all_candidates, max_candidates)
            print(f"  Sampling {max_candidates} candidates")
        else:
            candidates = all_candidates
        
        # Load job embeddings
        job_embeddings = []
        job_records = []
        job_texts = []
        
        for job in all_jobs:
            job_emb = {
                'title_embedding': job.title_embedding,
                'skills_embedding': job.skills_embedding,
                'requirement_embedding': job.requirement_embedding
            }
            combined_emb = compute_combined_embedding(job_emb)
            if combined_emb is not None:
                job_embeddings.append(combined_emb)
                job_records.append(job)
                # Build job text for display
                job_text = f"Title: {job.title or 'N/A'} | Skills: {job.skills or 'N/A'} | Requirements: {(job.requirement or 'N/A')[:100]}"
                job_texts.append(job_text)
        
        job_embeddings = np.array(job_embeddings)
        print(f"  Loaded {len(job_embeddings)} job embeddings")
        
        # Process candidates
        candidate_texts = []
        candidate_embeddings = []
        
        for candidate in candidates:
            candidate_emb = {
                'title_embedding': candidate.title_embedding,
                'skills_embedding': candidate.skills_embedding,
                'experience_embedding': candidate.experience_embedding
            }
            combined_emb = compute_combined_embedding(candidate_emb)
            if combined_emb is not None:
                candidate_embeddings.append(combined_emb)
                # Build candidate text for display
                candidate_text = f"Title: {candidate.title or 'N/A'} | Skills: {candidate.skills or 'N/A'} | Experience: {(candidate.experience or 'N/A')[:100]}"
                candidate_texts.append(candidate_text)
        
        candidate_embeddings = np.array(candidate_embeddings)
        print(f"  Loaded {len(candidate_embeddings)} candidate embeddings")
        
        # Compute similarities
        similarities = np.dot(job_embeddings, candidate_embeddings.T)  # [num_jobs, num_candidates]
        
        return {
            'candidate_texts': candidate_texts,
            'job_texts': job_texts,
            'similarities': similarities,
            'candidates': candidates,
            'jobs': job_records
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n[ERROR] Failed to load from database")
        print(f"Error: {error_msg}")
        
        # Provide specific debugging info
        if "connection refused" in error_msg.lower() or "could not connect" in error_msg.lower():
            print(f"\n[DEBUG] Connection refused - PostgreSQL may not be running")
            print(f"Solutions:")
            print(f"  1. Check if PostgreSQL service is running:")
            print(f"     Windows: Get-Service -Name postgresql*")
            print(f"     Or check Services -> PostgreSQL")
            print(f"  2. Start PostgreSQL service:")
            print(f"     Windows: net start postgresql-x64-XX")
            print(f"  3. Verify connection settings in config/settings.py")
            print(f"  4. Run diagnostic: python scripts/check_postgresql_setup.py")
        elif "password authentication failed" in error_msg.lower():
            print(f"\n[DEBUG] Password authentication failed")
            print(f"Solutions:")
            print(f"  1. Check password in config/settings.py or .env file")
            print(f"  2. Verify database user exists")
        elif "does not exist" in error_msg.lower():
            print(f"\n[DEBUG] Database or table does not exist")
            print(f"Solutions:")
            print(f"  1. Create database: CREATE DATABASE job_recommendation_db;")
            print(f"  2. Run: python scripts/init_multi_field_tables.py")
            print(f"  3. Process data: python scripts/process_multi_field_embeddings.py")
        else:
            print(f"\n[DEBUG] Unknown error - check PostgreSQL logs")
            import traceback
            traceback.print_exc()
        
        return None
    finally:
        if db:
            db.close()


def test_two_tower_standalone():
    """Test Two-Tower model - chỉ sử dụng precomputed embeddings từ database."""
    print("=" * 100)
    print("TEST TWO-TOWER MODEL (CHỈ SỬ DỤNG PRECOMPUTED EMBEDDINGS TỪ DATABASE)")
    print("=" * 100)
    
    # Check database availability
    if not DATABASE_AVAILABLE:
        print("\n[ERROR] Database modules not available!")
        print("Please install required packages:")
        print("  pip install psycopg2-binary sqlalchemy")
        return
    
    # Load from database
    print("\n[INFO] Loading precomputed embeddings from database...")
    data = test_two_tower_with_database(max_candidates=5, top_k=3)
    
    if data is None:
        print("\n" + "=" * 100)
        print("[ERROR] Failed to load data from database!")
        print("=" * 100)
        print("\nPlease ensure:")
        print("  1. PostgreSQL is running")
        print("  2. Database connection is configured correctly in config/settings.py")
        print("  3. Database exists: job_recommendation_db")
        print("  4. Tables exist (run: python scripts/init_multi_field_tables.py)")
        print("  5. Data has been processed (run: python scripts/process_multi_field_embeddings.py)")
        print("\nTo debug connection issues, run:")
        print("  python scripts/check_postgresql_setup.py")
        print("  python test_db_connection.py")
        return
    
    print("\n[OK] Successfully loaded precomputed embeddings from database")
    
    candidate_texts = data['candidate_texts']
    job_texts = data['job_texts']
    similarities = data['similarities']
    
    print(f"\n[INFO] Using precomputed embeddings from database")
    print(f"  Similarity matrix shape: {similarities.shape}")
    
    # Get top recommendations for each candidate
    print(f"\n5. Top 3 Job Recommendations for each Candidate:")
    print("=" * 100)
    
    for candidate_idx, candidate_text in enumerate(candidate_texts):
        print(f"\n{'#'*100}")
        print(f"CANDIDATE {candidate_idx + 1}/{len(candidate_texts)}")
        print(f"{'#'*100}")
        
        # Show candidate info
        candidates_list = data.get('candidates')
        if candidates_list and candidate_idx < len(candidates_list) and candidates_list[candidate_idx]:
            candidate = candidates_list[candidate_idx]
            print(f"ID: {candidate.candidate_id}")
            print(f"Name: {candidate.name or 'N/A'}")
            print(f"Title: {candidate.title or 'N/A'}")
        else:
            print(f"Candidate Text: {candidate_text[:150]}...")
        
        # Get similarities for this candidate
        candidate_similarities = similarities[:, candidate_idx]  # [num_jobs]
        
        # Get top 3
        top_indices = np.argsort(candidate_similarities)[::-1][:3]
        
        print(f"\nTop 3 Job Recommendations:")
        jobs_list = data.get('jobs')
        for rank, job_idx in enumerate(top_indices, 1):
            similarity_score = float(candidate_similarities[job_idx])
            
            # Show job info
            if jobs_list and job_idx < len(jobs_list) and jobs_list[job_idx]:
                job = jobs_list[job_idx]
                print(f"\n  {rank}. Job ID: {job.job_id} - Similarity: {similarity_score:.4f}")
                print(f"     Title: {job.title or 'N/A'}")
                print(f"     Company: {job.company or 'N/A'}")
                print(f"     Location: {job.location or 'N/A'}")
            else:
                print(f"\n  {rank}. Job Index: {job_idx} - Similarity: {similarity_score:.4f}")
                print(f"     (Job record not found)")
    
    # Statistics
    print(f"\n\n{'='*100}")
    print("STATISTICS")
    print(f"{'='*100}")
    print(f"  Total similarity computations: {similarities.size}")
    print(f"  Average similarity: {np.mean(similarities):.4f}")
    print(f"  Max similarity: {np.max(similarities):.4f}")
    print(f"  Min similarity: {np.min(similarities):.4f}")
    print(f"  Median similarity: {np.median(similarities):.4f}")
    
    print(f"\n{'='*100}")
    print("TEST COMPLETED SUCCESSFULLY")
    print(f"{'='*100}")
    print("Note: Used precomputed embeddings from database.")


if __name__ == '__main__':
    test_two_tower_standalone()

