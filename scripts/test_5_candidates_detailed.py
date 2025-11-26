"""Test 5 candidates with detailed output: embedded fields and top 5 JD matches."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.services.matching_service import MatchingService
from src.database.repository import EmbeddingRepository
from src.data_processing.candidate_processor import CandidateProcessor
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_5_candidates_detailed():
    """Test 5 candidates and show embedded fields + top 5 JD matches."""
    logger.info("=" * 80)
    logger.info("TESTING 5 CANDIDATES WITH DETAILED OUTPUT")
    logger.info("=" * 80)
    
    db: Session = SessionLocal()
    try:
        repository = EmbeddingRepository(db)
        matching_service = MatchingService(db, use_faiss=False)
        processor = CandidateProcessor()
        
        # Load full candidate dataset
        candidate_file = 'data/processed/candidates_dataset.csv'
        processor.load_from_csv(candidate_file)
        candidate_df = pd.read_csv(candidate_file, low_memory=False)
        
        # Get all candidates from database
        all_candidates = repository.get_all_candidate_embeddings()
        
        if len(all_candidates) == 0:
            logger.error("No candidates found in database. Please run embedding generation first.")
            return
        
        # Select 5 candidates: first 2 + 3 random with good data
        import random
        first_2_ids = [c.candidate_id for c in all_candidates[:2]]
        
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
            return score
        
        # Find candidates with good data
        good_candidates = []
        all_other_candidates = []
        for idx, row in candidate_df.iterrows():
            cv_id = str(row.get('cv_id', ''))
            if cv_id and cv_id not in first_2_ids:
                score = score_candidate(row)
                if score >= 3:
                    good_candidates.append((cv_id, score))
                all_other_candidates.append(cv_id)
        
        # Select 3 random candidates
        if len(good_candidates) >= 3:
            random_candidates_with_data = random.sample(good_candidates, 3)
            random_ids = [cv_id for cv_id, _ in random_candidates_with_data]
        elif len(good_candidates) > 0:
            random_ids = [cv_id for cv_id, _ in good_candidates]
            remaining_needed = 3 - len(random_ids)
            other_available = [c for c in all_other_candidates if c not in random_ids]
            if remaining_needed > 0 and len(other_available) >= remaining_needed:
                random_ids.extend(random.sample(other_available, remaining_needed))
        else:
            random_ids = random.sample(all_other_candidates, min(3, len(all_other_candidates))) if len(all_other_candidates) >= 3 else all_other_candidates[:3]
        
        # Get all test candidate IDs
        test_candidate_ids = first_2_ids + random_ids[:3]
        test_candidates = [c for c in all_candidates if c.candidate_id in test_candidate_ids]
        
        # Limit to 5 candidates
        test_candidates = test_candidates[:5]
        
        logger.info(f"\nTesting with {len(test_candidates)} candidates: {[c.candidate_id for c in test_candidates]}\n")
        
        # Process each candidate
        for candidate_idx, candidate in enumerate(test_candidates, 1):
            candidate_id = candidate.candidate_id
            logger.info("=" * 80)
            logger.info(f"CANDIDATE {candidate_idx}: {candidate_id}")
            logger.info("=" * 80)
            logger.info(f"Name: {candidate.name or 'N/A'}")
            logger.info(f"Email: {candidate.email or 'N/A'}")
            logger.info("")
            
            # Get candidate row from CSV
            candidate_row = candidate_df[candidate_df['cv_id'] == int(candidate_id)] if 'cv_id' in candidate_df.columns else None
            if len(candidate_row) == 0:
                # Try string match
                candidate_row = candidate_df[candidate_df['cv_id'].astype(str) == str(candidate_id)]
            
            if len(candidate_row) > 0:
                row = candidate_row.iloc[0]
                
                # Show embedded fields
                logger.info("EMBEDDED FIELDS:")
                logger.info("-" * 80)
                field_texts = processor.get_field_texts(row)
                
                for field, text in field_texts.items():
                    # Truncate long text for display
                    display_text = text[:200] + "..." if len(text) > 200 else text
                    logger.info(f"  • {field.replace('_', ' ').title()}: {display_text}")
                
                # Show combined text (what was actually embedded)
                combined_text = processor.get_combined_text(row)
                logger.info("")
                logger.info("COMBINED TEXT FOR EMBEDDING:")
                logger.info("-" * 80)
                logger.info(f"  {combined_text[:500]}..." if len(combined_text) > 500 else f"  {combined_text}")
                logger.info("")
            else:
                logger.warning(f"Could not find candidate {candidate_id} in CSV file")
                logger.info("")
            
            # Find top 5 jobs
            logger.info("TOP 5 JOB MATCHES:")
            logger.info("-" * 80)
            matches = matching_service.find_jobs_for_candidate(
                candidate_id=candidate_id,
                top_k=5,
                use_faiss=False
            )
            
            for i, match in enumerate(matches, 1):
                logger.info(f"\n  {i}. {match['title']} (Similarity: {match['similarity_score']:.4f} = {match['similarity_score']*100:.2f}%)")
                logger.info(f"     Job ID: {match['job_id']}")
                if match.get('company'):
                    logger.info(f"     Company: {match['company']}")
                if match.get('location'):
                    logger.info(f"     Location: {match['location']}")
                if match.get('description'):
                    desc = match['description'][:200] + "..." if len(match['description']) > 200 else match['description']
                    logger.info(f"     Description: {desc}")
                if match.get('requirements'):
                    req = match['requirements'][:200] + "..." if len(match['requirements']) > 200 else match['requirements']
                    logger.info(f"     Requirements: {req}")
            
            logger.info("")
            logger.info("")
        
        logger.info("=" * 80)
        logger.info("✓ TEST COMPLETE")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    test_5_candidates_detailed()

