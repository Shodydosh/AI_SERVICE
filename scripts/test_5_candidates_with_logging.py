"""Test matching for 5 candidates and log detailed results."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import numpy as np
from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.database.repository import EmbeddingRepository
from src.services.matching_service import MatchingService
from src.database.models import CandidateEmbedding, JobDescriptionEmbedding
from datetime import datetime

# Configure logging to file and console
log_filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def test_candidates_matching(num_candidates: int = 5, top_k: int = 5):
    """Test matching for specified number of candidates."""
    logger.info("=" * 100)
    logger.info("TEST MATCHING FOR 5 CANDIDATES")
    logger.info("=" * 100)
    logger.info("")
    logger.info(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Model: VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
    logger.info(f"Top K: {top_k} jobs per candidate")
    logger.info("")
    
    db: Session = SessionLocal()
    try:
        # Check if we have candidates
        candidate_count = db.query(CandidateEmbedding).count()
        jd_count = db.query(JobDescriptionEmbedding).count()
        
        logger.info(f"Database Status:")
        logger.info(f"  - Total Candidates: {candidate_count}")
        logger.info(f"  - Total Job Descriptions: {jd_count}")
        logger.info("")
        
        if candidate_count == 0:
            logger.error("❌ No candidates found in database. Please generate candidate embeddings first.")
            return
        
        if jd_count == 0:
            logger.error("❌ No job descriptions found in database. Please generate JD embeddings first.")
            return
        
        if candidate_count < num_candidates:
            logger.warning(f"⚠️  Only {candidate_count} candidates available, testing with {candidate_count} candidates")
            num_candidates = candidate_count
        
        # Get candidates
        candidates = db.query(CandidateEmbedding).limit(num_candidates).all()
        
        logger.info(f"Testing with {len(candidates)} candidates:")
        logger.info("")
        
        # Initialize matching service
        matching_service = MatchingService(db, use_faiss=False, use_reranking=True)
        
        # Test each candidate
        all_results = []
        
        for idx, candidate in enumerate(candidates, 1):
            logger.info("=" * 100)
            logger.info(f"CANDIDATE {idx}: {candidate.candidate_id}")
            logger.info("=" * 100)
            logger.info("")
            
            # Candidate information
            logger.info("Candidate Information:")
            logger.info(f"  ID: {candidate.candidate_id}")
            logger.info(f"  Name: {candidate.name or 'N/A'}")
            logger.info(f"  Email: {candidate.email or 'N/A'}")
            logger.info(f"  Skills: {candidate.skills[:200] if candidate.skills else 'N/A'}...")
            logger.info(f"  Experience: {candidate.experience[:200] if candidate.experience else 'N/A'}...")
            logger.info(f"  Education: {candidate.education[:200] if candidate.education else 'N/A'}...")
            logger.info(f"  Summary: {candidate.summary[:200] if candidate.summary else 'N/A'}...")
            logger.info("")
            
            # Find matching jobs
            logger.info(f"Finding top {top_k} matching jobs...")
            logger.info("")
            
            try:
                matches = matching_service.find_jobs_for_candidate(
                    candidate_id=candidate.candidate_id,
                    top_k=top_k
                )
                
                if not matches:
                    logger.warning(f"  ⚠️  No matches found for candidate {candidate.candidate_id}")
                    logger.info("")
                    continue
                
                logger.info(f"✅ Found {len(matches)} matching jobs:")
                logger.info("")
                
                candidate_results = {
                    'candidate_id': candidate.candidate_id,
                    'name': candidate.name,
                    'skills': candidate.skills,
                    'matches': []
                }
                
                for i, match in enumerate(matches, 1):
                    similarity = match.get('similarity_score', 0) * 100
                    
                    logger.info(f"  {i}. {match.get('title', 'N/A')}")
                    logger.info(f"     Similarity: {similarity:.2f}%")
                    logger.info(f"     Job ID: {match.get('job_id', 'N/A')}")
                    logger.info(f"     Company: {match.get('company', 'N/A')}")
                    logger.info(f"     Location: {match.get('location', 'N/A')}")
                    if match.get('description'):
                        logger.info(f"     Description: {match.get('description', '')[:150]}...")
                    logger.info("")
                    
                    candidate_results['matches'].append({
                        'rank': i,
                        'title': match.get('title'),
                        'similarity': similarity,
                        'job_id': match.get('job_id'),
                        'company': match.get('company'),
                        'location': match.get('location')
                    })
                
                all_results.append(candidate_results)
                
                # Summary for this candidate
                top_similarity = matches[0].get('similarity_score', 0) * 100 if matches else 0
                avg_similarity = np.mean([m.get('similarity_score', 0) * 100 for m in matches]) if matches else 0
                
                logger.info(f"Summary for Candidate {idx}:")
                logger.info(f"  Top Similarity: {top_similarity:.2f}%")
                logger.info(f"  Average Similarity: {avg_similarity:.2f}%")
                logger.info("")
                
            except Exception as e:
                logger.error(f"❌ Error finding matches for candidate {candidate.candidate_id}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                logger.info("")
        
        # Overall summary
        logger.info("=" * 100)
        logger.info("OVERALL SUMMARY")
        logger.info("=" * 100)
        logger.info("")
        
        if all_results:
            all_top_similarities = []
            all_avg_similarities = []
            
            for result in all_results:
                if result['matches']:
                    top_sim = result['matches'][0]['similarity']
                    avg_sim = np.mean([m['similarity'] for m in result['matches']])
                    all_top_similarities.append(top_sim)
                    all_avg_similarities.append(avg_sim)
            
            if all_top_similarities:
                logger.info(f"Total Candidates Tested: {len(all_results)}")
                logger.info(f"Average Top Similarity: {np.mean(all_top_similarities):.2f}%")
                logger.info(f"Average Similarity (Top {top_k}): {np.mean(all_avg_similarities):.2f}%")
                logger.info(f"Best Match: {max(all_top_similarities):.2f}%")
                logger.info(f"Worst Match: {min(all_top_similarities):.2f}%")
                logger.info("")
        
        logger.info("=" * 100)
        logger.info("✅ TEST COMPLETE")
        logger.info("=" * 100)
        logger.info(f"Results logged to: {log_filename}")
        logger.info("")
        
        return all_results
        
    except Exception as e:
        logger.error(f"❌ Error in test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    finally:
        db.close()


if __name__ == "__main__":
    test_candidates_matching(num_candidates=5, top_k=5)

