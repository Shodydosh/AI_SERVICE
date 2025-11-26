"""Test Vietnamese embedding quality with existing JDs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pandas as pd
from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.database.repository import EmbeddingRepository
from src.services.matching_service import MatchingService
from src.embeddings.generator import EmbeddingGenerator
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_vietnamese_embedding():
    """Test Vietnamese embedding quality with sample candidate text."""
    logger.info("=" * 80)
    logger.info("TESTING VIETNAMESE EMBEDDING QUALITY")
    logger.info("=" * 80)
    
    db: Session = SessionLocal()
    try:
        repository = EmbeddingRepository(db)
        jd_embeddings = repository.get_all_jd_embeddings()
        
        if len(jd_embeddings) < 10:
            logger.warning(f"Only {len(jd_embeddings)} JDs available. Need at least 10 for testing.")
            return
        
        logger.info(f"Testing with {len(jd_embeddings)} job descriptions")
        logger.info("")
        
        # Test with Vietnamese candidate text
        test_candidates = [
            {
                "name": "Nguyen Van A",
                "skills": "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch",
                "experience": "5 năm kinh nghiệm phát triển phần mềm, chuyên về AI và Machine Learning",
                "summary": "Lập trình viên Python với kinh nghiệm trong phát triển ứng dụng AI"
            },
            {
                "name": "Tran Thi B",
                "skills": "Java, Spring Boot, MySQL, REST API",
                "experience": "3 năm phát triển backend với Java và Spring Framework",
                "summary": "Backend developer chuyên về Java và microservices"
            },
            {
                "name": "Le Van C",
                "skills": "React, JavaScript, Node.js, MongoDB",
                "experience": "4 năm phát triển full-stack web applications",
                "summary": "Full-stack developer với kinh nghiệm React và Node.js"
            }
        ]
        
        matching_service = MatchingService(db, use_faiss=False)
        
        for idx, candidate in enumerate(test_candidates, 1):
            logger.info("=" * 80)
            logger.info(f"TEST CANDIDATE {idx}: {candidate['name']}")
            logger.info("=" * 80)
            logger.info(f"Skills: {candidate['skills']}")
            logger.info(f"Experience: {candidate['experience']}")
            logger.info(f"Summary: {candidate['summary']}")
            logger.info("")
            
            # Generate embedding for candidate text
            candidate_text = matching_service.combine_candidate_fields(
                name=candidate['name'],
                skills=candidate['skills'],
                experience=candidate['experience'],
                summary=candidate['summary']
            )
            
            logger.info(f"Combined text: {candidate_text[:200]}...")
            logger.info("")
            
            # Find top 5 matching jobs
            matches = matching_service.find_jobs_for_candidate_text(
                candidate_text=candidate_text,
                top_k=5
            )
            
            logger.info(f"TOP 5 MATCHING JOBS:")
            logger.info("-" * 80)
            for i, match in enumerate(matches, 1):
                similarity_pct = match['similarity_score'] * 100
                logger.info(f"{i}. {match['title']}")
                logger.info(f"   Similarity: {similarity_pct:.2f}%")
                logger.info(f"   Job ID: {match['job_id']}")
                if match.get('company'):
                    logger.info(f"   Company: {match['company']}")
                if match.get('location'):
                    logger.info(f"   Location: {match['location']}")
                logger.info("")
        
        logger.info("=" * 80)
        logger.info("✓ VIETNAMESE EMBEDDING TEST COMPLETE")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Model: paraphrase-multilingual-mpnet-base-v2 (768 dimensions)")
        logger.info("This model is optimized for Vietnamese text and should provide")
        logger.info("better matching results compared to English-only models.")
        
    except Exception as e:
        logger.error(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    test_vietnamese_embedding()

