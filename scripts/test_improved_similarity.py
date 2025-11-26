"""Test improved similarity with new weights and text combination."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.database.repository import EmbeddingRepository
from src.embeddings.weighted_embedding import WeightedEmbeddingGenerator
from src.data_processing.jd_processor import JDProcessor
from src.data_processing.candidate_processor import CandidateProcessor
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))


def test_improved_similarity():
    """Test similarity with improved weights and text combination."""
    logger.info("=" * 80)
    logger.info("TESTING IMPROVED SIMILARITY WITH NEW WEIGHTS")
    logger.info("=" * 80)
    
    # Test candidates
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
        }
    ]
    
    # Test JDs
    test_jds = [
        {
            "title": "AI Engineer",
            "description": "Phát triển và triển khai các giải pháp AI và Machine Learning",
            "requirements": "Kinh nghiệm với Python, TensorFlow, PyTorch, Machine Learning"
        },
        {
            "title": "Java Developer",
            "description": "Phát triển ứng dụng backend với Java và Spring Framework",
            "requirements": "Kinh nghiệm với Java, Spring Boot, MySQL, REST API"
        }
    ]
    
    generator = WeightedEmbeddingGenerator()
    jd_processor = JDProcessor()
    candidate_processor = CandidateProcessor()
    
    logger.info("\nGenerating embeddings with improved weights...")
    logger.info("")
    
    # Generate JD embeddings
    jd_embeddings = []
    for jd in test_jds:
        import pandas as pd
        jd_row = pd.Series(jd)
        field_texts = jd_processor.get_field_texts(jd_row)
        embedding = generator.generate_weighted_embedding(
            field_texts=field_texts,
            weights=WeightedEmbeddingGenerator.DEFAULT_JD_WEIGHTS,
            method="repetition"
        )
        jd_embeddings.append({
            "jd": jd,
            "embedding": embedding
        })
        logger.info(f"Generated JD embedding: {jd['title']}")
    
    logger.info("")
    
    # Generate candidate embeddings and compare
    for candidate in test_candidates:
        logger.info("=" * 80)
        logger.info(f"CANDIDATE: {candidate['name']}")
        logger.info("=" * 80)
        logger.info(f"Skills: {candidate['skills']}")
        logger.info(f"Experience: {candidate['experience']}")
        logger.info("")
        
        import pandas as pd
        candidate_row = pd.Series(candidate)
        field_texts = candidate_processor.get_field_texts(candidate_row)
        candidate_embedding = generator.generate_weighted_embedding(
            field_texts=field_texts,
            weights=None,  # Use default with dynamic weights
            method="repetition",
            use_dynamic_weights=True
        )
        
        logger.info("Similarity with JDs:")
        logger.info("-" * 80)
        for jd_data in jd_embeddings:
            similarity = cosine_similarity(candidate_embedding, jd_data["embedding"])
            logger.info(f"  {jd_data['jd']['title']}: {similarity*100:.2f}%")
        logger.info("")
    
    logger.info("=" * 80)
    logger.info("✓ TEST COMPLETE")
    logger.info("=" * 80)
    logger.info("\nNote: To see full improvement, regenerate all embeddings with:")
    logger.info("  python scripts/clear_embeddings.py")
    logger.info("  python scripts/generate_embeddings_from_processed.py --jd-file data/processed/job_data.csv --candidate-file data/processed/candidates_dataset.csv")


if __name__ == "__main__":
    test_improved_similarity()

