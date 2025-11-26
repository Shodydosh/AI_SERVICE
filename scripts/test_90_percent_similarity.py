"""Test to achieve 90%+ similarity using AI engineering best practices."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from src.embeddings.weighted_embedding import WeightedEmbeddingGenerator
from src.data_processing.jd_processor import JDProcessor
from src.data_processing.candidate_processor import CandidateProcessor
import pandas as pd
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


def test_90_percent_similarity():
    """Test to achieve 90%+ similarity with perfect alignment."""
    logger.info("=" * 80)
    logger.info("TESTING FOR 90%+ SIMILARITY WITH PERFECT TEXT ALIGNMENT")
    logger.info("=" * 80)
    
    generator = WeightedEmbeddingGenerator()
    jd_processor = JDProcessor()
    candidate_processor = CandidateProcessor()
    
    # Create perfectly aligned test data
    # Key: Use EXACT same format and terms for JD and candidate
    
    test_jd = {
        "title": "AI Engineer",
        "skills": "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch",
        "requirements": "5 năm kinh nghiệm phát triển phần mềm, chuyên về AI và Machine Learning",
        "description": "Phát triển và triển khai các giải pháp AI và Machine Learning"
    }
    
    test_candidate = {
        "skills": "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch",
        "experience": "5 năm kinh nghiệm phát triển phần mềm, chuyên về AI và Machine Learning",
        "desired_job_translated": "AI Engineer",
        "summary": "Phát triển và triển khai các giải pháp AI và Machine Learning"
    }
    
    logger.info("\nTest JD:")
    logger.info(f"  Title: {test_jd['title']}")
    logger.info(f"  Skills: {test_jd['skills']}")
    logger.info(f"  Requirements: {test_jd['requirements']}")
    
    logger.info("\nTest Candidate:")
    logger.info(f"  Skills: {test_candidate['skills']}")
    logger.info(f"  Experience: {test_candidate['experience']}")
    logger.info(f"  Desired Job: {test_candidate['desired_job_translated']}")
    
    # Generate embeddings
    logger.info("\nGenerating embeddings with perfect alignment...")
    
    # JD embedding
    jd_row = pd.Series(test_jd)
    jd_field_texts = jd_processor.get_field_texts(jd_row)
    jd_embedding = generator.generate_weighted_embedding(
        field_texts=jd_field_texts,
        weights=WeightedEmbeddingGenerator.DEFAULT_JD_WEIGHTS,
        method="repetition"
    )
    
    # Candidate embedding
    candidate_row = pd.Series(test_candidate)
    candidate_field_texts = candidate_processor.get_field_texts(candidate_row)
    candidate_embedding = generator.generate_weighted_embedding(
        field_texts=candidate_field_texts,
        weights=None,  # Use default with dynamic weights
        method="repetition",
        use_dynamic_weights=True
    )
    
    # Calculate similarity
    similarity = cosine_similarity(jd_embedding, candidate_embedding)
    
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS:")
    logger.info("=" * 80)
    logger.info(f"Similarity: {similarity*100:.2f}%")
    
    if similarity >= 0.90:
        logger.info("✓ ACHIEVED 90%+ SIMILARITY!")
    elif similarity >= 0.85:
        logger.info("⚠ Close to 90% - need minor adjustments")
    else:
        logger.info("⚠ Below 90% - need further optimization")
    
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATIONS FOR 90%+ SIMILARITY:")
    logger.info("=" * 80)
    logger.info("1. Ensure JD and candidate use EXACT same format")
    logger.info("2. Use same field labels (e.g., 'Required Skills and Technologies')")
    logger.info("3. Repeat exact text multiple times (current: 3x multiplier)")
    logger.info("4. Increase weights for critical fields (current: skills=6.0, experience=5.5)")
    logger.info("5. Use exact matching for key terms (skills, experience)")
    logger.info("6. Consider cross-encoder re-ranking for final boost")
    logger.info("=" * 80)


if __name__ == "__main__":
    test_90_percent_similarity()

