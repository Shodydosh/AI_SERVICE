"""Test script for SimCSE Vietnamese model."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from src.embeddings.generator import EmbeddingGenerator
from src.embeddings.weighted_embedding import WeightedEmbeddingGenerator
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_simcse_model():
    """Test SimCSE Vietnamese model with sample Vietnamese text."""
    logger.info("=" * 80)
    logger.info("TESTING SIMCSE VIETNAMESE MODEL")
    logger.info("=" * 80)
    logger.info("")
    
    # Initialize generator
    logger.info("Loading SimCSE Vietnamese model...")
    generator = EmbeddingGenerator()
    logger.info(f"Model: {generator.model_name}")
    logger.info(f"Requires Vietnamese tokenization: {generator.requires_vietnamese_tokenization}")
    logger.info(f"Embedding dimension: {generator.get_embedding_dimension()}")
    logger.info("")
    
    # Test Vietnamese sentences
    test_sentences = [
        "Lập trình viên Java với kinh nghiệm Spring Boot",
        "Nhà phát triển phần mềm Java sử dụng Spring Framework",
        "Kỹ sư phần mềm chuyên về Python và Machine Learning",
        "Chuyên viên phát triển ứng dụng web với React và Node.js",
        "Quản lý dự án công nghệ thông tin"
    ]
    
    logger.info("Generating embeddings for Vietnamese sentences...")
    logger.info("")
    
    embeddings = []
    for i, sentence in enumerate(test_sentences, 1):
        logger.info(f"{i}. {sentence}")
        embedding = generator.generate_embedding(sentence)
        embeddings.append(embedding)
        logger.info(f"   Embedding shape: {len(embedding)}")
        logger.info(f"   Embedding norm: {np.linalg.norm(embedding):.4f}")
        logger.info("")
    
    # Calculate similarity between similar sentences
    logger.info("=" * 80)
    logger.info("SIMILARITY TEST")
    logger.info("=" * 80)
    logger.info("")
    
    # Java sentences should be similar
    java1 = np.array(embeddings[0])
    java2 = np.array(embeddings[1])
    similarity_java = np.dot(java1, java2)
    logger.info(f"Java Developer 1 vs Java Developer 2: {similarity_java * 100:.2f}%")
    logger.info("")
    
    # Java vs Python should be less similar
    python = np.array(embeddings[2])
    similarity_java_python = np.dot(java1, python)
    logger.info(f"Java Developer vs Python Engineer: {similarity_java_python * 100:.2f}%")
    logger.info("")
    
    # React vs Python should be less similar
    react = np.array(embeddings[3])
    similarity_react_python = np.dot(react, python)
    logger.info(f"React Developer vs Python Engineer: {similarity_react_python * 100:.2f}%")
    logger.info("")
    
    # Test weighted embedding
    logger.info("=" * 80)
    logger.info("TESTING WEIGHTED EMBEDDING")
    logger.info("=" * 80)
    logger.info("")
    
    weighted_generator = WeightedEmbeddingGenerator()
    
    # Test JD embedding
    jd_fields = {
        'title': 'Lập trình viên Java',
        'skills': 'Java, Spring Boot, MySQL, REST API',
        'requirements': '3 năm kinh nghiệm Java, Spring Framework',
        'description': 'Phát triển ứng dụng web backend với Java'
    }
    
    logger.info("Generating weighted JD embedding...")
    jd_embedding = weighted_generator.generate_weighted_embedding(
        field_texts=jd_fields,
        weights=WeightedEmbeddingGenerator.DEFAULT_JD_WEIGHTS,
        method="repetition"
    )
    logger.info(f"JD embedding shape: {len(jd_embedding)}")
    logger.info(f"JD embedding norm: {np.linalg.norm(jd_embedding):.4f}")
    logger.info("")
    
    # Test candidate embedding
    candidate_fields = {
        'skills': 'Java, Spring Boot, MySQL',
        'experience': '5 năm phát triển backend với Java và Spring Framework',
        'desired_job': 'Lập trình viên Java'
    }
    
    logger.info("Generating weighted candidate embedding...")
    candidate_embedding = weighted_generator.generate_weighted_embedding(
        field_texts=candidate_fields,
        weights=WeightedEmbeddingGenerator.DEFAULT_CANDIDATE_WEIGHTS,
        method="repetition"
    )
    logger.info(f"Candidate embedding shape: {len(candidate_embedding)}")
    logger.info(f"Candidate embedding norm: {np.linalg.norm(candidate_embedding):.4f}")
    logger.info("")
    
    # Calculate similarity
    similarity = np.dot(jd_embedding, candidate_embedding)
    logger.info(f"JD vs Candidate similarity: {similarity * 100:.2f}%")
    logger.info("")
    
    logger.info("=" * 80)
    logger.info("✅ SIMCSE VIETNAMESE MODEL TEST COMPLETE")
    logger.info("=" * 80)

if __name__ == "__main__":
    test_simcse_model()

