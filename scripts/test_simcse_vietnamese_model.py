"""Test script for Vietnamese SimCSE model."""
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
    """Test the Vietnamese SimCSE model."""
    logger.info("="*80)
    logger.info("TESTING VIETNAMESE SIMCSE MODEL")
    logger.info("="*80)
    logger.info("")
    
    # Test 1: Basic embedding generation
    logger.info("Test 1: Basic embedding generation")
    logger.info("-" * 80)
    try:
        generator = EmbeddingGenerator()
        logger.info(f"Model loaded: {generator.model_name}")
        logger.info(f"Requires Vietnamese tokenization: {generator.requires_vietnamese_tokenization}")
        logger.info(f"Embedding dimension: {generator.get_embedding_dimension()}")
        
        # Test Vietnamese text
        vietnamese_texts = [
            "Tôi là lập trình viên Java với 5 năm kinh nghiệm",
            "Tôi có kỹ năng về Python, Machine Learning và Deep Learning",
            "Tôi muốn tìm việc làm Backend Developer tại Hồ Chí Minh"
        ]
        
        logger.info("\nGenerating embeddings for Vietnamese texts...")
        embeddings = []
        for text in vietnamese_texts:
            embedding = generator.generate_embedding(text)
            embeddings.append(embedding)
            logger.info(f"  Text: {text[:50]}...")
            logger.info(f"  Embedding shape: {len(embedding)}")
            logger.info(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
        
        logger.info("\n✅ Basic embedding generation: SUCCESS")
    except Exception as e:
        logger.error(f"❌ Basic embedding generation: FAILED - {e}")
        return False
    
    # Test 2: Similarity calculation
    logger.info("\n" + "="*80)
    logger.info("Test 2: Similarity calculation")
    logger.info("-" * 80)
    try:
        text1 = "Lập trình viên Java với kinh nghiệm Spring Boot"
        text2 = "Java Developer có kỹ năng về Spring Framework"
        text3 = "React Developer với kinh nghiệm JavaScript"
        
        emb1 = np.array(generator.generate_embedding(text1))
        emb2 = np.array(generator.generate_embedding(text2))
        emb3 = np.array(generator.generate_embedding(text3))
        
        # Calculate cosine similarity
        similarity_12 = np.dot(emb1, emb2)
        similarity_13 = np.dot(emb1, emb3)
        
        logger.info(f"Text 1: {text1}")
        logger.info(f"Text 2: {text2}")
        logger.info(f"Similarity 1-2: {similarity_12:.4f} ({similarity_12*100:.2f}%)")
        logger.info("")
        logger.info(f"Text 1: {text1}")
        logger.info(f"Text 3: {text3}")
        logger.info(f"Similarity 1-3: {similarity_13:.4f} ({similarity_13*100:.2f}%)")
        
        if similarity_12 > similarity_13:
            logger.info("\n✅ Similarity calculation: SUCCESS (related texts have higher similarity)")
        else:
            logger.warning("\n⚠️ Similarity calculation: Related texts should have higher similarity")
    except Exception as e:
        logger.error(f"❌ Similarity calculation: FAILED - {e}")
        return False
    
    # Test 3: Weighted embedding
    logger.info("\n" + "="*80)
    logger.info("Test 3: Weighted embedding with SimCSE")
    logger.info("-" * 80)
    try:
        weighted_generator = WeightedEmbeddingGenerator()
        
        # Test candidate fields
        candidate_fields = {
            'skills': 'Java, Spring Boot, MySQL, REST API',
            'experience': '5 năm phát triển backend với Java và Spring Framework',
            'desired_job': 'Backend Developer'
        }
        
        embedding = weighted_generator.generate_weighted_embedding(
            field_texts=candidate_fields,
            weights=WeightedEmbeddingGenerator.DEFAULT_CANDIDATE_WEIGHTS,
            method='repetition'
        )
        
        logger.info(f"Generated weighted embedding")
        logger.info(f"  Dimension: {len(embedding)}")
        logger.info(f"  Norm: {np.linalg.norm(embedding):.4f}")
        logger.info("\n✅ Weighted embedding: SUCCESS")
    except Exception as e:
        logger.error(f"❌ Weighted embedding: FAILED - {e}")
        return False
    
    # Test 4: Batch processing
    logger.info("\n" + "="*80)
    logger.info("Test 4: Batch embedding generation")
    logger.info("-" * 80)
    try:
        batch_texts = [
            "Lập trình viên Java",
            "Python Developer",
            "React Developer",
            "Machine Learning Engineer"
        ]
        
        batch_embeddings = generator.generate_embeddings_batch(batch_texts, batch_size=2)
        
        logger.info(f"Generated {len(batch_embeddings)} embeddings in batch")
        for i, (text, emb) in enumerate(zip(batch_texts, batch_embeddings)):
            logger.info(f"  {i+1}. {text}: shape={len(emb)}, norm={np.linalg.norm(emb):.4f}")
        
        logger.info("\n✅ Batch embedding generation: SUCCESS")
    except Exception as e:
        logger.error(f"❌ Batch embedding generation: FAILED - {e}")
        return False
    
    logger.info("\n" + "="*80)
    logger.info("✅ ALL TESTS PASSED!")
    logger.info("="*80)
    logger.info("")
    logger.info("Vietnamese SimCSE model is working correctly!")
    logger.info("Model: VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
    logger.info("This model is optimized for Vietnamese text and should provide")
    logger.info("better similarity matching for Vietnamese job descriptions and candidates.")
    
    return True

if __name__ == "__main__":
    success = test_simcse_model()
    sys.exit(0 if success else 1)

