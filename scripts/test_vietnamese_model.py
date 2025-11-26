"""Test Vietnamese embedding model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vietnamese_model():
    """Test the Vietnamese multilingual model."""
    logger.info("=" * 80)
    logger.info("TESTING VIETNAMESE MULTILINGUAL MODEL")
    logger.info("=" * 80)
    
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    
    try:
        logger.info(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name)
        logger.info("✓ Model loaded successfully!")
        
        # Test with Vietnamese text
        test_texts = [
            "Kỹ năng lập trình Python, Machine Learning",
            "Kinh nghiệm làm việc tại công ty công nghệ",
            "Tốt nghiệp đại học chuyên ngành Công nghệ thông tin"
        ]
        
        logger.info("\nTesting Vietnamese text embedding:")
        logger.info("-" * 80)
        for text in test_texts:
            embedding = model.encode(text, normalize_embeddings=True)
            logger.info(f"Text: {text}")
            logger.info(f"  Dimension: {len(embedding)}")
            logger.info(f"  Norm: {sum(x*x for x in embedding)**0.5:.4f}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ Vietnamese model test completed successfully!")
        logger.info("=" * 80)
        logger.info(f"Model dimension: {model.get_sentence_embedding_dimension()}")
        logger.info("Ready to use for Vietnamese text embedding!")
        
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vietnamese_model()

