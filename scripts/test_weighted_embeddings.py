"""Test script to show how weighted embeddings work with detailed output."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.embeddings.weighted_embedding import WeightedEmbeddingGenerator
from src.data_processing.jd_processor import JDProcessor
from src.data_processing.candidate_processor import CandidateProcessor
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_weighted_embeddings():
    """Test weighted embeddings and show detailed output."""
    logger.info("=" * 80)
    logger.info("TESTING WEIGHTED EMBEDDINGS - DETAILED OUTPUT")
    logger.info("=" * 80)
    logger.info("")
    
    # Initialize weighted embedding generator
    weighted_gen = WeightedEmbeddingGenerator()
    
    # Test JD weighted embedding
    logger.info("=" * 80)
    logger.info("TEST 1: JOB DESCRIPTION WEIGHTED EMBEDDING")
    logger.info("=" * 80)
    
    jd_processor = JDProcessor()
    jd_processor.load_from_csv("data/processed/job_data.csv")
    jd_df = jd_processor.data.head(1)  # Get first JD
    
    if len(jd_df) > 0:
        row = jd_df.iloc[0]
        field_texts = jd_processor.get_field_texts(row)
        weights = WeightedEmbeddingGenerator.DEFAULT_JD_WEIGHTS
        
        logger.info(f"Job ID: {row.get('job_id', 'N/A')}")
        logger.info(f"Title: {row.get('title', 'N/A')}")
        logger.info("")
        logger.info("Field Texts Extracted:")
        for field, text in field_texts.items():
            weight = weights.get(field, 0)
            logger.info(f"  - {field}: {text[:100]}... (Weight: {weight})")
        logger.info("")
        
        # Show how repetition works
        logger.info("Weighted Text (with repetition):")
        text_parts = []
        for field, text in field_texts.items():
            if field in weights and weights[field] > 0:
                repeat_count = max(1, int(round(weights[field])))
                for i in range(repeat_count):
                    text_parts.append(f"{field}: {text}")
        
        weighted_text = " | ".join(text_parts)
        logger.info(f"  {weighted_text[:300]}...")
        logger.info(f"  (Total length: {len(weighted_text)} characters)")
        logger.info("")
        
        # Generate embedding
        embedding = weighted_gen.generate_weighted_embedding(
            field_texts=field_texts,
            weights=weights,
            method="repetition"
        )
        logger.info(f"✓ Generated embedding: {len(embedding)} dimensions")
        logger.info("")
    
    # Test Candidate weighted embedding
    logger.info("=" * 80)
    logger.info("TEST 2: CANDIDATE WEIGHTED EMBEDDING")
    logger.info("=" * 80)
    
    candidate_processor = CandidateProcessor()
    candidate_processor.load_from_csv("data/processed/candidates_dataset.csv")
    candidate_df = candidate_processor.data.head(1)  # Get first candidate
    
    if len(candidate_df) > 0:
        row = candidate_df.iloc[0]
        field_texts = candidate_processor.get_field_texts(row)
        weights = WeightedEmbeddingGenerator.DEFAULT_CANDIDATE_WEIGHTS
        
        logger.info(f"Candidate ID: {row.get('cv_id', row.get('candidate_id', 'N/A'))}")
        logger.info(f"Name: {row.get('user_name', row.get('name', 'N/A'))}")
        logger.info("")
        logger.info("Field Texts Extracted:")
        for field, text in field_texts.items():
            weight = weights.get(field, 0)
            logger.info(f"  - {field}: {text[:100] if text else 'N/A'}... (Weight: {weight})")
        logger.info("")
        
        # Show how repetition works
        logger.info("Weighted Text (with repetition):")
        text_parts = []
        for field, text in field_texts.items():
            if field in weights and weights[field] > 0:
                repeat_count = max(1, int(round(weights[field])))
                for i in range(repeat_count):
                    text_parts.append(f"{field}: {text}")
        
        weighted_text = " | ".join(text_parts)
        logger.info(f"  {weighted_text[:300]}...")
        logger.info(f"  (Total length: {len(weighted_text)} characters)")
        logger.info("")
        
        # Generate embedding
        embedding = weighted_gen.generate_weighted_embedding(
            field_texts=field_texts,
            weights=weights,
            method="repetition"
        )
        logger.info(f"✓ Generated embedding: {len(embedding)} dimensions")
        logger.info("")
    
    # Show weight comparison
    logger.info("=" * 80)
    logger.info("WEIGHT SUMMARY")
    logger.info("=" * 80)
    logger.info("")
    logger.info("JD Field Weights:")
    for field, weight in sorted(WeightedEmbeddingGenerator.DEFAULT_JD_WEIGHTS.items(), 
                                 key=lambda x: x[1], reverse=True):
        logger.info(f"  {field}: {weight} (repeats {max(1, int(round(weight)))} times)")
    logger.info("")
    logger.info("Candidate Field Weights:")
    for field, weight in sorted(WeightedEmbeddingGenerator.DEFAULT_CANDIDATE_WEIGHTS.items(), 
                                 key=lambda x: x[1], reverse=True):
        logger.info(f"  {field}: {weight} (repeats {max(1, int(round(weight)))} times)")
    logger.info("")
    logger.info("=" * 80)
    logger.info("✓ TEST COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    test_weighted_embeddings()

