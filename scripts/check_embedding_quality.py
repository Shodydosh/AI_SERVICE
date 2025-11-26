"""Check the quality of embeddings in the database."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import numpy as np
from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.database.repository import EmbeddingRepository
from src.embeddings.generator import EmbeddingGenerator
from src.embeddings.weighted_embedding import WeightedEmbeddingGenerator
from config.settings import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_embedding_statistics(db: Session, repository: EmbeddingRepository):
    """Check basic statistics of embeddings."""
    logger.info("=" * 80)
    logger.info("EMBEDDING STATISTICS")
    logger.info("=" * 80)
    logger.info("")
    
    # Check JD embeddings
    from src.database.models import JobDescriptionEmbedding, CandidateEmbedding
    
    jd_count = db.query(JobDescriptionEmbedding).count()
    candidate_count = db.query(CandidateEmbedding).count()
    
    logger.info(f"Total JD embeddings: {jd_count}")
    logger.info(f"Total Candidate embeddings: {candidate_count}")
    logger.info("")
    
    if jd_count == 0 and candidate_count == 0:
        logger.warning("⚠️  No embeddings found in database!")
        return False
    
    # Check embedding dimensions
    if jd_count > 0:
        sample_jd = db.query(JobDescriptionEmbedding).first()
        jd_dim = len(sample_jd.embedding) if sample_jd else 0
        jd_norm = np.linalg.norm(sample_jd.embedding) if sample_jd else 0
        logger.info(f"JD embedding dimension: {jd_dim}")
        logger.info(f"JD embedding norm (sample): {jd_norm:.4f}")
        logger.info(f"Expected dimension: {settings.EMBEDDING_DIMENSION}")
        logger.info("")
    
    if candidate_count > 0:
        sample_candidate = db.query(CandidateEmbedding).first()
        candidate_dim = len(sample_candidate.embedding) if sample_candidate else 0
        candidate_norm = np.linalg.norm(sample_candidate.embedding) if sample_candidate else 0
        logger.info(f"Candidate embedding dimension: {candidate_dim}")
        logger.info(f"Candidate embedding norm (sample): {candidate_norm:.4f}")
        logger.info(f"Expected dimension: {settings.EMBEDDING_DIMENSION}")
        logger.info("")
    
    return True


def check_similarity_quality(db: Session, repository: EmbeddingRepository):
    """Check similarity quality with sample queries."""
    logger.info("=" * 80)
    logger.info("SIMILARITY QUALITY TEST")
    logger.info("=" * 80)
    logger.info("")
    
    from src.database.models import JobDescriptionEmbedding, CandidateEmbedding
    
    # Test 1: Find similar JDs
    logger.info("Test 1: Finding similar job descriptions...")
    jds = db.query(JobDescriptionEmbedding).limit(5).all()
    if len(jds) >= 2:
        query_jd = jds[0]
        similar_jds = repository.find_similar_jds(query_jd.embedding, limit=5)
        
        logger.info(f"Query JD: {query_jd.title[:50]}...")
        logger.info("")
        logger.info("Top 5 similar JDs:")
        for i, jd in enumerate(similar_jds[:5], 1):
            query_vec = np.array(query_jd.embedding)
            jd_vec = np.array(jd.embedding)
            similarity = np.dot(query_vec, jd_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(jd_vec)
            )
            logger.info(f"  {i}. {jd.title[:50]}... (Similarity: {similarity * 100:.2f}%)")
        logger.info("")
    
    # Test 2: Find similar candidates
    logger.info("Test 2: Finding similar candidates...")
    candidates = db.query(CandidateEmbedding).limit(5).all()
    if len(candidates) >= 2:
        query_candidate = candidates[0]
        similar_candidates = repository.find_similar_candidates(query_candidate.embedding, limit=5)
        
        logger.info(f"Query Candidate: {query_candidate.name or query_candidate.candidate_id}")
        logger.info("")
        logger.info("Top 5 similar candidates:")
        for i, candidate in enumerate(similar_candidates[:5], 1):
            query_vec = np.array(query_candidate.embedding)
            candidate_vec = np.array(candidate.embedding)
            similarity = np.dot(query_vec, candidate_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(candidate_vec)
            )
            logger.info(f"  {i}. {candidate.name or candidate.candidate_id} (Similarity: {similarity * 100:.2f}%)")
        logger.info("")
    
    # Test 3: JD-Candidate matching
    logger.info("Test 3: JD-Candidate matching...")
    if jds and candidates:
        test_jd = jds[0]
        test_candidate = candidates[0]
        
        query_vec = np.array(test_candidate.embedding)
        jd_vec = np.array(test_jd.embedding)
        similarity = np.dot(query_vec, jd_vec) / (
            np.linalg.norm(query_vec) * np.linalg.norm(jd_vec)
        )
        
        logger.info(f"JD: {test_jd.title[:50]}...")
        logger.info(f"Candidate: {test_candidate.name or test_candidate.candidate_id}")
        logger.info(f"Similarity: {similarity * 100:.2f}%")
        logger.info("")
        
        # Find top matching JDs for candidate
        recommended_jds = repository.recommend_jobs_for_candidate(test_candidate.candidate_id, limit=3)
        logger.info(f"Top 3 recommended jobs for candidate {test_candidate.candidate_id}:")
        for i, jd in enumerate(recommended_jds, 1):
            query_vec = np.array(test_candidate.embedding)
            jd_vec = np.array(jd.embedding)
            similarity = np.dot(query_vec, jd_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(jd_vec)
            )
            logger.info(f"  {i}. {jd.title[:50]}... (Similarity: {similarity * 100:.2f}%)")
        logger.info("")


def check_embedding_normalization(db: Session):
    """Check if embeddings are properly normalized."""
    logger.info("=" * 80)
    logger.info("NORMALIZATION CHECK")
    logger.info("=" * 80)
    logger.info("")
    
    from src.database.models import JobDescriptionEmbedding, CandidateEmbedding
    
    # Check JD embeddings normalization
    jds = db.query(JobDescriptionEmbedding).limit(100).all()
    if jds:
        norms = [np.linalg.norm(jd.embedding) for jd in jds]
        avg_norm = np.mean(norms)
        min_norm = np.min(norms)
        max_norm = np.max(norms)
        
        logger.info(f"JD Embeddings (sample of {len(jds)}):")
        logger.info(f"  Average norm: {avg_norm:.4f}")
        logger.info(f"  Min norm: {min_norm:.4f}")
        logger.info(f"  Max norm: {max_norm:.4f}")
        logger.info(f"  Expected: 1.0000 (normalized)")
        
        if abs(avg_norm - 1.0) < 0.01:
            logger.info("  ✅ Embeddings are properly normalized")
        else:
            logger.warning(f"  ⚠️  Embeddings may not be normalized (avg norm: {avg_norm:.4f})")
        logger.info("")
    
    # Check candidate embeddings normalization
    candidates = db.query(CandidateEmbedding).limit(100).all()
    if candidates:
        norms = [np.linalg.norm(candidate.embedding) for candidate in candidates]
        avg_norm = np.mean(norms)
        min_norm = np.min(norms)
        max_norm = np.max(norms)
        
        logger.info(f"Candidate Embeddings (sample of {len(candidates)}):")
        logger.info(f"  Average norm: {avg_norm:.4f}")
        logger.info(f"  Min norm: {min_norm:.4f}")
        logger.info(f"  Max norm: {max_norm:.4f}")
        logger.info(f"  Expected: 1.0000 (normalized)")
        
        if abs(avg_norm - 1.0) < 0.01:
            logger.info("  ✅ Embeddings are properly normalized")
        else:
            logger.warning(f"  ⚠️  Embeddings may not be normalized (avg norm: {avg_norm:.4f})")
        logger.info("")


def test_vietnamese_understanding():
    """Test Vietnamese text understanding with sample queries."""
    logger.info("=" * 80)
    logger.info("VIETNAMESE TEXT UNDERSTANDING TEST")
    logger.info("=" * 80)
    logger.info("")
    
    generator = EmbeddingGenerator()
    
    # Test Vietnamese sentences
    test_pairs = [
        (
            "Lập trình viên Java với kinh nghiệm Spring Boot",
            "Nhà phát triển phần mềm Java sử dụng Spring Framework",
            "Should be similar (both Java developers)"
        ),
        (
            "Lập trình viên Java với kinh nghiệm Spring Boot",
            "Kỹ sư phần mềm chuyên về Python và Machine Learning",
            "Should be less similar (different technologies)"
        ),
        (
            "Chuyên viên phát triển ứng dụng web với React và Node.js",
            "Lập trình viên frontend React và backend Node.js",
            "Should be similar (both React/Node.js developers)"
        )
    ]
    
    logger.info("Testing Vietnamese sentence similarity:")
    logger.info("")
    
    for i, (text1, text2, expected) in enumerate(test_pairs, 1):
        emb1 = generator.generate_embedding(text1)
        emb2 = generator.generate_embedding(text2)
        
        vec1 = np.array(emb1)
        vec2 = np.array(emb2)
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        logger.info(f"Test {i}:")
        logger.info(f"  Text 1: {text1}")
        logger.info(f"  Text 2: {text2}")
        logger.info(f"  Similarity: {similarity * 100:.2f}%")
        logger.info(f"  Expected: {expected}")
        logger.info("")


def main():
    """Main function to check embedding quality."""
    logger.info("=" * 80)
    logger.info("EMBEDDING QUALITY CHECK")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Model: {settings.EMBEDDING_MODEL}")
    logger.info(f"Dimension: {settings.EMBEDDING_DIMENSION}")
    logger.info("")
    
    db: Session = SessionLocal()
    try:
        repository = EmbeddingRepository(db)
        
        # Check statistics
        if not check_embedding_statistics(db, repository):
            logger.error("No embeddings found. Please generate embeddings first.")
            return
        
        # Check normalization
        check_embedding_normalization(db)
        
        # Check similarity quality
        check_similarity_quality(db, repository)
        
        # Test Vietnamese understanding
        test_vietnamese_understanding()
        
        logger.info("=" * 80)
        logger.info("✅ EMBEDDING QUALITY CHECK COMPLETE")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error checking embedding quality: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    main()

