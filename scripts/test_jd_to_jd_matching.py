"""Script test JD-to-JD matching: In ra 10 sample JD và recommend 5 JD tương tự cho mỗi sample."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import argparse
from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.database.multi_field_repository import MultiFieldEmbeddingRepository
from src.services.enhanced_multi_filter_matching_service import EnhancedMultiFilterMatchingService
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_similar_jobs(
    db: Session,
    source_job_id: str,
    top_k: int = 5,
    exclude_job_id: str = None
) -> list:
    """
    Tìm các JD tương tự với một JD nguồn.
    
    Args:
        db: Database session
        source_job_id: Job ID nguồn
        top_k: Số lượng JD tương tự cần tìm
        exclude_job_id: Job ID cần loại trừ (thường là chính nó)
        
    Returns:
        List of similar jobs với similarity scores
    """
    repo = MultiFieldEmbeddingRepository(db)
    
    # Get source job
    source_job = repo.get_job_multi_embedding(source_job_id)
    if not source_job or not source_job.title_embedding:
        return []
    
    # Get all jobs
    all_jobs = repo.get_all_job_multi_embeddings()
    
    # Calculate similarities
    similarities = []
    source_title_emb = np.array(source_job.title_embedding, dtype=np.float32)
    source_skills_emb = np.array(source_job.skills_embedding or [], dtype=np.float32) if source_job.skills_embedding else None
    source_req_emb = np.array(source_job.requirement_embedding or [], dtype=np.float32) if source_job.requirement_embedding else None
    
    for job in all_jobs:
        if job.job_id == exclude_job_id:
            continue
        
        if not job.title_embedding:
            continue
        
        # Title similarity
        job_title_emb = np.array(job.title_embedding, dtype=np.float32)
        title_sim = cosine_similarity([source_title_emb], [job_title_emb])[0][0]
        
        # Skills similarity (if available)
        skills_sim = 0.0
        if source_skills_emb is not None and len(source_skills_emb) > 0 and job.skills_embedding:
            job_skills_emb = np.array(job.skills_embedding, dtype=np.float32)
            if len(job_skills_emb) > 0:
                skills_sim = cosine_similarity([source_skills_emb], [job_skills_emb])[0][0]
        
        # Requirement similarity (if available)
        req_sim = 0.0
        if source_req_emb is not None and len(source_req_emb) > 0 and job.requirement_embedding:
            job_req_emb = np.array(job.requirement_embedding, dtype=np.float32)
            if len(job_req_emb) > 0:
                req_sim = cosine_similarity([source_req_emb], [job_req_emb])[0][0]
        
        # Combined score (Title: 50%, Skills: 35%, Requirement: 15%)
        combined_score = (
            title_sim * 0.5 +
            skills_sim * 0.35 +
            req_sim * 0.15
        )
        
        similarities.append({
            'job_id': job.job_id,
            'title': job.title,
            'company': job.company,
            'location': job.location,
            'similarity_score': float(combined_score),
            'title_similarity': float(title_sim),
            'skills_similarity': float(skills_sim) if source_skills_emb is not None else None,
            'requirement_similarity': float(req_sim) if source_req_emb is not None else None
        })
    
    # Sort by combined score
    similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    return similarities[:top_k]


def test_jd_to_jd_matching(num_samples: int = 10, top_k: int = 5):
    """Test JD-to-JD matching."""
    logger.info("=" * 100)
    logger.info("🧪 TEST JD-TO-JD MATCHING")
    logger.info("=" * 100)
    logger.info(f"Sample {num_samples} JD và recommend {top_k} JD tương tự cho mỗi sample")
    logger.info("=" * 100)
    
    db: Session = next(get_db())
    try:
        repo = MultiFieldEmbeddingRepository(db)
        
        # Get sample jobs
        all_jobs = repo.get_all_job_multi_embeddings()
        
        # Filter jobs có embeddings
        valid_jobs = [job for job in all_jobs if job.title_embedding]
        
        if len(valid_jobs) < num_samples:
            logger.warning(f"Chỉ có {len(valid_jobs)} jobs có embeddings, sẽ dùng tất cả")
            sample_jobs = valid_jobs[:num_samples]
        else:
            # Lấy ngẫu nhiên num_samples jobs
            import random
            sample_jobs = random.sample(valid_jobs, num_samples)
        
        logger.info(f"\n📋 Đã chọn {len(sample_jobs)} sample JD\n")
        
        # Process each sample
        for idx, source_job in enumerate(sample_jobs, 1):
            logger.info("=" * 100)
            logger.info(f"SAMPLE {idx}/{len(sample_jobs)}")
            logger.info("=" * 100)
            
            # Display source job
            logger.info(f"\n📌 SOURCE JD:")
            logger.info(f"  Job ID: {source_job.job_id}")
            logger.info(f"  Title: {source_job.title or 'N/A'}")
            logger.info(f"  Company: {source_job.company or 'N/A'}")
            logger.info(f"  Location: {source_job.location or 'N/A'}")
            if source_job.skills:
                skills_preview = (source_job.skills[:100] + "...") if len(source_job.skills) > 100 else source_job.skills
                logger.info(f"  Skills: {skills_preview}")
            
            # Find similar jobs
            logger.info(f"\n🔍 Tìm {top_k} JD tương tự...")
            similar_jobs = find_similar_jobs(
                db=db,
                source_job_id=source_job.job_id,
                top_k=top_k,
                exclude_job_id=source_job.job_id
            )
            
            if not similar_jobs:
                logger.warning("  ⚠ Không tìm thấy JD tương tự")
                continue
            
            logger.info(f"\n✅ RECOMMENDED {len(similar_jobs)} JD:")
            
            for i, rec_job in enumerate(similar_jobs, 1):
                logger.info(f"\n  {i}. Job ID: {rec_job['job_id']}")
                logger.info(f"     Title: {rec_job['title'] or 'N/A'}")
                logger.info(f"     Company: {rec_job['company'] or 'N/A'}")
                logger.info(f"     Location: {rec_job['location'] or 'N/A'}")
                logger.info(f"     Similarity Score: {rec_job['similarity_score']:.4f}")
                logger.info(f"       - Title: {rec_job['title_similarity']:.4f}")
                if rec_job['skills_similarity'] is not None:
                    logger.info(f"       - Skills: {rec_job['skills_similarity']:.4f}")
                if rec_job['requirement_similarity'] is not None:
                    logger.info(f"       - Requirement: {rec_job['requirement_similarity']:.4f}")
            
            logger.info("")  # Empty line between samples
        
        logger.info("=" * 100)
        logger.info("✅ HOÀN THÀNH TEST JD-TO-JD MATCHING")
        logger.info("=" * 100)
        
    except Exception as e:
        logger.error(f"✗ Error: {e}", exc_info=True)
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test JD-to-JD matching')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of sample JD to test')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of similar JD to recommend for each sample')
    
    args = parser.parse_args()
    test_jd_to_jd_matching(args.samples, args.top_k)

