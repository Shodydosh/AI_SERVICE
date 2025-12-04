"""Background tasks for async processing."""
from sqlalchemy.orm import Session
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


async def update_faiss_and_recommend(
    candidate_id: str,
    db: Session
):
    """
    Background task: Update FAISS indices và pre-compute recommendations cho candidate mới.
    
    Args:
        candidate_id: Candidate ID vừa được tạo
        db: Database session
    """
    try:
        logger.info(f"Background task started for candidate {candidate_id}")
        
        # Step 1: Update FAISS indices
        from src.vector_search.multi_field_faiss_manager import MultiFieldFAISSManager
        from src.embeddings.multi_field_generator import MultiFieldEmbeddingGenerator
        
        generator = MultiFieldEmbeddingGenerator()
        dimension = generator.get_embedding_dimension()
        
        faiss_manager = MultiFieldFAISSManager(
            dimension=dimension,
            index_type="HNSW",
            index_params={
                "ef_search": 128,
                "ef_construction": 200,
                "M": 32
            },
            normalize=True
        )
        
        # Load existing indices nếu có
        base_path = Path("indices/multi_field")
        if base_path.exists() and (base_path / "jd_title_index.faiss").exists():
            try:
                faiss_manager.load_indices(base_path)
                logger.info(f"Loaded existing FAISS indices")
            except Exception as e:
                logger.warning(f"Could not load existing indices: {e}. Will rebuild.")
        
        # Rebuild indices với candidate mới
        faiss_manager.build_indices_from_db(db)
        faiss_manager.save_indices(base_path)
        
        logger.info(f"✓ Updated FAISS indices for candidate {candidate_id}")
        
        # Step 2: Pre-compute recommendations
        from src.services.multi_filter_matching_service import MultiFilterMatchingService
        
        matching_service = MultiFilterMatchingService(db, use_faiss=True)
        recommendations = matching_service.find_jobs_for_candidate(
            candidate_id=candidate_id,
            top_k=10
        )
        
        # Lưu recommendations vào processed table (nếu có)
        try:
            from src.database.repository import EmbeddingRepository
            repository = EmbeddingRepository(db)
            
            # Convert recommendations format
            recommendations_dict = {
                candidate_id: [
                    {
                        "job_id": rec.get("job_id"),
                        "title": rec.get("title"),
                        "similarity_score": rec.get("similarity_score", 0.0)
                    }
                    for rec in recommendations
                ]
            }
            
            repository.save_processed_recommendations_batch(
                all_recommendations=recommendations_dict,
                replace_existing=True
            )
            
            logger.info(f"✓ Pre-computed {len(recommendations)} recommendations for candidate {candidate_id}")
        except Exception as e:
            logger.warning(f"Could not save processed recommendations: {e}")
            # Không critical, chỉ log warning
        
        logger.info(f"✓ Background task completed for candidate {candidate_id}")
        
    except Exception as e:
        logger.error(f"Error in background task for candidate {candidate_id}: {e}", exc_info=True)
        # Không raise exception để không ảnh hưởng response


async def update_faiss_for_batch(
    candidate_ids: List[str],
    db: Session
):
    """
    Background task: Update FAISS indices cho batch candidates.
    
    Args:
        candidate_ids: List of candidate IDs
        db: Database session
    """
    try:
        logger.info(f"Background task started for batch of {len(candidate_ids)} candidates")
        
        from src.vector_search.multi_field_faiss_manager import MultiFieldFAISSManager
        from src.embeddings.multi_field_generator import MultiFieldEmbeddingGenerator
        
        generator = MultiFieldEmbeddingGenerator()
        dimension = generator.get_embedding_dimension()
        
        faiss_manager = MultiFieldFAISSManager(
            dimension=dimension,
            index_type="HNSW",
            index_params={
                "ef_search": 128,
                "ef_construction": 200,
                "M": 32
            },
            normalize=True
        )
        
        # Load existing indices nếu có
        base_path = Path("indices/multi_field")
        if base_path.exists() and (base_path / "jd_title_index.faiss").exists():
            try:
                faiss_manager.load_indices(base_path)
                logger.info(f"Loaded existing FAISS indices")
            except Exception as e:
                logger.warning(f"Could not load existing indices: {e}. Will rebuild.")
        
        # Rebuild indices
        faiss_manager.build_indices_from_db(db)
        faiss_manager.save_indices(base_path)
        
        logger.info(f"✓ Updated FAISS indices for batch of {len(candidate_ids)} candidates")
        
        # Pre-compute recommendations cho từng candidate
        from src.services.multi_filter_matching_service import MultiFilterMatchingService
        matching_service = MultiFilterMatchingService(db, use_faiss=True)
        
        from src.database.repository import EmbeddingRepository
        repository = EmbeddingRepository(db)
        
        all_recommendations = {}
        processed = 0
        
        for candidate_id in candidate_ids:
            try:
                recommendations = matching_service.find_jobs_for_candidate(
                    candidate_id=candidate_id,
                    top_k=10
                )
                
                if recommendations:
                    all_recommendations[candidate_id] = [
                        {
                            "job_id": rec.get("job_id"),
                            "title": rec.get("title"),
                            "similarity_score": rec.get("similarity_score", 0.0)
                        }
                        for rec in recommendations
                    ]
                    processed += 1
            except Exception as e:
                logger.warning(f"Could not pre-compute recommendations for candidate {candidate_id}: {e}")
        
        # Save batch recommendations
        if all_recommendations:
            try:
                repository.save_processed_recommendations_batch(
                    all_recommendations=all_recommendations,
                    replace_existing=True
                )
                logger.info(f"✓ Pre-computed recommendations for {processed}/{len(candidate_ids)} candidates")
            except Exception as e:
                logger.warning(f"Could not save processed recommendations: {e}")
        
        logger.info(f"✓ Background task completed for batch")
        
    except Exception as e:
        logger.error(f"Error in background task for batch: {e}", exc_info=True)

