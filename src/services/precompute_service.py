"""Service for pre-computing top 10 job recommendations for all candidates."""
from typing import List, Dict
from sqlalchemy.orm import Session
import logging
from tqdm import tqdm

from src.database.repository import EmbeddingRepository
from src.services.matching_service import MatchingService
from src.vector_search.faiss_manager import FAISSIndexManager
from pathlib import Path

logger = logging.getLogger(__name__)


class PrecomputeService:
    """Service for pre-computing recommendations."""
    
    def __init__(self, db: Session):
        """
        Initialize precompute service.
        
        Args:
            db: Database session
        """
        self.db = db
        self.repository = EmbeddingRepository(db)
        self.matching_service = MatchingService(
            db=db,
            use_faiss=True,
            use_reranking=True
        )
    
    def precompute_all_candidates(
        self,
        batch_size: int = 100,
        top_k: int = 10
    ) -> Dict[str, int]:
        """
        Pre-compute top 10 job recommendations for all candidates.
        
        Args:
            batch_size: Number of candidates to process in each batch
            top_k: Number of top jobs to save (default: 10)
        
        Returns:
            Dict with statistics: {
                'total_candidates': int,
                'processed': int,
                'failed': int,
                'total_recommendations': int
            }
        """
        logger.info("=" * 80)
        logger.info("PRE-COMPUTING RECOMMENDATIONS FOR ALL CANDIDATES")
        logger.info("=" * 80)
        
        # Get all candidates
        all_candidates = self.repository.get_all_candidate_embeddings()
        total_candidates = len(all_candidates)
        
        logger.info(f"Found {total_candidates} candidates to process")
        logger.info(f"Batch size: {batch_size}, Top K: {top_k}")
        logger.info("")
        
        if total_candidates == 0:
            logger.warning("No candidates found in database")
            return {
                'total_candidates': 0,
                'processed': 0,
                'failed': 0,
                'total_recommendations': 0
            }
        
        # Process in batches
        all_recommendations = {}
        processed = 0
        failed = 0
        
        # Process with progress bar
        for i in tqdm(range(0, total_candidates, batch_size), desc="Processing batches"):
            batch = all_candidates[i:i + batch_size]
            
            for candidate in tqdm(batch, desc=f"Batch {i//batch_size + 1}", leave=False):
                try:
                    candidate_id = candidate.candidate_id
                    
                    # Find top K jobs for this candidate
                    jobs = self.matching_service.find_jobs_for_candidate(
                        candidate_id=candidate_id,
                        top_k=top_k,
                        use_faiss=True,
                        use_processed=False  # Don't use pre-computed, compute fresh
                    )
                    
                    if jobs:
                        all_recommendations[candidate_id] = jobs
                        processed += 1
                    else:
                        logger.warning(f"No jobs found for candidate {candidate_id}")
                        failed += 1
                
                except Exception as e:
                    logger.error(f"Error processing candidate {candidate.candidate_id}: {e}")
                    failed += 1
            
            # Save batch to database
            if all_recommendations:
                try:
                    total_saved = self.repository.save_processed_recommendations_batch(
                        all_recommendations=all_recommendations,
                        replace_existing=True
                    )
                    logger.info(f"Saved {total_saved} recommendations for {len(all_recommendations)} candidates")
                    all_recommendations = {}  # Clear batch
                except Exception as e:
                    logger.error(f"Error saving batch: {e}")
        
        # Save remaining recommendations
        if all_recommendations:
            try:
                total_saved = self.repository.save_processed_recommendations_batch(
                    all_recommendations=all_recommendations,
                    replace_existing=True
                )
                logger.info(f"Saved final batch: {total_saved} recommendations")
            except Exception as e:
                logger.error(f"Error saving final batch: {e}")
        
        total_recommendations = processed * top_k
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("PRE-COMPUTATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total candidates: {total_candidates}")
        logger.info(f"Successfully processed: {processed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total recommendations saved: {total_recommendations}")
        logger.info("=" * 80)
        
        return {
            'total_candidates': total_candidates,
            'processed': processed,
            'failed': failed,
            'total_recommendations': total_recommendations
        }
    
    def regenerate_embeddings_and_recompute(
        self,
        jd_file: str = None,
        candidate_file: str = None,
        rebuild_faiss: bool = True
    ) -> Dict[str, any]:
        """
        Regenerate embeddings for all data and recompute recommendations.
        This is the full workflow that runs every 12 hours.
        
        Args:
            jd_file: Path to JD dataset (optional, will use existing if not provided)
            candidate_file: Path to candidate dataset (optional)
            rebuild_faiss: Whether to rebuild FAISS indices
        
        Returns:
            Dict with results from both embedding generation and pre-computation
        """
        logger.info("=" * 80)
        logger.info("FULL REGENERATION WORKFLOW")
        logger.info("=" * 80)
        
        results = {
            'embeddings': {},
            'precomputation': {}
        }
        
        # Step 1: Regenerate embeddings
        try:
            from src.services.embedding_service import EmbeddingService
            
            embedding_service = EmbeddingService(self.db)
            
            if jd_file:
                logger.info("Regenerating JD embeddings...")
                jd_count = embedding_service.process_jd_dataset(jd_file, 'csv')
                results['embeddings']['jd_count'] = jd_count
                logger.info(f"✓ Regenerated {jd_count} JD embeddings")
            
            if candidate_file:
                logger.info("Regenerating candidate embeddings...")
                candidate_count = embedding_service.process_candidate_dataset(candidate_file, 'csv')
                results['embeddings']['candidate_count'] = candidate_count
                logger.info(f"✓ Regenerated {candidate_count} candidate embeddings")
            
        except Exception as e:
            logger.error(f"Error regenerating embeddings: {e}")
            results['embeddings']['error'] = str(e)
        
        # Step 2: Rebuild FAISS indices
        if rebuild_faiss:
            try:
                logger.info("Rebuilding FAISS indices...")
                from src.vector_search.faiss_manager import FAISSIndexManager
                from config.settings import settings
                
                faiss_manager = FAISSIndexManager(
                    dimension=settings.EMBEDDING_DIMENSION,
                    index_type="HNSW",
                    index_params={
                        "ef_search": 64,
                        "ef_construction": 200,
                        "M": 32
                    },
                    normalize=True
                )
                
                if jd_file:
                    faiss_manager.build_index_from_db(self.db, dataset_type='jd')
                    faiss_manager.save_index('indices/jd_index.faiss', dataset_type='jd')
                    logger.info("✓ Rebuilt JD FAISS index")
                
                if candidate_file:
                    faiss_manager.build_index_from_db(self.db, dataset_type='candidate')
                    faiss_manager.save_index('indices/candidate_index.faiss', dataset_type='candidate')
                    logger.info("✓ Rebuilt candidate FAISS index")
                
                results['embeddings']['faiss_rebuilt'] = True
                
            except Exception as e:
                logger.error(f"Error rebuilding FAISS indices: {e}")
                results['embeddings']['faiss_error'] = str(e)
        
        # Step 3: Pre-compute recommendations
        try:
            logger.info("Pre-computing recommendations...")
            precompute_results = self.precompute_all_candidates(top_k=10)
            results['precomputation'] = precompute_results
        except Exception as e:
            logger.error(f"Error pre-computing recommendations: {e}")
            results['precomputation']['error'] = str(e)
        
        return results

