"""Background scheduler for periodic embedding refresh (12-hour cycle)."""
import time
import logging
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import SessionLocal, get_db_url
from src.services.embedding_service import OptimizedEmbeddingService

logger = logging.getLogger(__name__)


class EmbeddingScheduler:
    """
    Background scheduler for periodic embedding refresh.
    
    Runs every 12 hours to refresh embeddings that are older than 12 hours.
    Designed to run as a background worker without blocking realtime queries.
    """
    
    def __init__(
        self,
        refresh_interval_hours: float = 12.0,
        batch_size: int = 100,
        max_items_per_cycle: int = 10000
    ):
        """
        Initialize scheduler.
        
        Args:
            refresh_interval_hours: How often to run refresh cycle (default: 12 hours)
            batch_size: Batch size for processing (default: 100)
            max_items_per_cycle: Maximum items to process per cycle (default: 10000)
        """
        self.refresh_interval_hours = refresh_interval_hours
        self.batch_size = batch_size
        self.max_items_per_cycle = max_items_per_cycle
        self.running = False
        logger.info(f"EmbeddingScheduler initialized: refresh every {refresh_interval_hours} hours")
    
    def run_refresh_cycle(self, db: Optional[Session] = None) -> Dict[str, int]:
        """
        Run one refresh cycle.
        
        Args:
            db: Optional database session (creates new if None)
            
        Returns:
            Dict with statistics: {'candidates_processed', 'jobs_processed', 'duration_seconds'}
        """
        start_time = time.time()
        stats = {
            'candidates_processed': 0,
            'jobs_processed': 0,
            'duration_seconds': 0
        }
        
        if db is None:
            db = SessionLocal()
            close_db = True
        else:
            close_db = False
        
        try:
            service = OptimizedEmbeddingService(db, batch_size=self.batch_size)
            
            logger.info("=" * 80)
            logger.info("STARTING EMBEDDING REFRESH CYCLE")
            logger.info("=" * 80)
            
            # Process candidates
            logger.info("Fetching candidates needing refresh...")
            candidates = service.batch_get_candidates_needing_refresh(limit=self.max_items_per_cycle)
            logger.info(f"Found {len(candidates)} candidates needing refresh")
            
            if candidates:
                processed = service.batch_process_candidates(candidates)
                stats['candidates_processed'] = processed
                logger.info(f"Processed {processed} candidates")
            
            # Process jobs
            logger.info("Fetching jobs needing refresh...")
            jobs = service.batch_get_jobs_needing_refresh(limit=self.max_items_per_cycle)
            logger.info(f"Found {len(jobs)} jobs needing refresh")
            
            if jobs:
                processed = service.batch_process_jobs(jobs)
                stats['jobs_processed'] = processed
                logger.info(f"Processed {processed} jobs")
            
            # Clear expired cache
            service.cache.clear_expired()
            cache_stats = service.cache.get_stats()
            logger.info(f"Cache stats: {cache_stats}")
            
            duration = time.time() - start_time
            stats['duration_seconds'] = duration
            
            logger.info("=" * 80)
            logger.info(f"REFRESH CYCLE COMPLETED in {duration:.2f} seconds")
            logger.info(f"  Candidates: {stats['candidates_processed']}")
            logger.info(f"  Jobs: {stats['jobs_processed']}")
            logger.info("=" * 80)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error in refresh cycle: {e}", exc_info=True)
            raise
        finally:
            if close_db:
                db.close()
    
    def run_continuous(self, check_interval_seconds: int = 300):
        """
        Run scheduler continuously (for background worker).
        
        Args:
            check_interval_seconds: How often to check if refresh is needed (default: 5 minutes)
        """
        self.running = True
        last_refresh = None
        
        logger.info("Starting continuous embedding scheduler...")
        logger.info(f"Refresh interval: {self.refresh_interval_hours} hours")
        logger.info(f"Check interval: {check_interval_seconds} seconds")
        
        try:
            while self.running:
                now = datetime.now()
                
                # Check if refresh is needed
                should_refresh = False
                if last_refresh is None:
                    should_refresh = True
                    logger.info("First refresh cycle")
                else:
                    time_since_refresh = now - last_refresh
                    if time_since_refresh >= timedelta(hours=self.refresh_interval_hours):
                        should_refresh = True
                        logger.info(f"Refresh interval reached ({time_since_refresh})")
                
                if should_refresh:
                    try:
                        stats = self.run_refresh_cycle()
                        last_refresh = datetime.now()
                        logger.info(f"Next refresh scheduled for: {last_refresh + timedelta(hours=self.refresh_interval_hours)}")
                    except Exception as e:
                        logger.error(f"Error in refresh cycle: {e}", exc_info=True)
                        # Continue running even if one cycle fails
                
                # Sleep until next check
                time.sleep(check_interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
        finally:
            self.running = False
            logger.info("Scheduler stopped")
    
    def stop(self):
        """Stop the scheduler."""
        self.running = False
        logger.info("Stopping scheduler...")


def run_scheduler_worker(
    refresh_interval_hours: float = 12.0,
    batch_size: int = 100,
    check_interval_seconds: int = 300
):
    """
    Run scheduler as a background worker.
    
    This function is designed to be run as a separate process/daemon.
    """
    scheduler = EmbeddingScheduler(
        refresh_interval_hours=refresh_interval_hours,
        batch_size=batch_size
    )
    scheduler.run_continuous(check_interval_seconds=check_interval_seconds)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Embedding refresh scheduler')
    parser.add_argument('--refresh-interval', type=float, default=12.0,
                       help='Refresh interval in hours (default: 12)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for processing (default: 100)')
    parser.add_argument('--check-interval', type=int, default=300,
                       help='Check interval in seconds (default: 300)')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit (for cron jobs)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    scheduler = EmbeddingScheduler(
        refresh_interval_hours=args.refresh_interval,
        batch_size=args.batch_size
    )
    
    if args.once:
        # Run once (for cron jobs)
        scheduler.run_refresh_cycle()
    else:
        # Run continuously
        scheduler.run_continuous(check_interval_seconds=args.check_interval)








