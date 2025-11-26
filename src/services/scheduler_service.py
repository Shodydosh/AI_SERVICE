"""Scheduler service for periodic tasks."""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy.orm import Session
import logging
from datetime import datetime

from src.database.connection import SessionLocal
from src.services.precompute_service import PrecomputeService

logger = logging.getLogger(__name__)


class SchedulerService:
    """Service for managing scheduled tasks."""
    
    def __init__(self):
        """Initialize scheduler service."""
        self.scheduler = BackgroundScheduler()
        self.is_running = False
    
    def start(self):
        """Start the scheduler."""
        if not self.is_running:
            self.scheduler.start()
            self.is_running = True
            logger.info("Scheduler started")
    
    def stop(self):
        """Stop the scheduler."""
        if self.is_running:
            self.scheduler.shutdown()
            self.is_running = False
            logger.info("Scheduler stopped")
    
    def add_regeneration_job(
        self,
        hours: int = 12,
        jd_file: str = None,
        candidate_file: str = None
    ):
        """
        Add job to regenerate embeddings and recompute recommendations.
        
        Args:
            hours: Interval in hours (default: 12)
            jd_file: Path to JD dataset file (optional)
            candidate_file: Path to candidate dataset file (optional)
        """
        def run_regeneration():
            """Run the regeneration workflow."""
            logger.info("=" * 80)
            logger.info(f"SCHEDULED JOB STARTED at {datetime.now()}")
            logger.info("=" * 80)
            
            db: Session = SessionLocal()
            try:
                precompute_service = PrecomputeService(db)
                results = precompute_service.regenerate_embeddings_and_recompute(
                    jd_file=jd_file,
                    candidate_file=candidate_file,
                    rebuild_faiss=True
                )
                
                logger.info("=" * 80)
                logger.info(f"SCHEDULED JOB COMPLETED at {datetime.now()}")
                logger.info(f"Results: {results}")
                logger.info("=" * 80)
                
            except Exception as e:
                logger.error(f"Error in scheduled job: {e}", exc_info=True)
            finally:
                db.close()
        
        # Add job to run every N hours
        trigger = IntervalTrigger(hours=hours)
        self.scheduler.add_job(
            func=run_regeneration,
            trigger=trigger,
            id='regeneration_job',
            name='Regenerate embeddings and recompute recommendations',
            replace_existing=True,
            max_instances=1  # Only one instance at a time
        )
        
        logger.info(f"Added regeneration job to run every {hours} hours")
        logger.info(f"Next run: {self.scheduler.get_job('regeneration_job').next_run_time}")
    
    def add_precompute_job(self, hours: int = 12):
        """
        Add job to only recompute recommendations (without regenerating embeddings).
        
        Args:
            hours: Interval in hours (default: 12)
        """
        def run_precompute():
            """Run the pre-computation workflow."""
            logger.info("=" * 80)
            logger.info(f"PRE-COMPUTE JOB STARTED at {datetime.now()}")
            logger.info("=" * 80)
            
            db: Session = SessionLocal()
            try:
                precompute_service = PrecomputeService(db)
                results = precompute_service.precompute_all_candidates(top_k=10)
                
                logger.info("=" * 80)
                logger.info(f"PRE-COMPUTE JOB COMPLETED at {datetime.now()}")
                logger.info(f"Results: {results}")
                logger.info("=" * 80)
                
            except Exception as e:
                logger.error(f"Error in pre-compute job: {e}", exc_info=True)
            finally:
                db.close()
        
        # Add job to run every N hours
        trigger = IntervalTrigger(hours=hours)
        self.scheduler.add_job(
            func=run_precompute,
            trigger=trigger,
            id='precompute_job',
            name='Pre-compute recommendations',
            replace_existing=True,
            max_instances=1
        )
        
        logger.info(f"Added pre-compute job to run every {hours} hours")
        logger.info(f"Next run: {self.scheduler.get_job('precompute_job').next_run_time}")
    
    def get_job_status(self) -> dict:
        """Get status of all scheduled jobs."""
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None,
                'trigger': str(job.trigger)
            })
        
        return {
            'is_running': self.is_running,
            'jobs': jobs
        }

