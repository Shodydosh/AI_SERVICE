"""Run embedding refresh scheduler as a background worker."""
import sys
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.embedding_scheduler import run_scheduler_worker
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_scheduler.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for scheduler worker."""
    parser = argparse.ArgumentParser(description='Embedding refresh scheduler worker')
    parser.add_argument('--refresh-interval', type=float, default=12.0,
                       help='Refresh interval in hours (default: 12)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for processing (default: 100)')
    parser.add_argument('--check-interval', type=int, default=300,
                       help='Check interval in seconds (default: 300 = 5 minutes)')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit (for cron jobs)')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("EMBEDDING REFRESH SCHEDULER")
    logger.info("=" * 80)
    logger.info(f"Refresh interval: {args.refresh_interval} hours")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Check interval: {args.check_interval} seconds")
    logger.info(f"Mode: {'Once' if args.once else 'Continuous'}")
    logger.info("=" * 80)
    
    if args.once:
        # Run once (for cron jobs)
        from src.services.embedding_scheduler import EmbeddingScheduler
        scheduler = EmbeddingScheduler(
            refresh_interval_hours=args.refresh_interval,
            batch_size=args.batch_size
        )
        scheduler.run_refresh_cycle()
    else:
        # Run continuously
        run_scheduler_worker(
            refresh_interval_hours=args.refresh_interval,
            batch_size=args.batch_size,
            check_interval_seconds=args.check_interval
        )


if __name__ == '__main__':
    main()

