"""Process all raw data files and save to processed folder."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from scripts.process_to_processed import process_to_processed_folder

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def process_all_raw_data():
    """Process all CSV/JSON files in data/raw/ folder."""
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    
    if not raw_dir.exists():
        logger.error(f"Raw data directory not found: {raw_dir}")
        return False
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all data files
    csv_files = list(raw_dir.glob("*.csv"))
    json_files = list(raw_dir.glob("*.json"))
    all_files = csv_files + json_files
    
    if not all_files:
        logger.warning(f"No data files found in {raw_dir}")
        return False
    
    logger.info(f"Found {len(all_files)} file(s) to process")
    logger.info("")
    
    success_count = 0
    failed_count = 0
    
    # Process each file
    for file_path in all_files:
        logger.info("=" * 80)
        logger.info(f"Processing: {file_path.name}")
        logger.info("=" * 80)
        
        # Try to detect dataset type from filename
        filename_lower = file_path.name.lower()
        if "job" in filename_lower or "jd" in filename_lower:
            dataset_type = "jd"
        elif "candidate" in filename_lower or "resume" in filename_lower:
            dataset_type = "candidate"
        else:
            logger.warning(f"Could not determine dataset type for {file_path.name}")
            logger.warning("Skipping... (use process_to_processed.py with --type flag)")
            failed_count += 1
            continue
        
        # Determine file format
        file_format = "csv" if file_path.suffix == ".csv" else "json"
        
        # Process file
        success = process_to_processed_folder(
            input_file=str(file_path),
            dataset_type=dataset_type,
            file_format=file_format,
            validate=True
        )
        
        if success:
            success_count += 1
        else:
            failed_count += 1
        
        logger.info("")
    
    # Summary
    logger.info("=" * 80)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total files: {len(all_files)}")
    logger.info(f"Successfully processed: {success_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info("")
    
    if success_count > 0:
        logger.info(f"Processed files saved to: {processed_dir}")
    
    return failed_count == 0


if __name__ == "__main__":
    import sys
    success = process_all_raw_data()
    sys.exit(0 if success else 1)

