"""Process data and save to processed folder."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from src.utils.clean_data import clean_dataset

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def process_to_processed_folder(
    input_file: str,
    dataset_type: str,
    file_format: str = "csv",
    validate: bool = True
):
    """
    Process data file and save to processed folder.
    
    Args:
        input_file: Path to input file (can be in data/raw/ or anywhere)
        dataset_type: "jd" or "candidate"
        file_format: "csv" or "json"
        validate: Whether to validate before processing
    """
    input_path = Path(input_file)
    
    # Determine output path
    if "raw" in str(input_path):
        # If input is in raw folder, put output in processed folder
        output_path = Path(str(input_path).replace("raw", "processed"))
    else:
        # Otherwise, create processed folder structure
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = processed_dir / input_path.name
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing: {input_file}")
    logger.info(f"Output: {output_path}")
    
    # Process data
    success = clean_dataset(
        input_file=str(input_path),
        output_file=str(output_path),
        dataset_type=dataset_type,
        file_format=file_format,
        validate=validate
    )
    
    if success:
        logger.info(f"✓ Processed data saved to: {output_path}")
        return True
    else:
        logger.error("✗ Processing failed")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Process data and save to processed folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process JD data from raw folder
  python scripts/process_to_processed.py --input data/raw/job_data.csv --type jd
  
  # Process candidate data
  python scripts/process_to_processed.py --input data/raw/candidates_dataset.csv --type candidate
  
  # Process without validation
  python scripts/process_to_processed.py --input data/raw/job_data.csv --type jd --no-validate
  
  # Process JSON file
  python scripts/process_to_processed.py --input data/raw/job_data.json --type jd --format json
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file path (e.g., data/raw/job_data.csv)"
    )
    
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["jd", "candidate"],
        help="Dataset type: 'jd' or 'candidate'"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        default="csv",
        choices=["csv", "json"],
        help="File format (default: csv)"
    )
    
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation before processing"
    )
    
    args = parser.parse_args()
    
    success = process_to_processed_folder(
        input_file=args.input,
        dataset_type=args.type,
        file_format=args.format,
        validate=not args.no_validate
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

