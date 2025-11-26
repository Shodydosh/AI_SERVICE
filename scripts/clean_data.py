"""Simple script to clean data."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from src.utils.clean_data import clean_dataset

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Clean dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean JD dataset
  python scripts/clean_data.py --input data/raw/job_data.csv --output data/processed/jd_clean.csv --type jd
  
  # Clean candidate dataset
  python scripts/clean_data.py --input data/raw/candidates.csv --output data/processed/candidates_clean.csv --type candidate
  
  # Skip validation
  python scripts/clean_data.py --input data/raw/job_data.csv --output data/processed/jd_clean.csv --type jd --no-validate
        """
    )
    
    parser.add_argument("--input", required=True, help="Input file path")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: auto-saves to data/processed/)"
    )
    parser.add_argument("--type", required=True, choices=["jd", "candidate"], help="Dataset type")
    parser.add_argument("--format", default="csv", choices=["csv", "json"], help="File format")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation")
    
    args = parser.parse_args()
    
    # Auto-determine output path if not provided
    output_file = args.output
    if output_file is None:
        from pathlib import Path
        input_path = Path(args.input)
        
        # If input is in raw folder, put output in processed folder
        if "raw" in str(input_path):
            output_path = Path(str(input_path).replace("raw", "processed"))
        else:
            # Otherwise, create processed folder structure
            processed_dir = Path("data/processed")
            processed_dir.mkdir(parents=True, exist_ok=True)
            output_path = processed_dir / input_path.name
        
        output_file = str(output_path)
        logger.info(f"Output will be saved to: {output_file}")
    
    success = clean_dataset(
        input_file=args.input,
        output_file=output_file,
        dataset_type=args.type,
        file_format=args.format,
        validate=not args.no_validate
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

