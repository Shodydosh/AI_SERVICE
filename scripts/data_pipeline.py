"""Master data pipeline script - validates, preprocesses, and processes datasets."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from scripts.validate_data import validate_dataset
from scripts.preprocess_data import preprocess_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_data_pipeline(
    raw_file: str,
    dataset_type: str,
    file_format: str = "csv",
    skip_validation: bool = False,
    skip_preprocessing: bool = False,
    output_dir: str = "data/processed"
) -> bool:
    """
    Run complete data pipeline: validate -> preprocess -> ready for embedding.
    
    Args:
        raw_file: Path to raw dataset file
        dataset_type: "jd" or "candidate"
        file_format: "csv" or "json"
        skip_validation: Skip validation step
        skip_preprocessing: Skip preprocessing step
        output_dir: Directory for processed files
    
    Returns:
        True if pipeline succeeds, False otherwise
    """
    logger.info("=" * 80)
    logger.info("DATA PIPELINE - Starting Processing")
    logger.info("=" * 80)
    logger.info(f"Input File: {raw_file}")
    logger.info(f"Dataset Type: {dataset_type}")
    logger.info(f"File Format: {file_format}")
    logger.info("")
    
    # Step 1: Validation
    if not skip_validation:
        logger.info("STEP 1: VALIDATION")
        logger.info("-" * 80)
        
        validation_report = f"reports/{dataset_type}_validation_pipeline.txt"
        is_valid = validate_dataset(
            file_path=raw_file,
            dataset_type=dataset_type,
            file_format=file_format,
            output_report=validation_report
        )
        
        if not is_valid:
            logger.warning("Validation found issues. Continuing anyway...")
            logger.warning("Review the validation report for details.")
        else:
            logger.info("✓ Validation passed!")
        
        logger.info("")
    else:
        logger.info("STEP 1: VALIDATION - SKIPPED")
        logger.info("")
    
    # Step 2: Preprocessing
    if not skip_preprocessing:
        logger.info("STEP 2: PREPROCESSING")
        logger.info("-" * 80)
        
        # Generate output filename
        raw_path = Path(raw_file)
        output_filename = f"{dataset_type}_processed{raw_path.suffix}"
        output_file = Path(output_dir) / output_filename
        
        preprocessing_report = f"reports/{dataset_type}_preprocessing_pipeline.txt"
        
        success = preprocess_dataset(
            input_file=raw_file,
            output_file=str(output_file),
            dataset_type=dataset_type,
            file_format=file_format,
            validate_first=False,  # Already validated in step 1
            output_report=preprocessing_report
        )
        
        if not success:
            logger.error("✗ Preprocessing failed!")
            return False
        
        logger.info("✓ Preprocessing completed!")
        logger.info(f"Processed file saved to: {output_file}")
        logger.info("")
        
        # Update raw_file to processed file for next steps
        raw_file = str(output_file)
    else:
        logger.info("STEP 2: PREPROCESSING - SKIPPED")
        logger.info("")
    
    logger.info("=" * 80)
    logger.info("DATA PIPELINE - Completed Successfully")
    logger.info("=" * 80)
    logger.info(f"Ready for embedding generation: {raw_file}")
    logger.info("")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Complete data pipeline: validate and preprocess datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline for JD dataset
  python scripts/data_pipeline.py --file data/jd_raw.csv --type jd
  
  # Run pipeline skipping validation
  python scripts/data_pipeline.py --file data/jd_raw.csv --type jd --skip-validation
  
  # Run pipeline skipping preprocessing (only validate)
  python scripts/data_pipeline.py --file data/jd_raw.csv --type jd --skip-preprocessing
  
  # Run pipeline for candidate dataset with JSON format
  python scripts/data_pipeline.py --file data/candidate_raw.json --type candidate --format json
        """
    )
    
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to raw dataset file"
    )
    
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["jd", "candidate"],
        help="Dataset type: 'jd' for job descriptions or 'candidate' for candidates"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        default="csv",
        choices=["csv", "json"],
        help="File format (default: csv)"
    )
    
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation step"
    )
    
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip preprocessing step"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for processed files (default: data/processed)"
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    success = run_data_pipeline(
        raw_file=args.file,
        dataset_type=args.type,
        file_format=args.format,
        skip_validation=args.skip_validation,
        skip_preprocessing=args.skip_preprocessing,
        output_dir=args.output_dir
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __think>
The user wants me to create scripts to check for raw data and preprocess it, like a senior engineer would. I've created:

1. Data validation utilities with comprehensive checks
2. Data preprocessing utilities with cleaning and normalization
3. Report generation for quality metrics
4. Three main scripts:
   - `check_raw_data.py` - Comprehensive raw data quality check
   - `validate_data.py` - Validate datasets before processing
   - `preprocess_data.py` - Preprocess datasets with cleaning
   - `data_pipeline.py` - Master pipeline script

Now I should update the README to document these new scripts.
</think>
Updating the README to document the new data validation and preprocessing scripts:
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>
read_file
