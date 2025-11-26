"""Script to preprocess raw data before embedding generation."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import logging
from src.utils.data_preprocessor import DataPreprocessor
from src.utils.data_validator import DataValidator
from src.utils.report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def preprocess_dataset(
    input_file: str,
    output_file: str,
    dataset_type: str,
    file_format: str = "csv",
    validate_first: bool = True,
    output_report: str = None
) -> bool:
    """
    Preprocess a dataset file.
    
    Args:
        input_file: Path to input dataset file
        output_file: Path to save preprocessed dataset
        dataset_type: "jd" or "candidate"
        file_format: "csv" or "json"
        validate_first: Whether to validate before preprocessing
        output_report: Optional path to save preprocessing report
    
    Returns:
        True if preprocessing succeeds, False otherwise
    """
    logger.info(f"Preprocessing {dataset_type} dataset: {input_file}")
    
    # Validate first if requested
    if validate_first:
        logger.info("Running validation before preprocessing...")
        validator = DataValidator(dataset_type=dataset_type)
        
        file_valid, error = validator.validate_file_exists(input_file)
        if not file_valid:
            logger.error(error)
            return False
        
        try:
            if file_format.lower() == "csv":
                df = pd.read_csv(input_file)
            elif file_format.lower() == "json":
                df = pd.read_json(input_file)
            else:
                logger.error(f"Unsupported file format: {file_format}")
                return False
            
            validation_results = validator.validate_all(df)
            
            if not validation_results.get("overall_valid", False):
                logger.warning("Validation found issues. Continuing with preprocessing...")
                logger.warning("Consider fixing errors for best results.")
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            return False
    
    # Load data
    try:
        if file_format.lower() == "csv":
            df = pd.read_csv(input_file)
        elif file_format.lower() == "json":
            df = pd.read_json(input_file)
        else:
            logger.error(f"Unsupported file format: {file_format}")
            return False
        
        logger.info(f"Loaded {len(df)} rows from {input_file}")
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        return False
    
    # Preprocess data
    try:
        preprocessor = DataPreprocessor(dataset_type=dataset_type)
        df_processed = preprocessor.preprocess(df)
        
        logger.info(f"Preprocessed {len(df_processed)} rows")
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return False
    
    # Save preprocessed data
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_format.lower() == "csv":
            df_processed.to_csv(output_path, index=False)
        elif file_format.lower() == "json":
            df_processed.to_json(output_path, orient='records', indent=2)
        
        logger.info(f"Preprocessed data saved to: {output_file}")
    except Exception as e:
        logger.error(f"Error saving preprocessed data: {e}")
        return False
    
    # Generate report
    if output_report:
        report_generator = ReportGenerator()
        preprocessing_stats = preprocessor.get_preprocessing_stats()
        report_text = report_generator.generate_preprocessing_report(
            preprocessing_stats,
            output_path=output_report
        )
        print("\n" + report_text + "\n")
    
    logger.info("✓ Preprocessing completed successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess JD or candidate datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess JD dataset
  python scripts/preprocess_data.py --input data/jd_raw.csv --output data/jd_processed.csv --type jd
  
  # Preprocess candidate dataset with validation
  python scripts/preprocess_data.py --input data/candidate_raw.csv --output data/candidate_processed.csv --type candidate --validate
  
  # Preprocess without validation
  python scripts/preprocess_data.py --input data/jd_raw.csv --output data/jd_processed.csv --type jd --no-validate
  
  # Preprocess JSON file with report
  python scripts/preprocess_data.py --input data/jd_raw.json --output data/jd_processed.json --type jd --format json --report reports/preprocessing_report.txt
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input dataset file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save preprocessed dataset"
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
        "--validate",
        action="store_true",
        default=True,
        help="Validate data before preprocessing (default: True)"
    )
    
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Skip validation before preprocessing"
    )
    
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Optional path to save preprocessing report"
    )
    
    args = parser.parse_args()
    
    # Preprocess dataset
    success = preprocess_dataset(
        input_file=args.input,
        output_file=args.output,
        dataset_type=args.type,
        file_format=args.format,
        validate_first=args.validate,
        output_report=args.report
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

