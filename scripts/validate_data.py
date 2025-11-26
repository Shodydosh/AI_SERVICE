"""Script to validate raw data before processing."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import logging
from src.utils.data_validator import DataValidator
from src.utils.report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_dataset(
    file_path: str,
    dataset_type: str,
    file_format: str = "csv",
    output_report: str = None
) -> bool:
    """
    Validate a dataset file.
    
    Args:
        file_path: Path to the dataset file
        dataset_type: "jd" or "candidate"
        file_format: "csv" or "json"
        output_report: Optional path to save validation report
    
    Returns:
        True if validation passes, False otherwise
    """
    logger.info(f"Validating {dataset_type} dataset: {file_path}")
    
    # Initialize validator
    validator = DataValidator(dataset_type=dataset_type)
    
    # Check file exists
    file_valid, error = validator.validate_file_exists(file_path)
    if not file_valid:
        logger.error(error)
        return False
    
    # Check file format
    format_valid, error = validator.validate_file_format(file_path, file_format)
    if not format_valid:
        logger.error(error)
        return False
    
    # Load data using processor (which handles column mapping)
    try:
        if dataset_type == "jd":
            from src.data_processing.jd_processor import JDProcessor
            processor = JDProcessor(auto_map_columns=True)
            if file_format.lower() == "csv":
                df = processor.load_from_csv(file_path)
            elif file_format.lower() == "json":
                df = processor.load_from_json(file_path)
            else:
                logger.error(f"Unsupported file format: {file_format}")
                return False
        elif dataset_type == "candidate":
            from src.data_processing.candidate_processor import CandidateProcessor
            processor = CandidateProcessor(auto_map_columns=True)
            if file_format.lower() == "csv":
                df = processor.load_from_csv(file_path)
            elif file_format.lower() == "json":
                df = processor.load_from_json(file_path)
            else:
                logger.error(f"Unsupported file format: {file_format}")
                return False
        else:
            logger.error(f"Unknown dataset type: {dataset_type}")
            return False
        
        logger.info(f"Loaded {len(df)} rows from {file_path}")
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        return False
    
    # Run validation
    validation_results = validator.validate_all(df)
    
    # Generate report
    report_generator = ReportGenerator()
    report_text = report_generator.generate_validation_report(
        validation_results,
        output_path=output_report
    )
    
    # Print report to console
    print("\n" + report_text + "\n")
    
    # Save JSON report if output path provided
    if output_report:
        json_report_path = str(Path(output_report).with_suffix('.json'))
        report_generator.save_json_report(validation_results, json_report_path)
    
    # Return validation status
    is_valid = validation_results.get("overall_valid", False)
    
    if is_valid:
        logger.info("✓ Validation passed!")
    else:
        logger.error("✗ Validation failed! Please fix errors before processing.")
    
    return is_valid


def main():
    parser = argparse.ArgumentParser(
        description="Validate JD or candidate datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate JD dataset
  python scripts/validate_data.py --file data/jd_dataset.csv --type jd
  
  # Validate candidate dataset with report
  python scripts/validate_data.py --file data/candidate_dataset.csv --type candidate --report reports/validation_report.txt
  
  # Validate JSON file
  python scripts/validate_data.py --file data/jd_dataset.json --type jd --format json
        """
    )
    
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the dataset file (CSV or JSON)"
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
        "--report",
        type=str,
        default=None,
        help="Optional path to save validation report"
    )
    
    args = parser.parse_args()
    
    # Validate dataset
    is_valid = validate_dataset(
        file_path=args.file,
        dataset_type=args.type,
        file_format=args.format,
        output_report=args.report
    )
    
    # Exit with appropriate code
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()

