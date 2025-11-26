"""Comprehensive script to check raw data quality and generate reports."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import logging
from datetime import datetime
from src.utils.data_validator import DataValidator
from src.utils.report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_raw_data(
    file_path: str,
    dataset_type: str,
    file_format: str = "csv",
    output_dir: str = "reports"
) -> bool:
    """
    Comprehensive raw data quality check.
    
    Args:
        file_path: Path to the dataset file
        dataset_type: "jd" or "candidate"
        file_format: "csv" or "json"
        output_dir: Directory to save reports
    
    Returns:
        True if data quality is acceptable, False otherwise
    """
    logger.info(f"Checking raw data quality for {dataset_type} dataset: {file_path}")
    
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
        
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        return False
    
    # Run comprehensive validation
    logger.info("Running comprehensive validation...")
    validation_results = validator.validate_all(df)
    
    # Generate reports
    report_generator = ReportGenerator()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for report files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"{dataset_type}_validation_report_{timestamp}.txt"
    json_filename = f"{dataset_type}_validation_report_{timestamp}.json"
    
    report_path = output_path / report_filename
    json_path = output_path / json_filename
    
    # Generate text report
    report_text = report_generator.generate_validation_report(
        validation_results,
        output_path=str(report_path)
    )
    
    # Save JSON report
    report_generator.save_json_report(validation_results, str(json_path))
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("RAW DATA QUALITY CHECK SUMMARY")
    print("=" * 80)
    print(f"File: {file_path}")
    print(f"Dataset Type: {dataset_type.upper()}")
    print(f"Total Rows: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Overall Status: {'✓ VALID' if validation_results.get('overall_valid') else '✗ INVALID'}")
    
    summary = validation_results.get("summary", {})
    print(f"Errors: {summary.get('total_errors', 0)}")
    print(f"Warnings: {summary.get('total_warnings', 0)}")
    print(f"\nDetailed reports saved to:")
    print(f"  - {report_path}")
    print(f"  - {json_path}")
    print("=" * 80 + "\n")
    
    # Print full report
    print(report_text)
    
    # Return status
    is_valid = validation_results.get("overall_valid", False)
    
    if is_valid:
        logger.info("✓ Data quality check passed!")
    else:
        logger.warning("⚠ Data quality check found issues. Review the report for details.")
    
    return is_valid


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive raw data quality check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check JD dataset
  python scripts/check_raw_data.py --file data/jd_dataset.csv --type jd
  
  # Check candidate dataset with custom output directory
  python scripts/check_raw_data.py --file data/candidate_dataset.csv --type candidate --output reports/quality_checks
  
  # Check JSON file
  python scripts/check_raw_data.py --file data/jd_dataset.json --type jd --format json
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
        "--output",
        type=str,
        default="reports",
        help="Output directory for reports (default: reports)"
    )
    
    args = parser.parse_args()
    
    # Check raw data
    is_valid = check_raw_data(
        file_path=args.file,
        dataset_type=args.type,
        file_format=args.format,
        output_dir=args.output
    )
    
    # Exit with appropriate code (0 for success, 1 for issues found)
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()

