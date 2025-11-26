"""Clean data utilities - simplified interface."""
import pandas as pd
from typing import Optional
import logging
from .data_validator import DataValidator
from .data_preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)


def clean_dataset(
    input_file: str,
    output_file: str,
    dataset_type: str,
    file_format: str = "csv",
    validate: bool = True
) -> bool:
    """
    Clean dataset - validates and preprocesses in one step.
    
    Args:
        input_file: Path to input file
        output_file: Path to save cleaned data
        dataset_type: "jd" or "candidate"
        file_format: "csv" or "json"
        validate: Whether to validate before cleaning
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load data using processor (which handles column mapping)
        if dataset_type == "jd":
            from src.data_processing.jd_processor import JDProcessor
            processor = JDProcessor(auto_map_columns=True)
            if file_format.lower() == "csv":
                df = processor.load_from_csv(input_file)
            elif file_format.lower() == "json":
                df = processor.load_from_json(input_file)
            else:
                raise ValueError(f"Unsupported format: {file_format}")
        elif dataset_type == "candidate":
            from src.data_processing.candidate_processor import CandidateProcessor
            processor = CandidateProcessor(auto_map_columns=True)
            if file_format.lower() == "csv":
                df = processor.load_from_csv(input_file)
            elif file_format.lower() == "json":
                df = processor.load_from_json(input_file)
            else:
                raise ValueError(f"Unsupported format: {file_format}")
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        logger.info(f"Loaded {len(df)} records from {input_file}")
        
        # Validate if requested
        if validate:
            logger.info("Validating data...")
            validator = DataValidator(dataset_type=dataset_type)
            validation_results = validator.validate_all(df)
            
            if not validation_results.get("overall_valid", False):
                errors = validation_results.get("structure", {}).get("errors", [])
                errors.extend(validation_results.get("quality", {}).get("errors", []))
                if errors:
                    logger.warning(f"Validation found {len(errors)} errors, but continuing...")
                    for error in errors[:5]:  # Show first 5 errors
                        logger.warning(f"  - {error}")
        
        # Preprocess
        logger.info("Cleaning data...")
        preprocessor = DataPreprocessor(dataset_type=dataset_type)
        df_cleaned = preprocessor.preprocess(df)
        
        # Save cleaned data
        from pathlib import Path
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_format.lower() == "csv":
            df_cleaned.to_csv(output_path, index=False)
        else:
            df_cleaned.to_json(output_path, orient='records', indent=2)
        
        logger.info(f"✓ Cleaned data saved to {output_file}")
        logger.info(f"  Records: {len(df)} → {len(df_cleaned)}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error cleaning data: {e}")
        return False


def quick_clean(input_file: str, dataset_type: str) -> Optional[pd.DataFrame]:
    """
    Quick clean - returns cleaned DataFrame without saving.
    
    Args:
        input_file: Path to input file
        dataset_type: "jd" or "candidate"
    
    Returns:
        Cleaned DataFrame or None if error
    """
    try:
        # Load
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        elif input_file.endswith('.json'):
            df = pd.read_json(input_file)
        else:
            raise ValueError("Unsupported file format")
        
        # Clean
        preprocessor = DataPreprocessor(dataset_type=dataset_type)
        df_cleaned = preprocessor.preprocess(df)
        
        return df_cleaned
    
    except Exception as e:
        logger.error(f"Error in quick clean: {e}")
        return None

