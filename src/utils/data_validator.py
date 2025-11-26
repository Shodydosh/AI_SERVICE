"""Data validation utilities."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DataValidator:
    """Comprehensive data validator for JD and candidate datasets."""
    
    def __init__(self, dataset_type: str = "jd"):
        """
        Initialize validator.
        
        Args:
            dataset_type: "jd" for job descriptions or "candidate" for candidates
        """
        self.dataset_type = dataset_type.lower()
        self.required_fields = self._get_required_fields()
        self.optional_fields = self._get_optional_fields()
        self.validation_results = {}
    
    def _get_required_fields(self) -> List[str]:
        """Get required fields based on dataset type."""
        if self.dataset_type == "jd":
            return ["job_id", "title", "description"]
        elif self.dataset_type == "candidate":
            return ["candidate_id"]
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
    
    def _get_optional_fields(self) -> List[str]:
        """Get optional fields based on dataset type."""
        if self.dataset_type == "jd":
            return ["company", "requirements", "location", "skills"]
        elif self.dataset_type == "candidate":
            return ["name", "email", "skills", "experience", "education", "summary", "resume_text"]
        else:
            return []
    
    def validate_file_exists(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """Check if file exists and is readable."""
        try:
            path = Path(file_path)
            if not path.exists():
                return False, f"File does not exist: {file_path}"
            if not path.is_file():
                return False, f"Path is not a file: {file_path}"
            return True, None
        except Exception as e:
            return False, f"Error checking file: {str(e)}"
    
    def validate_file_format(self, file_path: str, expected_format: str) -> Tuple[bool, Optional[str]]:
        """Validate file format."""
        path = Path(file_path)
        actual_format = path.suffix.lower()
        
        format_map = {
            "csv": ".csv",
            "json": ".json"
        }
        
        expected_ext = format_map.get(expected_format.lower(), expected_format)
        if not actual_format.startswith(expected_ext):
            return False, f"Expected {expected_format} format, got {actual_format}"
        
        return True, None
    
    def validate_structure(self, df: pd.DataFrame) -> Dict[str, any]:
        """Validate DataFrame structure."""
        results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        # Check if DataFrame is empty
        if df.empty:
            results["is_valid"] = False
            results["errors"].append("DataFrame is empty")
            return results
        
        results["stats"]["total_rows"] = len(df)
        results["stats"]["total_columns"] = len(df.columns)
        
        # Check required columns
        missing_required = [col for col in self.required_fields if col not in df.columns]
        if missing_required:
            results["is_valid"] = False
            results["errors"].append(f"Missing required columns: {missing_required}")
        
        # Check for unexpected columns (warn only)
        all_expected = self.required_fields + self.optional_fields
        unexpected = [col for col in df.columns if col not in all_expected]
        if unexpected:
            results["warnings"].append(f"Unexpected columns found: {unexpected}")
        
        return results
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """Validate data quality metrics."""
        results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "quality_metrics": {}
        }
        
        # Check for duplicates
        if self.dataset_type == "jd":
            id_col = "job_id"
        else:
            id_col = "candidate_id"
        
        if id_col in df.columns:
            duplicates = df[df.duplicated(subset=[id_col], keep=False)]
            if not duplicates.empty:
                results["is_valid"] = False
                results["errors"].append(
                    f"Found {len(duplicates)} duplicate {id_col} values"
                )
                results["quality_metrics"]["duplicate_count"] = len(duplicates)
        
        # Check for missing values in required fields
        for field in self.required_fields:
            if field in df.columns:
                missing_count = df[field].isna().sum()
                empty_count = (df[field].astype(str).str.strip() == "").sum()
                total_missing = missing_count + empty_count
                
                if total_missing > 0:
                    results["is_valid"] = False
                    results["errors"].append(
                        f"Required field '{field}' has {total_missing} missing/empty values"
                    )
                    results["quality_metrics"][f"{field}_missing"] = total_missing
        
        # Check for missing values in optional fields (warnings)
        for field in self.optional_fields:
            if field in df.columns:
                missing_count = df[field].isna().sum()
                empty_count = (df[field].astype(str).str.strip() == "").sum()
                total_missing = missing_count + empty_count
                
                if total_missing > 0:
                    missing_pct = (total_missing / len(df)) * 100
                    results["warnings"].append(
                        f"Optional field '{field}' has {total_missing} ({missing_pct:.1f}%) missing/empty values"
                    )
                    results["quality_metrics"][f"{field}_missing"] = total_missing
        
        # Check data types
        if self.dataset_type == "jd" and "job_id" in df.columns:
            if not df["job_id"].dtype in [object, str]:
                results["warnings"].append("job_id should be string type")
        
        if self.dataset_type == "candidate" and "candidate_id" in df.columns:
            if not df["candidate_id"].dtype in [object, str]:
                results["warnings"].append("candidate_id should be string type")
        
        # Check text field lengths
        text_fields = ["description", "requirements", "summary", "experience", "resume_text"]
        for field in text_fields:
            if field in df.columns:
                lengths = df[field].astype(str).str.len()
                min_len = lengths.min()
                max_len = lengths.max()
                avg_len = lengths.mean()
                
                results["quality_metrics"][f"{field}_length"] = {
                    "min": int(min_len) if not pd.isna(min_len) else 0,
                    "max": int(max_len) if not pd.isna(max_len) else 0,
                    "avg": float(avg_len) if not pd.isna(avg_len) else 0.0
                }
                
                # Warn if text is too short
                if min_len < 10 and field in self.required_fields:
                    results["warnings"].append(
                        f"Field '{field}' has very short values (min: {min_len} chars)"
                    )
        
        # Check email format if email column exists
        if "email" in df.columns:
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            invalid_emails = df[df["email"].notna() & ~df["email"].astype(str).str.match(email_pattern, na=False)]
            if len(invalid_emails) > 0:
                results["warnings"].append(
                    f"Found {len(invalid_emails)} invalid email formats"
                )
                results["quality_metrics"]["invalid_emails"] = len(invalid_emails)
        
        return results
    
    def validate_completeness(self, df: pd.DataFrame) -> Dict[str, any]:
        """Check data completeness for embedding generation."""
        results = {
            "is_valid": True,
            "warnings": [],
            "completeness_scores": {}
        }
        
        if self.dataset_type == "jd":
            text_fields = ["title", "description", "requirements", "skills"]
        else:
            text_fields = ["summary", "skills", "experience", "education", "resume_text"]
        
        for field in text_fields:
            if field in df.columns:
                non_empty = df[field].notna() & (df[field].astype(str).str.strip() != "")
                completeness = (non_empty.sum() / len(df)) * 100
                results["completeness_scores"][field] = completeness
                
                if completeness < 50:
                    results["warnings"].append(
                        f"Field '{field}' has low completeness: {completeness:.1f}%"
                    )
        
        return results
    
    def validate_all(self, df: pd.DataFrame) -> Dict[str, any]:
        """Run all validation checks."""
        logger.info(f"Running comprehensive validation for {self.dataset_type} dataset...")
        
        all_results = {
            "dataset_type": self.dataset_type,
            "overall_valid": True,
            "structure": {},
            "quality": {},
            "completeness": {},
            "summary": {
                "total_errors": 0,
                "total_warnings": 0
            }
        }
        
        # Structure validation
        structure_results = self.validate_structure(df)
        all_results["structure"] = structure_results
        if not structure_results["is_valid"]:
            all_results["overall_valid"] = False
        
        # Quality validation
        quality_results = self.validate_data_quality(df)
        all_results["quality"] = quality_results
        if not quality_results["is_valid"]:
            all_results["overall_valid"] = False
        
        # Completeness validation
        completeness_results = self.validate_completeness(df)
        all_results["completeness"] = completeness_results
        
        # Summary
        all_results["summary"]["total_errors"] = (
            len(structure_results.get("errors", [])) +
            len(quality_results.get("errors", []))
        )
        all_results["summary"]["total_warnings"] = (
            len(structure_results.get("warnings", [])) +
            len(quality_results.get("warnings", [])) +
            len(completeness_results.get("warnings", []))
        )
        
        return all_results

