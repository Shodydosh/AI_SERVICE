"""Column name mapping utilities for different dataset formats."""
from typing import Dict, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ColumnMapper:
    """Maps different column name formats to standard format."""
    
    # Common column name variations
    JD_COLUMN_MAPPINGS = {
        # job_id variations
        'job_id': ['jobid', 'job_id', 'job id', 'id', 'jobid', 'job-id'],
        # title variations
        'title': ['job title', 'jobtitle', 'title', 'job_title', 'position', 'position title'],
        # description variations
        'description': ['job description', 'jobdescription', 'description', 'job_description', 'desc', 'job desc'],
        # company variations
        'company': ['name company', 'company', 'company name', 'employer', 'company_name'],
        # requirements variations
        'requirements': ['job requirements', 'requirements', 'job_requirements', 'req', 'requirement'],
        # location variations
        'location': ['job address', 'address', 'location', 'job_location', 'city', 'job address'],
        # skills variations
        'skills': ['skill', 'skills', 'required skills', 'technical skills', 'competencies']
    }
    
    CANDIDATE_COLUMN_MAPPINGS = {
        # candidate_id variations
        'candidate_id': ['candidateid', 'candidate_id', 'candidate id', 'id', 'candidate-id', 'user_id', 'cv_id', 'userid'],
        # name variations
        'name': ['name', 'full name', 'fullname', 'candidate name', 'applicant name', 'user_name'],
        # email variations
        'email': ['email', 'email address', 'e-mail', 'email_address'],
        # skills variations
        'skills': ['skill', 'skills', 'technical skills', 'competencies', 'skill set'],
        # experience variations
        'experience': ['experience', 'work experience', 'work_experience', 'employment history', 'work history', 'work_experience'],
        # education variations
        'education': ['education', 'educational background', 'qualification', 'qualifications', 'degree'],
        # summary variations
        'summary': ['summary', 'professional summary', 'profile', 'about', 'bio', 'biography'],
        # resume_text variations
        'resume_text': ['resume', 'resume text', 'cv', 'curriculum vitae', 'resume_text', 'cv text']
    }
    
    @classmethod
    def normalize_column_name(cls, col_name: str) -> str:
        """Normalize column name to lowercase with underscores."""
        if pd.isna(col_name):
            return ""
        return str(col_name).lower().strip().replace(' ', '_').replace('-', '_')
    
    @classmethod
    def map_jd_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Map JD dataset columns to standard names."""
        df = df.copy()
        column_mapping = {}
        
        # Create a mapping of normalized names to original names
        normalized_to_original = {cls.normalize_column_name(col): col for col in df.columns}
        
        # Find matches for each standard column
        for standard_col, variations in cls.JD_COLUMN_MAPPINGS.items():
            found = False
            for variation in variations:
                normalized_variation = cls.normalize_column_name(variation)
                # Check exact match first
                if normalized_variation in normalized_to_original:
                    original_col = normalized_to_original[normalized_variation]
                    if original_col not in column_mapping.values():
                        column_mapping[original_col] = standard_col
                        logger.info(f"Mapped '{original_col}' -> '{standard_col}'")
                        found = True
                        break
                # Check partial match (contains)
                else:
                    for norm_col, orig_col in normalized_to_original.items():
                        if normalized_variation in norm_col or norm_col in normalized_variation:
                            if orig_col not in column_mapping.values():
                                column_mapping[orig_col] = standard_col
                                logger.info(f"Mapped '{orig_col}' -> '{standard_col}' (partial match)")
                                found = True
                                break
                    if found:
                        break
        
        # Rename columns
        if column_mapping:
            df = df.rename(columns=column_mapping)
            logger.info(f"Mapped {len(column_mapping)} columns to standard format")
        
        return df
    
    @classmethod
    def map_candidate_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Map candidate dataset columns to standard names."""
        df = df.copy()
        column_mapping = {}
        
        # Normalize all column names first
        normalized_cols = {cls.normalize_column_name(col): col for col in df.columns}
        
        # Find matches for each standard column
        for standard_col, variations in cls.CANDIDATE_COLUMN_MAPPINGS.items():
            for variation in variations:
                normalized_variation = cls.normalize_column_name(variation)
                if normalized_variation in normalized_cols:
                    original_col = normalized_cols[normalized_variation]
                    if original_col not in column_mapping.values():
                        column_mapping[original_col] = standard_col
                        logger.info(f"Mapped '{original_col}' -> '{standard_col}'")
                        break
        
        # Rename columns
        if column_mapping:
            df = df.rename(columns=column_mapping)
            logger.info(f"Mapped {len(column_mapping)} columns")
        
        return df
    
    @classmethod
    def auto_map_columns(cls, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Automatically map columns based on dataset type."""
        if dataset_type.lower() == "jd":
            return cls.map_jd_columns(df)
        elif dataset_type.lower() == "candidate":
            return cls.map_candidate_columns(df)
        else:
            logger.warning(f"Unknown dataset type: {dataset_type}, skipping column mapping")
            return df

