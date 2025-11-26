"""Data preprocessing utilities."""
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Comprehensive data preprocessor for JD and candidate datasets."""
    
    def __init__(self, dataset_type: str = "jd"):
        """
        Initialize preprocessor.
        
        Args:
            dataset_type: "jd" for job descriptions or "candidate" for candidates
        """
        self.dataset_type = dataset_type.lower()
        self.preprocessing_stats = {}
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text)
        # Replace multiple spaces, tabs, newlines with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text."""
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        return text
    
    def normalize_email(self, email: str) -> Optional[str]:
        """Normalize email address."""
        if pd.isna(email) or email is None:
            return None
        
        email = str(email).strip().lower()
        
        # Basic email validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            logger.warning(f"Invalid email format: {email}")
            return None
        
        return email
    
    def normalize_id(self, id_value: str) -> str:
        """Normalize ID values."""
        if pd.isna(id_value) or id_value is None:
            return ""
        
        id_str = str(id_value).strip()
        # Remove special characters, keep alphanumeric and underscores
        id_str = re.sub(r'[^\w]', '_', id_str)
        return id_str
    
    def extract_and_clean_skills(self, skills_text: str) -> str:
        """Extract and clean skills from text."""
        if pd.isna(skills_text) or skills_text is None:
            return ""
        
        skills = str(skills_text)
        
        # Split by common delimiters
        skill_list = re.split(r'[,;|]', skills)
        
        # Clean each skill
        cleaned_skills = []
        for skill in skill_list:
            skill = skill.strip()
            if skill:
                # Capitalize first letter
                skill = skill.capitalize()
                cleaned_skills.append(skill)
        
        # Join with comma
        return ', '.join(cleaned_skills)
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "fill_empty") -> pd.DataFrame:
        """Handle missing values based on strategy."""
        df = df.copy()
        
        if strategy == "fill_empty":
            # Fill missing values with empty strings for text fields
            text_columns = df.select_dtypes(include=['object']).columns
            df[text_columns] = df[text_columns].fillna("")
        
        elif strategy == "drop":
            # Drop rows with missing required fields
            if self.dataset_type == "jd":
                required = ["job_id", "title", "description"]
            else:
                required = ["candidate_id"]
            
            required_cols = [col for col in required if col in df.columns]
            df = df.dropna(subset=required_cols)
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """Remove duplicate rows."""
        if subset is None:
            if self.dataset_type == "jd":
                subset = ["job_id"]
            else:
                subset = ["candidate_id"]
        
        subset = [col for col in subset if col in df.columns]
        
        if not subset:
            return df
        
        initial_count = len(df)
        df = df.drop_duplicates(subset=subset, keep='first')
        removed_count = initial_count - len(df)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate rows")
            self.preprocessing_stats["duplicates_removed"] = removed_count
        
        return df
    
    def preprocess_text_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Preprocess text columns."""
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                df[col] = df[col].apply(self.clean_text)
        
        return df
    
    def preprocess_jd_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess job description data."""
        logger.info("Preprocessing JD data...")
        df = df.copy()
        
        # Normalize IDs
        if "job_id" in df.columns:
            df["job_id"] = df["job_id"].apply(self.normalize_id)
        
        # Clean text fields
        text_fields = ["title", "description", "requirements", "company", "location"]
        text_fields = [f for f in text_fields if f in df.columns]
        df = self.preprocess_text_columns(df, text_fields)
        
        # Process skills
        if "skills" in df.columns:
            df["skills"] = df["skills"].apply(self.extract_and_clean_skills)
        
        # Handle missing values
        df = self.handle_missing_values(df, strategy="fill_empty")
        
        # Remove duplicates
        df = self.remove_duplicates(df)
        
        logger.info(f"Preprocessed {len(df)} job descriptions")
        return df
    
    def preprocess_candidate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess candidate data."""
        logger.info("Preprocessing candidate data...")
        df = df.copy()
        
        # Normalize IDs
        if "candidate_id" in df.columns:
            df["candidate_id"] = df["candidate_id"].apply(self.normalize_id)
        
        # Normalize email
        if "email" in df.columns:
            df["email"] = df["email"].apply(self.normalize_email)
        
        # Clean text fields
        text_fields = ["name", "summary", "experience", "education", "resume_text"]
        text_fields = [f for f in text_fields if f in df.columns]
        df = self.preprocess_text_columns(df, text_fields)
        
        # Process skills
        if "skills" in df.columns:
            df["skills"] = df["skills"].apply(self.extract_and_clean_skills)
        
        # Handle missing values
        df = self.handle_missing_values(df, strategy="fill_empty")
        
        # Remove duplicates
        df = self.remove_duplicates(df)
        
        logger.info(f"Preprocessed {len(df)} candidates")
        return df
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main preprocessing method."""
        initial_count = len(df)
        
        if self.dataset_type == "jd":
            df = self.preprocess_jd_data(df)
        elif self.dataset_type == "candidate":
            df = self.preprocess_candidate_data(df)
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
        
        final_count = len(df)
        self.preprocessing_stats["initial_rows"] = initial_count
        self.preprocessing_stats["final_rows"] = final_count
        self.preprocessing_stats["rows_removed"] = initial_count - final_count
        
        return df
    
    def get_preprocessing_stats(self) -> Dict:
        """Get preprocessing statistics."""
        return self.preprocessing_stats.copy()

