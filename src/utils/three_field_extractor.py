"""Helper functions to extract 3 fields (title, skills, experience/requirement) from CSV data."""
from typing import Dict, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ThreeFieldExtractor:
    """Extract exactly 3 fields: title, skills, experience/requirement."""
    
    @staticmethod
    def extract_candidate_fields(row: pd.Series) -> Dict[str, str]:
        """
        Extract 3 fields from candidate row: title, skills, experience.
        
        Args:
            row: Pandas Series with candidate data
        
        Returns:
            Dictionary with keys: 'title', 'skills', 'experience'
        """
        def safe_get(field_name: str, alternatives: list = None) -> Optional[str]:
            """Safely get field value, trying alternatives if main field not found."""
            # Try main field first
            if field_name in row.index:
                try:
                    value = row[field_name]
                    if pd.notna(value) and value is not None:
                        text = str(value).strip()
                        if text and text.lower() != 'nan':
                            return text
                except (KeyError, AttributeError, TypeError):
                    pass
            
            # Try alternatives
            if alternatives:
                for alt_field in alternatives:
                    if alt_field in row.index:
                        try:
                            value = row[alt_field]
                            if pd.notna(value) and value is not None:
                                text = str(value).strip()
                                if text and text.lower() != 'nan':
                                    return text
                        except (KeyError, AttributeError, TypeError):
                            continue
            
            return None
        
        # Extract title (desired_job_translated or current job title)
        title = safe_get('desired_job_translated', [
            'desired_job', 'job_title', 'title', 'target', 'target job'
        ]) or ''
        
        # Extract skills
        skills = safe_get('skills', [
            'Skills', 'skill', 'technical_skills', 'competencies', 'summary'
        ]) or ''
        
        # Extract experience
        experience = safe_get('experience', [
            'Experience', 'work_experience', 'work experience', 
            'employment_history', 'work history'
        ]) or ''
        
        return {
            'title': title,
            'skills': skills,
            'experience': experience
        }
    
    @staticmethod
    def extract_job_fields(row: pd.Series) -> Dict[str, str]:
        """
        Extract 3 fields from job row: title, skills, requirement.
        
        Args:
            row: Pandas Series with job data
        
        Returns:
            Dictionary with keys: 'title', 'skills', 'requirement'
        """
        def safe_get(field_name: str, alternatives: list = None) -> Optional[str]:
            """Safely get field value, trying alternatives if main field not found."""
            # Try main field first
            if field_name in row.index:
                try:
                    value = row[field_name]
                    if pd.notna(value) and value is not None:
                        text = str(value).strip()
                        if text and text.lower() != 'nan':
                            return text
                except (KeyError, AttributeError, TypeError):
                    pass
            
            # Try alternatives
            if alternatives:
                for alt_field in alternatives:
                    if alt_field in row.index:
                        try:
                            value = row[alt_field]
                            if pd.notna(value) and value is not None:
                                text = str(value).strip()
                                if text and text.lower() != 'nan':
                                    return text
                        except (KeyError, AttributeError, TypeError):
                            continue
            
            return None
        
        # Extract title (required)
        title = safe_get('title', [
            'Job Title', 'job_title', 'job title', 'position', 'position_title'
        ]) or ''
        
        # Extract skills
        skills = safe_get('skills', [
            'Skills', 'skill', 'required_skills', 'technical_skills', 'competencies'
        ]) or ''
        
        # Extract requirement
        requirement = safe_get('requirements', [
            'requirement', 'Job Requirements', 'job_requirements', 
            'req', 'requirement_text'
        ]) or ''
        
        # If no requirement, use description as fallback
        if not requirement:
            requirement = safe_get('description', [
                'Job Description', 'job_description', 'desc'
            ]) or ''
        
        return {
            'title': title,
            'skills': skills,
            'requirement': requirement
        }




