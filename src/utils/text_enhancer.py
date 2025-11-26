"""Advanced text enhancement for better embedding similarity."""
import re
import logging
from typing import List, Dict, Optional
import unicodedata

logger = logging.getLogger(__name__)


class TextEnhancer:
    """Enhance text for better semantic matching using AI engineering techniques."""
    
    # Domain-specific term expansions (Vietnamese + English)
    SKILL_SYNONYMS = {
        'python': ['python', 'py', 'python3', 'python programming'],
        'java': ['java', 'java programming', 'java development'],
        'javascript': ['javascript', 'js', 'node.js', 'nodejs', 'typescript', 'ts'],
        'react': ['react', 'reactjs', 'react.js', 'react native'],
        'machine learning': ['machine learning', 'ml', 'ai', 'artificial intelligence', 'deep learning', 'neural network'],
        'ai': ['ai', 'artificial intelligence', 'machine learning', 'ml', 'deep learning'],
        'backend': ['backend', 'back-end', 'server-side', 'api development'],
        'frontend': ['frontend', 'front-end', 'client-side', 'ui development'],
        'fullstack': ['fullstack', 'full-stack', 'full stack', 'full stack developer'],
        'database': ['database', 'db', 'sql', 'nosql', 'mongodb', 'mysql', 'postgresql'],
        'spring': ['spring', 'spring boot', 'spring framework', 'spring mvc'],
        'tensorflow': ['tensorflow', 'tf', 'tensor flow'],
        'pytorch': ['pytorch', 'torch', 'py torch'],
    }
    
    # Job title variations
    JOB_TITLE_VARIATIONS = {
        'engineer': ['engineer', 'developer', 'programmer', 'coder', 'specialist'],
        'developer': ['developer', 'engineer', 'programmer', 'coder'],
        'lập trình viên': ['lập trình viên', 'developer', 'engineer', 'programmer'],
        'nhân viên': ['nhân viên', 'staff', 'employee', 'worker'],
    }
    
    @classmethod
    def normalize_text(cls, text: str) -> str:
        """
        Normalize text for better matching.
        - Remove extra whitespace
        - Normalize unicode
        - Lowercase
        - Remove special characters (keep Vietnamese)
        """
        if not text:
            return ""
        
        # Normalize unicode (NFD -> NFC)
        text = unicodedata.normalize('NFC', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Keep Vietnamese characters, alphanumeric, and common punctuation
        text = re.sub(r'[^\w\s\u00C0-\u1EF9.,;:!?()\-]', ' ', text)
        
        # Remove extra whitespace again
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    @classmethod
    def expand_skills(cls, skills_text: str) -> str:
        """
        Expand skills with synonyms and related terms for better matching.
        Uses domain knowledge to add relevant synonyms.
        Optimized for 90%+ similarity - keep original text prominent.
        """
        if not skills_text:
            return ""
        
        # Keep original text as primary (for exact matching)
        # Only add minimal, highly relevant synonyms
        skills_lower = skills_text.lower()
        relevant_synonyms = []
        
        # Find and add only the most relevant synonyms (max 2)
        for skill, synonyms in cls.SKILL_SYNONYMS.items():
            if skill in skills_lower:
                # Add first synonym only (most relevant)
                if synonyms and synonyms[0].lower() != skills_text.lower():
                    relevant_synonyms.append(synonyms[0])
                break  # Only match first skill to avoid over-expansion
        
        # Return original with minimal expansion (for 90%+ similarity, exact match is key)
        if relevant_synonyms:
            return f"{skills_text} {relevant_synonyms[0]}"
        return skills_text
    
    @classmethod
    def enhance_job_title(cls, title: str) -> str:
        """
        Enhance job title with variations for better matching.
        """
        if not title:
            return ""
        
        title_lower = title.lower()
        enhanced = title
        
        # Add variations
        for key, variations in cls.JOB_TITLE_VARIATIONS.items():
            if key in title_lower:
                # Add first variation if not already present
                for var in variations:
                    if var not in title_lower and var != key:
                        enhanced = f"{title} {var}"
                        break
                break
        
        return enhanced
    
    @classmethod
    def create_semantic_phrases(cls, text: str, field_type: str = "general") -> str:
        """
        Create semantic phrases that emphasize key concepts.
        Optimized for 90%+ similarity - keep original text prominent, add minimal variations.
        """
        if not text:
            return ""
        
        # For 90%+ similarity, prioritize exact matching
        # Only add 1-2 minimal variations to help matching without diluting
        if field_type == "skills":
            # Keep original prominent, add one variation
            return f"{text} Proficient in {text}"
        elif field_type == "experience":
            # Keep original prominent
            return text  # Don't add variations for experience to maintain exact match
        elif field_type == "requirements":
            # Keep original prominent
            return text  # Don't add variations for requirements to maintain exact match
        
        return text  # Default: return original for exact matching
    
    @classmethod
    def enhance_candidate_text(
        cls,
        skills: Optional[str] = None,
        experience: Optional[str] = None,
        desired_job: Optional[str] = None,
        summary: Optional[str] = None
    ) -> str:
        """
        Enhance candidate text with multiple techniques for better matching.
        """
        parts = []
        
        # Enhance skills with expansion and semantic phrases
        if skills:
            expanded_skills = cls.expand_skills(skills)
            enhanced_skills = cls.create_semantic_phrases(expanded_skills, "skills")
            parts.append(f"Required Skills and Technologies: {enhanced_skills}")
            # Repeat for emphasis (weighted embedding will handle this, but we add here too)
            parts.append(f"Skills: {expanded_skills}")
        
        # Enhance experience
        if experience:
            enhanced_exp = cls.create_semantic_phrases(experience, "experience")
            parts.append(f"Work Experience: {enhanced_exp}")
            parts.append(f"Job Requirements: {enhanced_exp}")  # Match JD format
        
        # Enhance desired job
        if desired_job:
            enhanced_title = cls.enhance_job_title(desired_job)
            parts.append(f"Job Title: {enhanced_title}")
            parts.append(f"Desired Job Title: {enhanced_title}")
        
        # Add summary
        if summary:
            parts.append(f"Professional Summary: {summary}")
        
        return " ".join(parts)
    
    @classmethod
    def enhance_jd_text(
        cls,
        title: Optional[str] = None,
        skills: Optional[str] = None,
        requirements: Optional[str] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Enhance JD text with multiple techniques for better matching.
        """
        parts = []
        
        # Enhance title
        if title:
            enhanced_title = cls.enhance_job_title(title)
            parts.append(f"Job Title: {enhanced_title}")
            # Repeat for emphasis
            parts.append(f"Position: {enhanced_title}")
        
        # Enhance skills with expansion
        if skills:
            expanded_skills = cls.expand_skills(skills)
            enhanced_skills = cls.create_semantic_phrases(expanded_skills, "skills")
            parts.append(f"Required Skills and Technologies: {enhanced_skills}")
            # Repeat for emphasis
            parts.append(f"Skills Required: {expanded_skills}")
        
        # Enhance requirements
        if requirements:
            enhanced_req = cls.create_semantic_phrases(requirements, "requirements")
            parts.append(f"Job Requirements: {enhanced_req}")
        
        # Add description (limited)
        if description:
            # Take first 300 chars for focus
            desc_short = description[:300] + "..." if len(description) > 300 else description
            parts.append(f"Job Description: {desc_short}")
        
        return " ".join(parts)

