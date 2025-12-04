"""Contextual Embeddings Service: Tạo composite embeddings với prompt engineering."""
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ContextualEmbeddingsService:
    """
    Contextual Embeddings Service tạo composite embeddings với context.
    
    Ví dụ: "[Title] working on [Skills] with [Requirements]"
    """
    
    def __init__(self, embedding_generator):
        """
        Initialize contextual embeddings service.
        
        Args:
            embedding_generator: Embedding generator instance
        """
        self.embedding_generator = embedding_generator
        logger.info("ContextualEmbeddingsService initialized")
    
    def create_jd_contextual_text(
        self,
        title: Optional[str] = None,
        skills: Optional[str] = None,
        requirements: Optional[str] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Tạo contextual text cho JD với prompt engineering.
        
        Args:
            title: Job title
            skills: Required skills
            requirements: Job requirements
            description: Job description
            
        Returns:
            Contextual text string
        """
        parts = []
        
        if title:
            # Emphasize title (repeat for importance)
            parts.append(f"Job Position: {title}")
            parts.append(f"Role: {title}")
        
        if skills:
            # Truncate skills if too long
            skills_text = skills[:500] if len(skills) > 500 else skills
            parts.append(f"Required Skills: {skills_text}")
        
        if requirements:
            # Truncate requirements if too long
            reqs_text = requirements[:400] if len(requirements) > 400 else requirements
            parts.append(f"Requirements: {reqs_text}")
        
        if description:
            # Truncate description if too long
            desc_text = description[:300] if len(description) > 300 else description
            parts.append(f"Description: {desc_text}")
        
        # Join with separator
        contextual_text = " | ".join(parts)
        
        return contextual_text
    
    def create_candidate_contextual_text(
        self,
        desired_job: Optional[str] = None,
        skills: Optional[str] = None,
        experience: Optional[str] = None,
        summary: Optional[str] = None
    ) -> str:
        """
        Tạo contextual text cho Candidate với prompt engineering.
        
        Args:
            desired_job: Desired job title
            skills: Candidate skills
            experience: Work experience
            summary: Resume summary
            
        Returns:
            Contextual text string
        """
        parts = []
        
        if desired_job:
            # Emphasize desired job (repeat for importance)
            parts.append(f"Desired Position: {desired_job}")
            parts.append(f"Looking for: {desired_job}")
        
        if skills:
            # Truncate skills if too long
            skills_text = skills[:500] if len(skills) > 500 else skills
            parts.append(f"My Skills: {skills_text}")
        
        if experience:
            # Truncate experience if too long
            exp_text = experience[:400] if len(experience) > 400 else experience
            parts.append(f"Experience: {exp_text}")
        
        if summary:
            # Truncate summary if too long
            summary_text = summary[:300] if len(summary) > 300 else summary
            parts.append(f"Summary: {summary_text}")
        
        # Join with separator
        contextual_text = " | ".join(parts)
        
        return contextual_text
    
    def generate_contextual_embedding(
        self,
        contextual_text: str,
        field_type: str = "jd"
    ) -> List[float]:
        """
        Generate embedding từ contextual text.
        
        Args:
            contextual_text: Contextual text
            field_type: "jd" or "candidate"
            
        Returns:
            Embedding vector
        """
        if not contextual_text or not contextual_text.strip():
            # Return zero embedding if empty
            dimension = self.embedding_generator.get_embedding_dimension()
            return [0.0] * dimension
        
        # Use embedding generator
        embedding = self.embedding_generator.generate_embedding(contextual_text)
        
        return embedding

