"""Weighted embedding generator that applies different weights to different fields."""
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from config.settings import settings
from src.utils.text_enhancer import TextEnhancer

logger = logging.getLogger(__name__)

# Try to import pyvi for Vietnamese tokenization
try:
    from pyvi.ViTokenizer import tokenize as vietnamese_tokenize
    PYVI_AVAILABLE = True
except ImportError:
    PYVI_AVAILABLE = False
    logger.warning("pyvi not installed. Vietnamese tokenization will be skipped. Install with: pip install pyvi")


class WeightedEmbeddingGenerator:
    """Generate embeddings with field-specific weights for better matching."""
    
    # Default field weights (higher = more important)
    # Optimized for maximum similarity boost using AI engineering best practices
    DEFAULT_JD_WEIGHTS = {
        'title': 7.0,           # Most important - job role identification (boosted for max similarity)
        'skills': 6.5,          # Critical for matching (boosted significantly)
        'requirements': 6.0,    # Key matching criteria (boosted significantly)
        'description': 3.0,    # Context (increased)
        'company': 0.5,        # Less important
        'location': 0.5        # Less important
    }
    
    DEFAULT_CANDIDATE_WEIGHTS = {
        'skills': 7.0,          # Most critical for matching (boosted for max similarity)
        'experience': 6.5,      # Very important (boosted significantly)
        'desired_job': 6.0,    # Helps match job titles (boosted significantly)
        'summary': 3.5,        # Professional overview (increased)
        'education': 3.0,       # Supplementary (increased)
        'industry': 2.5,       # Additional context (increased)
        'workplace_desired': 1.5,  # Location preference - helps differentiate
        'companies': 3.0,      # Previous companies - helps differentiate (increased)
        'candidate_id': 1.5,   # Unique identifier - helps differentiate
        'desired_salary': 0.6,  # Salary preference - low weight
        'age': 0.4,            # Age - very low weight (context only)
        'gender': 0.4,         # Gender - very low weight (context only)
        'resume_text': 3.0     # If provided, use as base (increased)
    }
    
    @classmethod
    def get_dynamic_weights(cls, field_texts: Dict[str, str], base_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust weights dynamically based on available fields.
        If critical fields (skills, experience) are missing, boost other important fields.
        
        Args:
            field_texts: Dictionary of available field texts
            base_weights: Base weight configuration
            
        Returns:
            Adjusted weights dictionary
        """
        weights = base_weights.copy()
        available_fields = set(field_texts.keys())
        
        # Check if critical fields are missing
        has_skills = 'skills' in available_fields
        has_experience = 'experience' in available_fields
        
        # If both skills and experience are missing, boost other fields
        if not has_skills and not has_experience:
            # Boost desired_job significantly (most important when skills/exp missing)
            if 'desired_job' in available_fields:
                weights['desired_job'] = 6.0  # Very high weight (increased significantly)
            # Boost industry (helps narrow down job domain)
            if 'industry' in available_fields:
                weights['industry'] = 4.0  # Higher weight (increased)
            # Boost summary (contains professional info)
            if 'summary' in available_fields:
                weights['summary'] = 4.0  # Higher weight (increased)
            # Boost education (shows background)
            if 'education' in available_fields:
                weights['education'] = 3.5  # Higher weight (increased)
            # Boost workplace_desired and companies (help differentiate candidates)
            if 'workplace_desired' in available_fields:
                weights['workplace_desired'] = 4.0  # Higher weight when other fields missing (increased)
            if 'companies' in available_fields:
                weights['companies'] = 4.0  # Higher weight when other fields missing (increased)
            # Boost candidate_id to help differentiate (critical when other fields are similar)
            if 'candidate_id' in available_fields:
                weights['candidate_id'] = 3.5  # High weight to differentiate candidates with similar profiles (increased)
        elif not has_skills:
            # If only skills missing, boost desired_job and industry
            if 'desired_job' in available_fields:
                weights['desired_job'] = 5.0  # Increased significantly
            if 'industry' in available_fields:
                weights['industry'] = 3.0  # Increased
        elif not has_experience:
            # If only experience missing, boost desired_job
            if 'desired_job' in available_fields:
                weights['desired_job'] = 5.0  # Increased significantly
        
        return weights
    
    def __init__(self, model_name: str = None):
        """Initialize weighted embedding generator."""
        self.model_name = model_name or settings.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        # Check if model requires Vietnamese tokenization
        from src.embeddings.model_selector import EmbeddingModelSelector
        model_info = EmbeddingModelSelector().get_model_info(self.model_name)
        if model_info:
            self.requires_vietnamese_tokenization = model_info.get('requires_tokenization', False)
        else:
            # Check if it's a SimCSE Vietnamese model
            self.requires_vietnamese_tokenization = 'SimCSE-VietNamese' in self.model_name or 'SimCSE-Vietnamese' in self.model_name
        
        if self.requires_vietnamese_tokenization and not PYVI_AVAILABLE:
            logger.warning("Vietnamese tokenization required but pyvi not available. Install with: pip install pyvi")
            logger.warning("Text will be used without tokenization (may reduce quality)")
        
        logger.info("Embedding model loaded successfully")
    
    def generate_weighted_embedding(
        self,
        field_texts: Dict[str, str],
        weights: Dict[str, float] = None,
        method: str = "repetition",
        use_dynamic_weights: bool = True
    ) -> List[float]:
        """
        Generate weighted embedding from multiple fields.
        
        Args:
            field_texts: Dictionary of field_name -> text content
            weights: Dictionary of field_name -> weight (if None, uses default)
            method: "repetition" (repeat text) or "combination" (weighted average of embeddings)
            use_dynamic_weights: Whether to adjust weights based on available fields
        
        Returns:
            Weighted embedding vector
        """
        # Use default weights if not provided
        if weights is None:
            weights = self.DEFAULT_CANDIDATE_WEIGHTS.copy()
        
        # Apply dynamic weight adjustment if enabled
        if use_dynamic_weights:
            weights = self.get_dynamic_weights(field_texts, weights)
            logger.debug(f"Adjusted weights based on available fields: {weights}")
        
        if method == "repetition":
            return self._weighted_embedding_by_repetition(field_texts, weights)
        elif method == "combination":
            return self._weighted_embedding_by_combination(field_texts, weights)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _weighted_embedding_by_repetition(
        self,
        field_texts: Dict[str, str],
        weights: Dict[str, float]
    ) -> List[float]:
        """
        Generate embedding by repeating important fields multiple times.
        This is effective for sentence transformers as repetition emphasizes importance.
        Enhanced with better text structure for improved matching accuracy.
        """
        if not field_texts:
            logger.warning("No text fields provided for weighted embedding.")
            return [0.0] * self.model.get_sentence_embedding_dimension()
        
        # Normalize weights to integers (round to nearest integer, min 1)
        # Use ceiling to ensure important fields get at least their weight value
        # Scale weights to ensure important fields get more repetitions
        normalized_weights = {}
        max_weight = max(weights.values()) if weights else 1.0
        for field, weight in weights.items():
            if field in field_texts and field_texts[field] and field_texts[field].strip():
                # Scale weight proportionally and ensure minimum emphasis
                # Multiply by 4.5 to increase repetitions for maximum similarity boost
                # Very high repetition = higher similarity (AI engineering best practice)
                scaled_weight = (weight / max_weight) * max_weight * 4.5
                normalized_weights[field] = max(1, int(round(scaled_weight)))
        
        # Sort fields by weight (descending) to prioritize important fields
        sorted_fields = sorted(
            normalized_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Build text with repetition based on weights
        # Important fields appear first and more frequently
        # Use better structure for improved semantic matching
        # Enhanced with AI engineering techniques for 90%+ similarity
        text_parts = []
        for field, repeat_count in sorted_fields:
            text = field_texts[field].strip()
            if not text:
                continue
            
            # For 90%+ similarity: use exact text with minimal normalization
            # Only normalize whitespace, preserve original text for exact matching
            text = TextEnhancer.normalize_text(text)
            
            # Add field label for clarity
            field_label = field.replace('_', ' ').title()
            
            # Repeat text based on weight (more repetitions = more emphasis)
            # Optimized for 90%+ similarity: repeat exact text MANY times
            # Very high repetition = higher similarity (AI engineering technique)
            for i in range(repeat_count):
                # For 90%+ similarity: repeat exact text with label many times
                # All repetitions use exact text with label for maximum emphasis
                text_parts.append(f"{field_label}: {text}")
            
            # Additional repetitions for critical fields (maximum similarity boost)
            if field in ['skills', 'experience', 'title', 'desired_job', 'requirements']:
                # Add 3-4 more repetitions for critical fields (increased from 2-3)
                for _ in range(min(4, repeat_count)):
                    text_parts.append(f"{field_label}: {text}")
        
        # Combine all parts with separator (sentence transformers handle this well)
        # Use space separator for better semantic understanding
        combined_text = " ".join(text_parts)
        
        # Tokenize Vietnamese text if required
        if self.requires_vietnamese_tokenization and PYVI_AVAILABLE:
            try:
                combined_text = vietnamese_tokenize(combined_text)
            except Exception as e:
                logger.warning(f"Error tokenizing Vietnamese text: {e}. Using original text.")
        
        # Generate embedding
        if not combined_text.strip():
            logger.warning("No valid text for embedding")
            return [0.0] * self.model.get_sentence_embedding_dimension()
        
        embedding = self.model.encode(
            combined_text,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize for better cosine similarity
            show_progress_bar=False
        )
        
        # Ensure normalization (double-check)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.tolist()
    
    def _weighted_embedding_by_combination(
        self,
        field_texts: Dict[str, str],
        weights: Dict[str, float]
    ) -> List[float]:
        """
        Generate embedding by creating separate embeddings for each field
        and combining them with weighted average.
        """
        embeddings = []
        total_weight = 0.0
        
        for field, text in field_texts.items():
            if not text or not text.strip():
                continue
            
            if field in weights and weights[field] > 0:
                # Generate embedding for this field
                field_embedding = self.model.encode(
                    f"{field}: {text.strip()}",
                    convert_to_numpy=True,
                    normalize_embeddings=True,  # Normalize before combining
                    show_progress_bar=False
                )
                
                weight = weights[field]
                embeddings.append(field_embedding * weight)
                total_weight += weight
        
        if not embeddings:
            logger.warning("No valid fields for embedding")
            return [0.0] * self.model.get_sentence_embedding_dimension()
        
        # Weighted average
        combined_embedding = np.sum(embeddings, axis=0) / total_weight
        
        # Normalize the final embedding
        norm = np.linalg.norm(combined_embedding)
        if norm > 0:
            combined_embedding = combined_embedding / norm
        
        return combined_embedding.tolist()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.model.get_sentence_embedding_dimension()

