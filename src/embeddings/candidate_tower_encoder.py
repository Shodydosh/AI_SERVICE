"""Candidate Tower Encoder - encodes candidates into 3 separate embeddings."""
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from config.settings import settings
from .job_tower_encoder import preprocess_job_title, preprocess_job_skills

# Try to import pyvi for Vietnamese tokenization
try:
    from pyvi.ViTokenizer import tokenize as vietnamese_tokenize
    PYVI_AVAILABLE = True
except ImportError:
    PYVI_AVAILABLE = False

logger = logging.getLogger(__name__)


def preprocess_candidate_experience(experience: str) -> str:
    """Preprocess candidate experience."""
    if not experience:
        return ""
    text = experience.strip().lower()
    if len(text) > 2000:
        text = text[:2000]
    return text


class CandidateTowerEncoder:
    """Candidate Tower Encoder - encodes candidates into 3 separate embeddings."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize Candidate Tower encoder.
        
        Args:
            model_name: Name of the model to use. If None, uses model from settings.
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        logger.info(f"Loading Candidate Tower encoder model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.requires_vietnamese_tokenization = 'SimCSE-VietNamese' in self.model_name or 'SimCSE-Vietnamese' in self.model_name
        logger.info(f"Candidate Tower encoder initialized (dimension: {self.dimension})")
    
    def _tokenize_vietnamese(self, text: str) -> str:
        """Tokenize Vietnamese text if required."""
        if not self.requires_vietnamese_tokenization or not PYVI_AVAILABLE:
            return text
        try:
            return vietnamese_tokenize(text)
        except Exception as e:
            logger.warning(f"Error tokenizing Vietnamese text: {e}. Using original text.")
            return text
    
    def _generate_single_embedding(self, text: str, field_name: str = None) -> List[float]:
        """Generate embedding for a single text field."""
        if not text or not text.strip():
            if field_name:
                logger.debug(f"Empty text for field '{field_name}', using zero embedding")
            return [0.0] * self.dimension
        
        try:
            processed_text = self._tokenize_vietnamese(text.strip())
            
            embedding = self.model.encode(
                processed_text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            # Ensure normalization
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            else:
                embedding = np.zeros(self.dimension)
            
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding for {field_name}: {e}")
            raise
    
    def encode_candidate(
        self,
        title: Optional[str] = None,
        skills: Optional[str] = None,
        experience: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Encode candidate into 3 embeddings.
        
        Args:
            title: Desired job title or current job title
            skills: Candidate skills
            experience: Work experience
        
        Returns:
            Dict with keys: 'title_embedding', 'skills_embedding', 'experience_embedding'
        """
        # Preprocess
        title_text = preprocess_job_title(title or "")
        skills_text = preprocess_job_skills(skills or "")
        exp_text = preprocess_candidate_experience(experience or "")
        
        # Encode each field
        title_emb = self._generate_single_embedding(title_text, field_name="title")
        skills_emb = self._generate_single_embedding(skills_text, field_name="skills")
        exp_emb = self._generate_single_embedding(exp_text, field_name="experience")
        
        return {
            'title_embedding': title_emb,
            'skills_embedding': skills_emb,
            'experience_embedding': exp_emb
        }
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.dimension


