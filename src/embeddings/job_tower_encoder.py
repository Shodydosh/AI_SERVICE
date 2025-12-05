"""Job Tower Encoder - encodes jobs into 3 separate embeddings."""
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from config.settings import settings

# Try to import pyvi for Vietnamese tokenization
try:
    from pyvi.ViTokenizer import tokenize as vietnamese_tokenize
    PYVI_AVAILABLE = True
except ImportError:
    PYVI_AVAILABLE = False

logger = logging.getLogger(__name__)


def preprocess_job_title(title: str) -> str:
    """Preprocess job title."""
    if not title:
        return ""
    text = title.strip().lower()
    text = " ".join(text.split())  # Remove extra spaces
    if len(text) > 200:
        text = text[:200]
    return text


def preprocess_job_skills(skills: str) -> str:
    """Preprocess job skills."""
    if not skills:
        return ""
    text = skills.strip().lower()
    if len(text) > 1000:
        text = text[:1000]
    return text


def preprocess_job_requirements(requirements: str) -> str:
    """Preprocess job requirements."""
    if not requirements:
        return ""
    text = requirements.strip().lower()
    sentences = text.split('.')
    if len(sentences) > 3 and len(text) > 2000:
        text = '. '.join(sentences[:3]) + '.'
    if len(text) > 2000:
        text = text[:2000]
    return text


class JobTowerEncoder:
    """Job Tower Encoder - encodes jobs into 3 separate embeddings."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize Job Tower encoder.
        
        Args:
            model_name: Name of the model to use. If None, uses model from settings.
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        logger.info(f"Loading Job Tower encoder model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.requires_vietnamese_tokenization = 'SimCSE-VietNamese' in self.model_name or 'SimCSE-Vietnamese' in self.model_name
        logger.info(f"Job Tower encoder initialized (dimension: {self.dimension})")
    
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
    
    def encode_job(
        self,
        title: str,
        skills: Optional[str] = None,
        requirements: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Encode job into 3 embeddings.
        
        Args:
            title: Job title (required)
            skills: Job skills/technologies
            requirements: Job requirements/experience requirements
        
        Returns:
            Dict with keys: 'title_embedding', 'skills_embedding', 'requirement_embedding'
        """
        # Preprocess
        title_text = preprocess_job_title(title or "")
        skills_text = preprocess_job_skills(skills or "")
        req_text = preprocess_job_requirements(requirements or "")
        
        # Encode each field
        title_emb = self._generate_single_embedding(title_text, field_name="title")
        skills_emb = self._generate_single_embedding(skills_text, field_name="skills")
        req_emb = self._generate_single_embedding(req_text, field_name="requirements")
        
        return {
            'title_embedding': title_emb,
            'skills_embedding': skills_emb,
            'requirement_embedding': req_emb
        }
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.dimension


