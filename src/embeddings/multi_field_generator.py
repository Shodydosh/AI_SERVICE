"""Multi-field embedding generator that creates 3 separate embeddings per record."""
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import logging
import numpy as np
from config.settings import settings
from .model_selector import EmbeddingModelSelector

logger = logging.getLogger(__name__)

# Try to import pyvi for Vietnamese tokenization
try:
    from pyvi.ViTokenizer import tokenize as vietnamese_tokenize
    PYVI_AVAILABLE = True
except ImportError:
    PYVI_AVAILABLE = False
    logger.warning("pyvi not installed. Vietnamese tokenization will be skipped.")


class MultiFieldEmbeddingGenerator:
    """Generate 3 separate embeddings for title, skills, and experience/requirements."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize multi-field embedding generator.
        
        Args:
            model_name: Name of the model to use. If None, uses model from settings.
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        
        # Validate model
        model_info = EmbeddingModelSelector().get_model_info(self.model_name)
        if model_info:
            logger.info(f"Using recommended model: {model_info['name']}")
            logger.info(f"  Dimensions: {model_info['dimensions']}, Performance: {model_info['performance']}")
            self.requires_vietnamese_tokenization = model_info.get('requires_tokenization', False)
        else:
            logger.info(f"Using custom model: {self.model_name}")
            self.requires_vietnamese_tokenization = 'SimCSE-VietNamese' in self.model_name or 'SimCSE-Vietnamese' in self.model_name
        
        if self.requires_vietnamese_tokenization and not PYVI_AVAILABLE:
            logger.warning("Vietnamese tokenization required but pyvi not available.")
        
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Multi-field embedding generator initialized (dimension: {self.dimension})")
    
    def _tokenize_vietnamese(self, text: str) -> str:
        """Tokenize Vietnamese text if required."""
        if not self.requires_vietnamese_tokenization:
            return text
        
        if not PYVI_AVAILABLE:
            return text
        
        try:
            return vietnamese_tokenize(text)
        except Exception as e:
            logger.warning(f"Error tokenizing Vietnamese text: {e}. Using original text.")
            return text
    
    def _generate_single_embedding(self, text: str, field_name: str = None) -> List[float]:
        """Generate embedding for a single text field."""
        if not text or not text.strip():
            # Only log at debug level to reduce noise - empty fields are expected for some records
            if field_name:
                logger.debug(f"Empty text for field '{field_name}', using zero embedding")
            else:
                logger.debug("Empty text provided for embedding, using zero embedding")
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
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def generate_job_embeddings(
        self,
        title: str,
        skills: Optional[str] = None,
        requirements: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Generate 3 separate embeddings for job: title, skills, requirements.
        
        Args:
            title: Job title (required)
            skills: Job skills/technologies
            requirements: Job requirements/experience requirements
        
        Returns:
            Dict with keys: 'title_embedding', 'skills_embedding', 'requirement_embedding'
        """
        # Title embedding (required)
        title_emb = self._generate_single_embedding(title or "", field_name="title")
        
        # Skills embedding
        skills_emb = self._generate_single_embedding(skills or "", field_name="skills")
        
        # Requirements embedding
        requirement_emb = self._generate_single_embedding(requirements or "", field_name="requirements")
        
        return {
            'title_embedding': title_emb,
            'skills_embedding': skills_emb,
            'requirement_embedding': requirement_emb
        }
    
    def generate_candidate_embeddings(
        self,
        title: Optional[str] = None,  # desired_job or current job title
        skills: Optional[str] = None,
        experience: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Generate 3 separate embeddings for candidate: title, skills, experience.
        
        Args:
            title: Desired job title or current job title
            skills: Candidate skills
            experience: Work experience
        
        Returns:
            Dict with keys: 'title_embedding', 'skills_embedding', 'experience_embedding'
        """
        # Title embedding (desired job or current job)
        title_emb = self._generate_single_embedding(title or "", field_name="title")
        
        # Skills embedding
        skills_emb = self._generate_single_embedding(skills or "", field_name="skills")
        
        # Experience embedding
        experience_emb = self._generate_single_embedding(experience or "", field_name="experience")
        
        return {
            'title_embedding': title_emb,
            'skills_embedding': skills_emb,
            'experience_embedding': experience_emb
        }
    
    def generate_job_embeddings_batch(
        self,
        jobs: List[Dict[str, str]],
        batch_size: int = 32
    ) -> List[Dict[str, List[float]]]:
        """
        Generate embeddings for multiple jobs in batch.
        
        Args:
            jobs: List of dicts with keys: 'title', 'skills', 'requirements'
            batch_size: Batch size for processing
        
        Returns:
            List of embedding dicts
        """
        results = []
        
        # Process in batches
        for i in range(0, len(jobs), batch_size):
            batch = jobs[i:i + batch_size]
            
            for job in batch:
                embeddings = self.generate_job_embeddings(
                    title=job.get('title', ''),
                    skills=job.get('skills'),
                    requirements=job.get('requirements')
                )
                results.append(embeddings)
        
        return results
    
    def generate_candidate_embeddings_batch(
        self,
        candidates: List[Dict[str, str]],
        batch_size: int = 32
    ) -> List[Dict[str, List[float]]]:
        """
        Generate embeddings for multiple candidates in batch.
        
        Args:
            candidates: List of dicts with keys: 'title', 'skills', 'experience'
            batch_size: Batch size for processing
        
        Returns:
            List of embedding dicts
        """
        results = []
        
        # Process in batches
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            
            for candidate in batch:
                embeddings = self.generate_candidate_embeddings(
                    title=candidate.get('title'),
                    skills=candidate.get('skills'),
                    experience=candidate.get('experience')
                )
                results.append(embeddings)
        
        return results
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.dimension
