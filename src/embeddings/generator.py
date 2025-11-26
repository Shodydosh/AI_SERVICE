"""Embedding generator using sentence transformers."""
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
import logging
from config.settings import settings
from .model_selector import EmbeddingModelSelector

logger = logging.getLogger(__name__)

# Try to import pyvi for Vietnamese tokenization
try:
    from pyvi.ViTokenizer import tokenize as vietnamese_tokenize
    PYVI_AVAILABLE = True
except ImportError:
    PYVI_AVAILABLE = False
    logger.warning("pyvi not installed. Vietnamese tokenization will be skipped. Install with: pip install pyvi")


class EmbeddingGenerator:
    """Generator for text embeddings."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding generator with a model.
        
        Args:
            model_name: Name of the model to use. If None, uses model from settings.
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        
        # Validate model
        model_info = EmbeddingModelSelector().get_model_info(self.model_name)
        if model_info:
            logger.info(f"Using recommended model: {model_info['name']}")
            logger.info(f"  Dimensions: {model_info['dimensions']}, Performance: {model_info['performance']}")
            # Check if model requires Vietnamese tokenization
            self.requires_vietnamese_tokenization = model_info.get('requires_tokenization', False)
        else:
            logger.info(f"Using custom model: {self.model_name}")
            # Check if it's a SimCSE Vietnamese model
            self.requires_vietnamese_tokenization = 'SimCSE-VietNamese' in self.model_name or 'SimCSE-Vietnamese' in self.model_name
        
        if self.requires_vietnamese_tokenization and not PYVI_AVAILABLE:
            logger.warning("Vietnamese tokenization required but pyvi not available. Install with: pip install pyvi")
            logger.warning("Text will be used without tokenization (may reduce quality)")
        
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info("Embedding model loaded successfully")
    
    def _tokenize_vietnamese(self, text: str) -> str:
        """
        Tokenize Vietnamese text if required by the model.
        
        Args:
            text: Input text
            
        Returns:
            Tokenized text
        """
        if not self.requires_vietnamese_tokenization:
            return text
        
        if not PYVI_AVAILABLE:
            return text
        
        try:
            return vietnamese_tokenize(text)
        except Exception as e:
            logger.warning(f"Error tokenizing Vietnamese text: {e}. Using original text.")
            return text
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text with optimized settings for precision."""
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * self.model.get_sentence_embedding_dimension()
        
        try:
            # Tokenize Vietnamese text if required
            processed_text = self._tokenize_vietnamese(text)
            
            embedding = self.model.encode(
                processed_text,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Normalize for better cosine similarity
                show_progress_bar=False
            )
            # Ensure normalization (double-check)
            import numpy as np
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch."""
        if not texts:
            return []
        
        # Filter out empty texts and tokenize if required
        valid_texts = []
        for text in texts:
            if text and text.strip():
                processed_text = self._tokenize_vietnamese(text)
                valid_texts.append(processed_text)
            else:
                valid_texts.append("")
        
        try:
            from tqdm import tqdm
            # Calculate number of batches
            num_batches = (len(valid_texts) + batch_size - 1) // batch_size
            
            embeddings = []
            with tqdm(total=len(valid_texts), desc="Generating embeddings", unit="text") as pbar:
                for i in range(0, len(valid_texts), batch_size):
                    batch = valid_texts[i:i + batch_size]
                    batch_embeddings = self.model.encode(
                        batch,
                        batch_size=len(batch),
                        convert_to_numpy=True,
                        normalize_embeddings=True,  # Normalize for better cosine similarity
                        show_progress_bar=False
                    )
                    # Ensure normalization (double-check)
                    import numpy as np
                    norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                    norms[norms == 0] = 1  # Avoid division by zero
                    batch_embeddings = batch_embeddings / norms
                    embeddings.extend(batch_embeddings.tolist())
                    pbar.update(len(batch))
            
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.model.get_sentence_embedding_dimension()

