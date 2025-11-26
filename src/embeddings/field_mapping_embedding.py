"""Field-by-field embedding generator that maps specific candidate fields to JD fields."""
from typing import List, Dict, Optional, Tuple
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
    logger.warning("pyvi not installed. Vietnamese tokenization will be skipped.")


class FieldMappingEmbeddingGenerator:
    """Generate embeddings with field-to-field mapping for better semantic matching."""
    
    # Field mappings: candidate_field -> jd_field
    # These mappings define which candidate fields should be compared with which JD fields
    FIELD_MAPPINGS = {
        'skills': 'requirements',      # Candidate skills -> JD requirements
        'experience': 'requirements',  # Candidate experience -> JD requirements  
        'desired_job': 'title'         # Candidate desired job -> JD title
    }
    
    def __init__(self, model_name: str = None):
        """Initialize field mapping embedding generator."""
        self.model_name = model_name or settings.EMBEDDING_MODEL
        logger.info(f"Loading embedding model for field mapping: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        # Check if model requires Vietnamese tokenization
        from src.embeddings.model_selector import EmbeddingModelSelector
        model_info = EmbeddingModelSelector().get_model_info(self.model_name)
        if model_info:
            self.requires_vietnamese_tokenization = model_info.get('requires_tokenization', False)
        else:
            self.requires_vietnamese_tokenization = 'SimCSE-VietNamese' in self.model_name or 'SimCSE-Vietnamese' in self.model_name
        
        if self.requires_vietnamese_tokenization and not PYVI_AVAILABLE:
            logger.warning("Vietnamese tokenization required but pyvi not available.")
        
        logger.info("Field mapping embedding model loaded successfully")
    
    def _tokenize_vietnamese(self, text: str) -> str:
        """Tokenize Vietnamese text if required."""
        if not self.requires_vietnamese_tokenization or not PYVI_AVAILABLE:
            return text
        
        try:
            return vietnamese_tokenize(text)
        except Exception as e:
            logger.warning(f"Error tokenizing Vietnamese text: {e}. Using original text.")
            return text
    
    def generate_field_embedding(self, field_text: str, field_name: str = "") -> List[float]:
        """
        Generate embedding for a single field with improved preprocessing.
        
        Args:
            field_text: Text content of the field
            field_name: Name of the field (for context)
        
        Returns:
            Embedding vector
        """
        if not field_text or not field_text.strip():
            logger.debug(f"Empty text for field {field_name}")
            return [0.0] * self.model.get_sentence_embedding_dimension()
        
        try:
            # Normalize text
            processed_text = TextEnhancer.normalize_text(field_text.strip())
            
            # Limit text length to avoid truncation issues
            max_length = 512  # Safe limit for most models
            if len(processed_text) > max_length:
                processed_text = processed_text[:max_length]
            
            # Add field label for better context (improved format)
            if field_name:
                # Use more descriptive labels
                field_labels = {
                    'skills': 'Required Skills',
                    'experience': 'Work Experience',
                    'desired_job': 'Desired Position',
                    'requirements': 'Job Requirements',
                    'title': 'Job Title'
                }
                label = field_labels.get(field_name, field_name.replace('_', ' ').title())
                processed_text = f"{label}: {processed_text}"
            
            # Tokenize Vietnamese if required
            processed_text = self._tokenize_vietnamese(processed_text)
            
            # Generate embedding with optimized settings
            embedding = self.model.encode(
                processed_text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=1
            )
            
            # Ensure normalization (double-check)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            else:
                logger.warning(f"Zero norm embedding for field {field_name}")
            
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding for field {field_name}: {e}")
            raise
    
    def generate_field_embeddings_batch(
        self,
        field_texts: List[Tuple[str, str]],
        batch_size: int = 32
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple fields in batch for better performance.
        
        Args:
            field_texts: List of (field_text, field_name) tuples
            batch_size: Batch size for processing
        
        Returns:
            List of embedding vectors
        """
        if not field_texts:
            return []
        
        embeddings = []
        processed_texts = []
        
        for field_text, field_name in field_texts:
            if not field_text or not field_text.strip():
                embeddings.append([0.0] * self.model.get_sentence_embedding_dimension())
                processed_texts.append("")
                continue
            
            # Normalize and prepare text
            processed_text = TextEnhancer.normalize_text(field_text.strip())
            if len(processed_text) > 512:
                processed_text = processed_text[:512]
            
            # Add field label
            if field_name:
                field_labels = {
                    'skills': 'Required Skills',
                    'experience': 'Work Experience',
                    'desired_job': 'Desired Position',
                    'requirements': 'Job Requirements',
                    'title': 'Job Title'
                }
                label = field_labels.get(field_name, field_name.replace('_', ' ').title())
                processed_text = f"{label}: {processed_text}"
            
            processed_texts.append(processed_text)
        
        # Tokenize all texts if required
        if self.requires_vietnamese_tokenization and PYVI_AVAILABLE:
            try:
                processed_texts = [self._tokenize_vietnamese(text) if text else text for text in processed_texts]
            except Exception as e:
                logger.warning(f"Error tokenizing batch: {e}")
        
        # Generate embeddings in batch
        try:
            batch_embeddings = self.model.encode(
                processed_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=batch_size
            )
            
            # Ensure normalization
            norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            batch_embeddings = batch_embeddings / norms
            
            embeddings = batch_embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            # Fallback to individual generation
            embeddings = []
            for field_text, field_name in field_texts:
                embeddings.append(self.generate_field_embedding(field_text, field_name))
        
        return embeddings
    
    def generate_candidate_field_embeddings(self, candidate_fields: Dict[str, str]) -> Dict[str, List[float]]:
        """
        Generate embeddings for candidate fields that have mappings.
        
        Args:
            candidate_fields: Dictionary of field_name -> text content
        
        Returns:
            Dictionary of field_name -> embedding vector
        """
        embeddings = {}
        
        for candidate_field, jd_field in self.FIELD_MAPPINGS.items():
            if candidate_field in candidate_fields and candidate_fields[candidate_field]:
                field_text = candidate_fields[candidate_field]
                embeddings[candidate_field] = self.generate_field_embedding(
                    field_text, 
                    field_name=candidate_field
                )
            else:
                logger.debug(f"Candidate field '{candidate_field}' not available or empty")
        
        return embeddings
    
    def generate_jd_field_embeddings(self, jd_fields: Dict[str, str]) -> Dict[str, List[float]]:
        """
        Generate embeddings for JD fields that are targets of mappings.
        
        Args:
            jd_fields: Dictionary of field_name -> text content
        
        Returns:
            Dictionary of field_name -> embedding vector
        """
        embeddings = {}
        
        # Get all target JD fields from mappings
        target_jd_fields = set(self.FIELD_MAPPINGS.values())
        
        for jd_field in target_jd_fields:
            if jd_field in jd_fields and jd_fields[jd_field]:
                field_text = jd_fields[jd_field]
                embeddings[jd_field] = self.generate_field_embedding(
                    field_text,
                    field_name=jd_field
                )
            else:
                logger.debug(f"JD field '{jd_field}' not available or empty")
        
        return embeddings
    
    def calculate_field_similarities(
        self,
        candidate_embeddings: Dict[str, List[float]],
        jd_embeddings: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Calculate similarity scores for each field mapping with improved accuracy.
        
        Args:
            candidate_embeddings: Dictionary of candidate_field -> embedding
            jd_embeddings: Dictionary of jd_field -> embedding
        
        Returns:
            Dictionary of candidate_field -> similarity_score
        """
        similarities = {}
        
        for candidate_field, jd_field in self.FIELD_MAPPINGS.items():
            if candidate_field in candidate_embeddings and jd_field in jd_embeddings:
                candidate_vec = np.array(candidate_embeddings[candidate_field])
                jd_vec = np.array(jd_embeddings[jd_field])
                
                # Ensure vectors are normalized
                candidate_norm = np.linalg.norm(candidate_vec)
                jd_norm = np.linalg.norm(jd_vec)
                
                if candidate_norm > 0 and jd_norm > 0:
                    # Calculate cosine similarity (dot product of normalized vectors)
                    similarity = np.dot(candidate_vec, jd_vec)
                    # Clamp to [-1, 1] range
                    similarity = max(-1.0, min(1.0, float(similarity)))
                    similarities[candidate_field] = similarity
                else:
                    logger.warning(f"Zero norm vector for {candidate_field} -> {jd_field}")
                    similarities[candidate_field] = 0.0
            else:
                logger.debug(f"Missing embeddings for mapping {candidate_field} -> {jd_field}")
                similarities[candidate_field] = 0.0
        
        return similarities
    
    def calculate_combined_similarity(
        self,
        candidate_embeddings: Dict[str, List[float]],
        jd_embeddings: Dict[str, List[float]],
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate weighted combined similarity from all field mappings.
        
        Args:
            candidate_embeddings: Dictionary of candidate_field -> embedding
            jd_embeddings: Dictionary of jd_field -> embedding
            weights: Optional weights for each field (default: equal weights)
        
        Returns:
            Tuple of (combined_similarity, field_similarities)
        """
        field_similarities = self.calculate_field_similarities(candidate_embeddings, jd_embeddings)
        
        # Default weights (can be customized)
        if weights is None:
            weights = {
                'skills': 0.4,        # Skills are most important
                'experience': 0.35,    # Experience is very important
                'desired_job': 0.25   # Desired job is important but less than skills/experience
            }
        
        # Calculate weighted average
        total_weight = 0.0
        weighted_sum = 0.0
        
        for field, similarity in field_similarities.items():
            if field in weights:
                weight = weights[field]
                weighted_sum += similarity * weight
                total_weight += weight
        
        if total_weight > 0:
            combined_similarity = weighted_sum / total_weight
        else:
            combined_similarity = 0.0
        
        return combined_similarity, field_similarities
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.model.get_sentence_embedding_dimension()

