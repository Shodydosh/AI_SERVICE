"""Improved field-by-field embedding generator with advanced text processing and better combination strategies."""
from typing import List, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import re
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


class ImprovedFieldMappingEmbeddingGenerator:
    """Improved embedding generator with advanced preprocessing and better combination strategies."""
    
    # Field mappings: candidate_field -> jd_field
    FIELD_MAPPINGS = {
        'skills': 'requirements',
        'experience': 'requirements',
        'desired_job': 'title'
    }
    
    # Stopwords to remove (common Vietnamese + English)
    STOPWORDS = {
        'và', 'của', 'cho', 'với', 'từ', 'trong', 'là', 'có', 'được', 'theo',
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'must', 'can'
    }
    
    def __init__(self, model_name: str = None, combination_method: str = "weighted_concatenate"):
        """
        Initialize improved embedding generator.
        
        Args:
            model_name: Embedding model name
            combination_method: "weighted_average", "weighted_concatenate", or "attention_weighted"
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.combination_method = combination_method
        logger.info(f"Loading improved embedding model: {self.model_name}")
        logger.info(f"Combination method: {combination_method}")
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
        
        logger.info("Improved embedding model loaded successfully")
    
    def _tokenize_vietnamese(self, text: str) -> str:
        """Tokenize Vietnamese text if required."""
        if not self.requires_vietnamese_tokenization or not PYVI_AVAILABLE:
            return text
        
        try:
            return vietnamese_tokenize(text)
        except Exception as e:
            logger.warning(f"Error tokenizing Vietnamese text: {e}. Using original text.")
            return text
    
    def _extract_keywords(self, text: str, max_keywords: int = 20) -> str:
        """Extract important keywords from text."""
        if not text:
            return ""
        
        # Split by common delimiters
        words = re.split(r'[,\s;|/\\]+', text.lower())
        
        # Filter out stopwords and short words
        keywords = [
            w.strip() for w in words 
            if len(w.strip()) > 2 
            and w.strip() not in self.STOPWORDS
            and not w.strip().isdigit()
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        # Return top keywords
        return " ".join(unique_keywords[:max_keywords])
    
    def _enhance_text_with_context(self, text: str, field_name: str, field_type: str = "candidate") -> str:
        """Enhance text with better context and preprocessing."""
        if not text or not text.strip():
            return ""
        
        # Normalize
        processed = TextEnhancer.normalize_text(text.strip())
        
        # Extract keywords for important fields
        if field_name in ['skills', 'requirements']:
            keywords = self._extract_keywords(processed, max_keywords=15)
            if keywords:
                # Combine original with keywords for better semantic understanding
                processed = f"{processed} {keywords}"
        
        # Limit length
        max_length = 512
        if len(processed) > max_length:
            processed = processed[:max_length]
        
        # Add context-aware labels
        context_labels = {
            'skills': 'Technical Skills Required',
            'experience': 'Professional Experience Required',
            'desired_job': 'Target Job Position',
            'requirements': 'Job Requirements and Qualifications',
            'title': 'Job Title and Role'
        }
        
        label = context_labels.get(field_name, field_name.replace('_', ' ').title())
        
        # Use better format for embedding
        if field_type == "candidate":
            formatted = f"Candidate {label}: {processed}"
        else:
            formatted = f"Job {label}: {processed}"
        
        return formatted
    
    def generate_field_embedding(self, field_text: str, field_name: str = "", field_type: str = "candidate") -> List[float]:
        """Generate embedding with improved preprocessing."""
        if not field_text or not field_text.strip():
            logger.debug(f"Empty text for field {field_name}")
            return [0.0] * self.model.get_sentence_embedding_dimension()
        
        try:
            # Enhanced preprocessing
            processed_text = self._enhance_text_with_context(field_text, field_name, field_type)
            
            # Tokenize Vietnamese if required
            processed_text = self._tokenize_vietnamese(processed_text)
            
            # Generate embedding
            embedding = self.model.encode(
                processed_text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=1
            )
            
            # Ensure normalization
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            else:
                logger.warning(f"Zero norm embedding for field {field_name}")
            
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding for field {field_name}: {e}")
            raise
    
    def generate_candidate_field_embeddings(self, candidate_fields: Dict[str, str]) -> Dict[str, List[float]]:
        """Generate embeddings for candidate fields."""
        embeddings = {}
        
        for candidate_field, jd_field in self.FIELD_MAPPINGS.items():
            if candidate_field in candidate_fields and candidate_fields[candidate_field]:
                field_text = candidate_fields[candidate_field]
                embeddings[candidate_field] = self.generate_field_embedding(
                    field_text,
                    field_name=candidate_field,
                    field_type="candidate"
                )
            else:
                logger.debug(f"Candidate field '{candidate_field}' not available")
        
        return embeddings
    
    def generate_jd_field_embeddings(self, jd_fields: Dict[str, str]) -> Dict[str, List[float]]:
        """Generate embeddings for JD fields."""
        embeddings = {}
        
        target_jd_fields = set(self.FIELD_MAPPINGS.values())
        
        for jd_field in target_jd_fields:
            if jd_field in jd_fields and jd_fields[jd_field]:
                field_text = jd_fields[jd_field]
                embeddings[jd_field] = self.generate_field_embedding(
                    field_text,
                    field_name=jd_field,
                    field_type="jd"
                )
            else:
                logger.debug(f"JD field '{jd_field}' not available")
        
        return embeddings
    
    def _combine_embeddings_weighted_average(
        self,
        field_embeddings: Dict[str, List[float]],
        weights: Dict[str, float]
    ) -> List[float]:
        """Combine embeddings using weighted average."""
        combined = None
        total_weight = 0.0
        
        for field, embedding in field_embeddings.items():
            if field in weights and weights[field] > 0:
                emb_array = np.array(embedding) * weights[field]
                if combined is None:
                    combined = emb_array
                else:
                    combined += emb_array
                total_weight += weights[field]
        
        if combined is not None and total_weight > 0:
            combined = combined / total_weight
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm
        
        return combined.tolist() if combined is not None else [0.0] * self.model.get_sentence_embedding_dimension()
    
    def _combine_embeddings_weighted_concatenate(
        self,
        field_embeddings: Dict[str, List[float]],
        weights: Dict[str, float]
    ) -> List[float]:
        """Combine embeddings using weighted concatenation (better for differentiation)."""
        # Normalize each embedding first
        normalized_embs = {}
        for field, embedding in field_embeddings.items():
            if field in weights and weights[field] > 0:
                emb_array = np.array(embedding)
                norm = np.linalg.norm(emb_array)
                if norm > 0:
                    emb_array = emb_array / norm
                # Apply weight
                emb_array = emb_array * np.sqrt(weights[field])  # Use sqrt for better distribution
                normalized_embs[field] = emb_array
        
        if not normalized_embs:
            return [0.0] * self.model.get_sentence_embedding_dimension()
        
        # Concatenate weighted embeddings
        # Sort by weight for consistency
        sorted_fields = sorted(
            normalized_embs.items(),
            key=lambda x: weights.get(x[0], 0),
            reverse=True
        )
        
        # Concatenate (but we need to keep same dimension, so use weighted sum of concatenated parts)
        # Alternative: Use PCA or just weighted sum with different approach
        # For now, use weighted sum but with better weighting
        combined = None
        total_weight = 0.0
        
        for field, emb_array in sorted_fields:
            weight = weights.get(field, 0)
            # Use power weighting for better differentiation
            power_weight = weight ** 1.5  # Emphasize important fields more
            if combined is None:
                combined = emb_array * power_weight
            else:
                combined += emb_array * power_weight
            total_weight += power_weight
        
        if combined is not None and total_weight > 0:
            combined = combined / total_weight
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm
        
        return combined.tolist() if combined is not None else [0.0] * self.model.get_sentence_embedding_dimension()
    
    def _combine_embeddings_attention_weighted(
        self,
        field_embeddings: Dict[str, List[float]],
        weights: Dict[str, float],
        content_lengths: Optional[Dict[str, int]] = None
    ) -> List[float]:
        """Combine embeddings using attention-like weighting based on content."""
        if not field_embeddings:
            return [0.0] * self.model.get_sentence_embedding_dimension()
        
        # Calculate attention scores based on weights and content
        attention_scores = {}
        total_score = 0.0
        
        for field, embedding in field_embeddings.items():
            if field in weights:
                base_weight = weights[field]
                
                # Adjust based on content length if provided
                length_factor = 1.0
                if content_lengths and field in content_lengths:
                    length = content_lengths[field]
                    # Prefer fields with moderate length (not too short, not too long)
                    if 10 <= length <= 200:
                        length_factor = 1.2
                    elif length > 200:
                        length_factor = 1.0
                    else:
                        length_factor = 0.8
                
                attention_score = base_weight * length_factor
                attention_scores[field] = attention_score
                total_score += attention_score
        
        # Normalize attention scores
        if total_score > 0:
            attention_scores = {k: v / total_score for k, v in attention_scores.items()}
        else:
            # Equal weights if no valid scores
            attention_scores = {k: 1.0 / len(field_embeddings) for k in field_embeddings.keys()}
        
        # Weighted combination
        combined = None
        for field, embedding in field_embeddings.items():
            if field in attention_scores:
                emb_array = np.array(embedding) * attention_scores[field]
                if combined is None:
                    combined = emb_array
                else:
                    combined += emb_array
        
        if combined is not None:
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm
        
        return combined.tolist() if combined is not None else [0.0] * self.model.get_sentence_embedding_dimension()
    
    def combine_field_embeddings(
        self,
        field_embeddings: Dict[str, List[float]],
        weights: Dict[str, float],
        content_lengths: Optional[Dict[str, int]] = None
    ) -> List[float]:
        """
        Combine field embeddings using specified method.
        
        Args:
            field_embeddings: Dictionary of field -> embedding
            weights: Dictionary of field -> weight
            content_lengths: Optional dictionary of field -> text length
        
        Returns:
            Combined embedding vector
        """
        if not field_embeddings:
            return [0.0] * self.model.get_sentence_embedding_dimension()
        
        if self.combination_method == "weighted_average":
            return self._combine_embeddings_weighted_average(field_embeddings, weights)
        elif self.combination_method == "weighted_concatenate":
            return self._combine_embeddings_weighted_concatenate(field_embeddings, weights)
        elif self.combination_method == "attention_weighted":
            return self._combine_embeddings_attention_weighted(field_embeddings, weights, content_lengths)
        else:
            logger.warning(f"Unknown combination method: {self.combination_method}, using weighted_average")
            return self._combine_embeddings_weighted_average(field_embeddings, weights)
    
    def calculate_field_similarities(
        self,
        candidate_embeddings: Dict[str, List[float]],
        jd_embeddings: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Calculate similarity scores for each field mapping."""
        similarities = {}
        
        for candidate_field, jd_field in self.FIELD_MAPPINGS.items():
            if candidate_field in candidate_embeddings and jd_field in jd_embeddings:
                candidate_vec = np.array(candidate_embeddings[candidate_field])
                jd_vec = np.array(jd_embeddings[jd_field])
                
                # Ensure normalized
                cand_norm = np.linalg.norm(candidate_vec)
                jd_norm = np.linalg.norm(jd_vec)
                
                if cand_norm > 0 and jd_norm > 0:
                    similarity = np.dot(candidate_vec, jd_vec)
                    similarity = max(-1.0, min(1.0, float(similarity)))
                    similarities[candidate_field] = similarity
                else:
                    similarities[candidate_field] = 0.0
            else:
                similarities[candidate_field] = 0.0
        
        return similarities
    
    def calculate_combined_similarity(
        self,
        candidate_embeddings: Dict[str, List[float]],
        jd_embeddings: Dict[str, List[float]],
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate weighted combined similarity with improved weighting."""
        field_similarities = self.calculate_field_similarities(candidate_embeddings, jd_embeddings)
        
        # Improved weights with better distribution
        if weights is None:
            weights = {
                'skills': 0.45,      # Increased importance
                'experience': 0.35,  # Maintained
                'desired_job': 0.20  # Slightly reduced
            }
        
        # Use harmonic mean for better handling of low similarities
        # This gives more weight to fields with higher similarity
        total_weight = 0.0
        weighted_sum = 0.0
        
        for field, similarity in field_similarities.items():
            if field in weights:
                weight = weights[field]
                # Apply non-linear weighting to emphasize high similarities
                adjusted_similarity = similarity ** 0.9  # Slight boost for high similarities
                weighted_sum += adjusted_similarity * weight
                total_weight += weight
        
        if total_weight > 0:
            combined_similarity = weighted_sum / total_weight
        else:
            combined_similarity = 0.0
        
        return combined_similarity, field_similarities
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.model.get_sentence_embedding_dimension()

