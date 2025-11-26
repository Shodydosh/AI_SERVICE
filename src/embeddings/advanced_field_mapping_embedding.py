"""Advanced field-by-field embedding generator with enhanced techniques for better quality."""
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


class AdvancedFieldMappingEmbeddingGenerator:
    """Advanced embedding generator with multiple enhancement techniques."""
    
    FIELD_MAPPINGS = {
        'skills': 'requirements',
        'experience': 'requirements',
        'desired_job': 'title'
    }
    
    # Enhanced stopwords
    STOPWORDS = {
        'và', 'của', 'cho', 'với', 'từ', 'trong', 'là', 'có', 'được', 'theo',
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that'
    }
    
    def __init__(self, model_name: str = None, use_semantic_expansion: bool = True, use_keyword_boost: bool = True):
        """
        Initialize advanced embedding generator.
        
        Args:
            model_name: Embedding model name
            use_semantic_expansion: Use semantic phrase expansion
            use_keyword_boost: Boost important keywords
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.use_semantic_expansion = use_semantic_expansion
        self.use_keyword_boost = use_keyword_boost
        
        logger.info(f"Loading advanced embedding model: {self.model_name}")
        logger.info(f"Semantic expansion: {use_semantic_expansion}, Keyword boost: {use_keyword_boost}")
        
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
        
        logger.info("Advanced embedding model loaded successfully")
    
    def _tokenize_vietnamese(self, text: str) -> str:
        """Tokenize Vietnamese text if required."""
        if not self.requires_vietnamese_tokenization or not PYVI_AVAILABLE:
            return text
        
        try:
            return vietnamese_tokenize(text)
        except Exception as e:
            logger.warning(f"Error tokenizing Vietnamese text: {e}. Using original text.")
            return text
    
    def _extract_keywords_advanced(self, text: str, max_keywords: int = 25) -> List[str]:
        """Extract important keywords with frequency weighting."""
        if not text:
            return []
        
        # Split by common delimiters
        words = re.split(r'[,\s;|/\\\-]+', text.lower())
        
        # Count word frequencies
        word_freq = {}
        for w in words:
            w = w.strip()
            if len(w) > 2 and w not in self.STOPWORDS and not w.isdigit():
                word_freq[w] = word_freq.get(w, 0) + 1
        
        # Sort by frequency and length (prefer longer, more frequent words)
        sorted_words = sorted(
            word_freq.items(),
            key=lambda x: (x[1], len(x[0])),
            reverse=True
        )
        
        return [w for w, _ in sorted_words[:max_keywords]]
    
    def _semantic_expand_text(self, text: str, field_name: str) -> str:
        """Expand text with semantic variations for better matching."""
        if not text or not self.use_semantic_expansion:
            return text
        
        # Add semantic variations based on field type
        variations = []
        
        if field_name in ['skills', 'requirements']:
            # For skills, add proficiency indicators
            variations.append(f"Proficient in {text}")
            variations.append(f"Experience with {text}")
        
        elif field_name in ['experience', 'work_experience']:
            # For experience, add context
            variations.append(f"Worked as {text}")
            variations.append(f"Background in {text}")
        
        elif field_name in ['title', 'desired_job']:
            # For titles, add role variations
            variations.append(f"Position: {text}")
            variations.append(f"Role: {text}")
        
        # Combine with original (original first for emphasis)
        if variations:
            return f"{text} {' '.join(variations[:2])}"  # Limit to 2 variations
        
        return text
    
    def _enhance_text_advanced(self, text: str, field_name: str, field_type: str = "candidate") -> str:
        """Advanced text enhancement with multiple techniques."""
        if not text or not text.strip():
            return ""
        
        # Step 1: Normalize
        processed = TextEnhancer.normalize_text(text.strip())
        
        # Step 2: Extract and boost keywords
        if self.use_keyword_boost:
            keywords = self._extract_keywords_advanced(processed, max_keywords=20)
            if keywords:
                # Add top keywords for emphasis (but keep original prominent)
                top_keywords = " ".join(keywords[:10])
                processed = f"{processed} {top_keywords}"
        
        # Step 3: Semantic expansion
        if self.use_semantic_expansion:
            processed = self._semantic_expand_text(processed, field_name)
        
        # Step 4: Limit length
        max_length = 512
        if len(processed) > max_length:
            # Truncate but try to keep complete words
            truncated = processed[:max_length]
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.8:  # If we can find a good break point
                processed = truncated[:last_space]
            else:
                processed = truncated
        
        # Step 5: Add context-aware labels
        context_labels = {
            'skills': 'Technical Skills and Competencies',
            'experience': 'Professional Work Experience',
            'desired_job': 'Target Job Position and Role',
            'requirements': 'Job Requirements and Qualifications',
            'title': 'Job Title and Position'
        }
        
        label = context_labels.get(field_name, field_name.replace('_', ' ').title())
        
        # Step 6: Format with context
        if field_type == "candidate":
            formatted = f"Candidate Profile - {label}: {processed}"
        else:
            formatted = f"Job Description - {label}: {processed}"
        
        return formatted
    
    def generate_field_embedding(self, field_text: str, field_name: str = "", field_type: str = "candidate") -> List[float]:
        """Generate embedding with advanced preprocessing and better error handling."""
        if not field_text or not field_text.strip():
            logger.debug(f"Empty text for field {field_name}")
            # Return a small random vector instead of zero to avoid filtering issues
            dim = self.model.get_sentence_embedding_dimension()
            random_vec = np.random.normal(0, 0.001, size=dim)
            norm = np.linalg.norm(random_vec)
            if norm > 0:
                random_vec = random_vec / norm
            return random_vec.tolist()
        
        try:
            # Advanced preprocessing
            processed_text = self._enhance_text_advanced(field_text, field_name, field_type)
            
            # Ensure we have some text after processing
            if not processed_text or not processed_text.strip():
                logger.warning(f"Text became empty after processing for field {field_name}, using original")
                processed_text = field_text[:512]  # Use original with length limit
            
            # Tokenize Vietnamese if required
            processed_text = self._tokenize_vietnamese(processed_text)
            
            # Generate embedding with better parameters
            embedding = self.model.encode(
                processed_text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=1,
                device='cpu'  # Explicit device
            )
            
            # Ensure normalization
            norm = np.linalg.norm(embedding)
            if norm > 1e-6:
                embedding = embedding / norm
            else:
                logger.warning(f"Zero norm embedding for field {field_name}, using fallback")
                # Fallback: use a small random vector
                dim = embedding.shape[0]
                embedding = np.random.normal(0, 0.01, size=dim)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding for field {field_name}: {e}")
            # Return a fallback embedding instead of raising
            dim = self.model.get_sentence_embedding_dimension()
            fallback = np.random.normal(0, 0.001, size=dim)
            norm = np.linalg.norm(fallback)
            if norm > 0:
                fallback = fallback / norm
            return fallback.tolist()
    
    def generate_candidate_field_embeddings(self, candidate_fields: Dict[str, str]) -> Dict[str, List[float]]:
        """Generate embeddings for candidate fields with improved handling."""
        embeddings = {}
        
        for candidate_field, jd_field in self.FIELD_MAPPINGS.items():
            if candidate_field in candidate_fields and candidate_fields[candidate_field]:
                field_text = candidate_fields[candidate_field]
                # Ensure field_text is not empty
                if field_text and str(field_text).strip():
                    embeddings[candidate_field] = self.generate_field_embedding(
                        str(field_text).strip(),
                        field_name=candidate_field,
                        field_type="candidate"
                    )
                else:
                    logger.debug(f"Candidate field '{candidate_field}' is empty")
            else:
                logger.debug(f"Candidate field '{candidate_field}' not available")
        
        # Ensure at least one embedding exists
        if not embeddings:
            logger.warning("No candidate field embeddings generated, creating fallback")
            # Create a minimal embedding from any available text
            all_text = " ".join([str(v) for v in candidate_fields.values() if v and str(v).strip()])
            if all_text:
                embeddings['skills'] = self.generate_field_embedding(
                    all_text[:200],  # Use first 200 chars
                    field_name='skills',
                    field_type="candidate"
                )
        
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
    
    def combine_field_embeddings_advanced(
        self,
        field_embeddings: Dict[str, List[float]],
        weights: Dict[str, float],
        content_lengths: Optional[Dict[str, int]] = None
    ) -> List[float]:
        """
        Advanced combination using learned attention mechanism.
        
        Combines embeddings with:
        1. Base weights
        2. Content length factors
        3. Embedding quality factors (norm-based)
        4. Field importance boost
        """
        if not field_embeddings:
            return [0.0] * self.model.get_sentence_embedding_dimension()
        
        # Calculate attention scores
        attention_scores = {}
        total_score = 0.0
        
        for field, embedding in field_embeddings.items():
            if field in weights:
                base_weight = weights[field]
                
                # Factor 1: Content length
                length_factor = 1.0
                if content_lengths and field in content_lengths:
                    length = content_lengths[field]
                    # Optimal length range: 20-300 chars
                    if 20 <= length <= 300:
                        length_factor = 1.3
                    elif 10 <= length < 20:
                        length_factor = 1.1
                    elif 300 < length <= 500:
                        length_factor = 1.0
                    else:
                        length_factor = 0.9
                
                # Factor 2: Embedding quality (norm-based)
                emb_array = np.array(embedding)
                emb_norm = np.linalg.norm(emb_array)
                quality_factor = min(1.2, max(0.8, emb_norm))  # Normalize should be ~1.0
                
                # Factor 3: Field importance boost
                importance_boost = 1.0
                if field == 'skills':
                    importance_boost = 1.15  # Skills are most important
                elif field == 'experience':
                    importance_boost = 1.10
                
                # Combined attention score
                attention_score = base_weight * length_factor * quality_factor * importance_boost
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
            # Final normalization
            norm = np.linalg.norm(combined)
            if norm > 1e-6:
                combined = combined / norm
            else:
                # Fallback: use average
                combined = np.mean([np.array(emb) for emb in field_embeddings.values()], axis=0)
                norm = np.linalg.norm(combined)
                if norm > 1e-6:
                    combined = combined / norm
        
        return combined.tolist() if combined is not None else [0.0] * self.model.get_sentence_embedding_dimension()
    
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
                
                if cand_norm > 1e-6 and jd_norm > 1e-6:
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
        """Calculate weighted combined similarity with advanced weighting."""
        field_similarities = self.calculate_field_similarities(candidate_embeddings, jd_embeddings)
        
        # Advanced weights
        if weights is None:
            weights = {
                'skills': 0.48,      # Increased
                'experience': 0.35,  # Maintained
                'desired_job': 0.17  # Reduced
            }
        
        # Use geometric mean for better handling of low similarities
        # This emphasizes fields with high similarity
        weighted_product = 1.0
        total_weight = 0.0
        
        for field, similarity in field_similarities.items():
            if field in weights and weights[field] > 0:
                weight = weights[field]
                # Apply non-linear transformation: boost high similarities more
                adjusted_similarity = (similarity + 1) / 2  # Normalize to [0, 1]
                adjusted_similarity = adjusted_similarity ** (1 / (1 + weight))  # Weighted power
                weighted_product *= (adjusted_similarity ** weight)
                total_weight += weight
        
        if total_weight > 0 and weighted_product > 0:
            # Convert back to [-1, 1] range
            combined_similarity = 2 * (weighted_product ** (1 / total_weight)) - 1
        else:
            # Fallback to arithmetic mean
            weighted_sum = sum(sim * weights.get(field, 0) for field, sim in field_similarities.items())
            total_weight = sum(weights.get(field, 0) for field in field_similarities.keys())
            combined_similarity = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return combined_similarity, field_similarities
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.model.get_sentence_embedding_dimension()

