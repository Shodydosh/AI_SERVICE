"""Different embedding methods for research evaluation."""
from typing import List, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from config.settings import settings

logger = logging.getLogger(__name__)

# Try to import pyvi
try:
    from pyvi.ViTokenizer import tokenize as vietnamese_tokenize
    PYVI_AVAILABLE = True
except ImportError:
    PYVI_AVAILABLE = False


class EmbeddingMethodBase:
    """Base class for embedding methods."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        logger.info(f"Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def _tokenize_vietnamese(self, text: str) -> str:
        """Tokenize Vietnamese text if needed."""
        if not PYVI_AVAILABLE:
            return text
        try:
            return vietnamese_tokenize(text)
        except:
            return text
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if not text or not text.strip():
            return [0.0] * self.dimension
        
        processed_text = self._tokenize_vietnamese(text)
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
        return embedding.tolist()


class Method1_Baseline(EmbeddingMethodBase):
    """Method 1: Baseline - Simple concatenation."""
    
    def __init__(self, model_name: str = None):
        super().__init__(model_name)
        self.method_name = "Baseline_SimCSE"
    
    def generate_jd_embedding(self, title: str, description: str, requirements: str, 
                             skills: str = None, company: str = None) -> List[float]:
        """Generate JD embedding by simple concatenation."""
        parts = []
        if title:
            parts.append(f"Title: {title}")
        if skills:
            parts.append(f"Skills: {skills}")
        if requirements:
            parts.append(f"Requirements: {requirements}")
        if description:
            parts.append(f"Description: {description}")
        
        combined = " ".join(parts)
        return self.generate_embedding(combined)
    
    def generate_candidate_embedding(self, skills: str, experience: str, 
                                    education: str = None, summary: str = None) -> List[float]:
        """Generate candidate embedding by simple concatenation."""
        parts = []
        if skills:
            parts.append(f"Skills: {skills}")
        if experience:
            parts.append(f"Experience: {experience}")
        if summary:
            parts.append(f"Summary: {summary}")
        if education:
            parts.append(f"Education: {education}")
        
        combined = " ".join(parts)
        return self.generate_embedding(combined)


class Method2_Weighted(EmbeddingMethodBase):
    """Method 2: Weighted embeddings."""
    
    def __init__(self, model_name: str = None):
        super().__init__(model_name)
        self.method_name = "Weighted_Embeddings"
        
        # Field weights
        self.jd_weights = {
            'title': 7.0,
            'skills': 6.5,
            'requirements': 6.0,
            'description': 3.0,
            'company': 0.5,
            'location': 0.5
        }
        
        self.candidate_weights = {
            'skills': 7.0,
            'experience': 6.5,
            'summary': 3.5,
            'education': 3.0
        }
    
    def generate_jd_embedding(self, title: str, description: str, requirements: str,
                             skills: str = None, company: str = None) -> List[float]:
        """Generate weighted JD embedding."""
        embeddings = {}
        
        if title:
            embeddings['title'] = np.array(self.generate_embedding(f"Title: {title}"))
        if skills:
            embeddings['skills'] = np.array(self.generate_embedding(f"Skills: {skills}"))
        if requirements:
            embeddings['requirements'] = np.array(self.generate_embedding(f"Requirements: {requirements}"))
        if description:
            embeddings['description'] = np.array(self.generate_embedding(f"Description: {description}"))
        
        # Weighted combination
        weighted_sum = np.zeros(self.dimension)
        total_weight = 0.0
        
        for field, embedding in embeddings.items():
            weight = self.jd_weights.get(field, 1.0)
            weighted_sum += embedding * weight
            total_weight += weight
        
        if total_weight > 0:
            result = weighted_sum / total_weight
        else:
            result = np.zeros(self.dimension)
        
        # Normalize
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        
        return result.tolist()
    
    def generate_candidate_embedding(self, skills: str, experience: str,
                                    education: str = None, summary: str = None) -> List[float]:
        """Generate weighted candidate embedding."""
        embeddings = {}
        
        if skills:
            embeddings['skills'] = np.array(self.generate_embedding(f"Skills: {skills}"))
        if experience:
            embeddings['experience'] = np.array(self.generate_embedding(f"Experience: {experience}"))
        if summary:
            embeddings['summary'] = np.array(self.generate_embedding(f"Summary: {summary}"))
        if education:
            embeddings['education'] = np.array(self.generate_embedding(f"Education: {education}"))
        
        # Weighted combination
        weighted_sum = np.zeros(self.dimension)
        total_weight = 0.0
        
        for field, embedding in embeddings.items():
            weight = self.candidate_weights.get(field, 1.0)
            weighted_sum += embedding * weight
            total_weight += weight
        
        if total_weight > 0:
            result = weighted_sum / total_weight
        else:
            result = np.zeros(self.dimension)
        
        # Normalize
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        
        return result.tolist()


class Method3_FieldSpecific(EmbeddingMethodBase):
    """Method 3: Field-specific embeddings with concatenation."""
    
    def __init__(self, model_name: str = None):
        super().__init__(model_name)
        self.method_name = "Field_Specific"
        # Use half dimension per field, then concatenate
        self.fields_dim = self.dimension // 2  # Split dimension
    
    def generate_jd_embedding(self, title: str, description: str, requirements: str,
                             skills: str = None, company: str = None) -> List[float]:
        """Generate field-specific JD embedding."""
        field_embeddings = []
        
        # Title + Skills (most important)
        if title or skills:
            combined = f"Title: {title or ''} Skills: {skills or ''}"
            emb = self.generate_embedding(combined)
            field_embeddings.append(emb[:self.fields_dim])
        
        # Requirements + Description
        if requirements or description:
            combined = f"Requirements: {requirements or ''} Description: {description or ''}"
            emb = self.generate_embedding(combined)
            field_embeddings.append(emb[:self.fields_dim])
        
        # Concatenate and pad if needed
        if field_embeddings:
            result = np.concatenate(field_embeddings)
            # Pad or truncate to target dimension
            if len(result) < self.dimension:
                result = np.pad(result, (0, self.dimension - len(result)))
            elif len(result) > self.dimension:
                result = result[:self.dimension]
        else:
            result = np.zeros(self.dimension)
        
        # Normalize
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        
        return result.tolist()
    
    def generate_candidate_embedding(self, skills: str, experience: str,
                                    education: str = None, summary: str = None) -> List[float]:
        """Generate field-specific candidate embedding."""
        field_embeddings = []
        
        # Skills + Experience (most important)
        if skills or experience:
            combined = f"Skills: {skills or ''} Experience: {experience or ''}"
            emb = self.generate_embedding(combined)
            field_embeddings.append(emb[:self.fields_dim])
        
        # Summary + Education
        if summary or education:
            combined = f"Summary: {summary or ''} Education: {education or ''}"
            emb = self.generate_embedding(combined)
            field_embeddings.append(emb[:self.fields_dim])
        
        # Concatenate
        if field_embeddings:
            result = np.concatenate(field_embeddings)
            if len(result) < self.dimension:
                result = np.pad(result, (0, self.dimension - len(result)))
            elif len(result) > self.dimension:
                result = result[:self.dimension]
        else:
            result = np.zeros(self.dimension)
        
        # Normalize
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        
        return result.tolist()


class Method4_MultiVector(EmbeddingMethodBase):
    """Method 4: Multi-vector embeddings with pooling."""
    
    def __init__(self, model_name: str = None, pooling: str = "mean"):
        super().__init__(model_name)
        self.method_name = "Multi_Vector"
        self.pooling = pooling  # "mean", "max", or "weighted"
    
    def generate_jd_embedding(self, title: str, description: str, requirements: str,
                             skills: str = None, company: str = None) -> List[float]:
        """Generate multi-vector JD embedding."""
        vectors = []
        weights = []
        
        # Title vector (high weight)
        if title:
            vectors.append(np.array(self.generate_embedding(f"Title: {title}")))
            weights.append(3.0)
        
        # Skills vector (high weight)
        if skills:
            vectors.append(np.array(self.generate_embedding(f"Skills: {skills}")))
            weights.append(3.0)
        
        # Requirements vector (medium weight)
        if requirements:
            vectors.append(np.array(self.generate_embedding(f"Requirements: {requirements}")))
            weights.append(2.0)
        
        # Description vector (low weight)
        if description:
            vectors.append(np.array(self.generate_embedding(f"Description: {description}")))
            weights.append(1.0)
        
        if not vectors:
            return [0.0] * self.dimension
        
        # Pooling
        if self.pooling == "mean":
            result = np.mean(vectors, axis=0)
        elif self.pooling == "max":
            result = np.max(vectors, axis=0)
        elif self.pooling == "weighted":
            weights = np.array(weights)
            weights = weights / weights.sum()
            result = np.average(vectors, axis=0, weights=weights)
        else:
            result = np.mean(vectors, axis=0)
        
        # Normalize
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        
        return result.tolist()
    
    def generate_candidate_embedding(self, skills: str, experience: str,
                                    education: str = None, summary: str = None) -> List[float]:
        """Generate multi-vector candidate embedding."""
        vectors = []
        weights = []
        
        # Skills vector (high weight)
        if skills:
            vectors.append(np.array(self.generate_embedding(f"Skills: {skills}")))
            weights.append(3.0)
        
        # Experience vector (high weight)
        if experience:
            vectors.append(np.array(self.generate_embedding(f"Experience: {experience}")))
            weights.append(3.0)
        
        # Summary vector (medium weight)
        if summary:
            vectors.append(np.array(self.generate_embedding(f"Summary: {summary}")))
            weights.append(2.0)
        
        # Education vector (low weight)
        if education:
            vectors.append(np.array(self.generate_embedding(f"Education: {education}")))
            weights.append(1.0)
        
        if not vectors:
            return [0.0] * self.dimension
        
        # Pooling
        if self.pooling == "mean":
            result = np.mean(vectors, axis=0)
        elif self.pooling == "max":
            result = np.max(vectors, axis=0)
        elif self.pooling == "weighted":
            weights = np.array(weights)
            weights = weights / weights.sum()
            result = np.average(vectors, axis=0, weights=weights)
        else:
            result = np.mean(vectors, axis=0)
        
        # Normalize
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        
        return result.tolist()


class Method5_Ensemble(EmbeddingMethodBase):
    """Method 5: Ensemble of multiple models."""
    
    def __init__(self, model_names: List[str] = None):
        if model_names is None:
            # Use current model as primary, could add more
            model_names = [settings.EMBEDDING_MODEL]
        
        self.model_name = model_names[0]  # Primary model
        super().__init__(self.model_name)
        self.method_name = "Ensemble"
        self.models = [self.model]  # For now, single model (can extend)
        self.weights = [1.0]  # Equal weights
    
    def generate_jd_embedding(self, title: str, description: str, requirements: str,
                             skills: str = None, company: str = None) -> List[float]:
        """Generate ensemble JD embedding."""
        combined = " ".join([
            f"Title: {title}" if title else "",
            f"Skills: {skills}" if skills else "",
            f"Requirements: {requirements}" if requirements else "",
            f"Description: {description}" if description else ""
        ])
        
        # For now, use single model (can extend to multiple)
        return self.generate_embedding(combined)
    
    def generate_candidate_embedding(self, skills: str, experience: str,
                                    education: str = None, summary: str = None) -> List[float]:
        """Generate ensemble candidate embedding."""
        combined = " ".join([
            f"Skills: {skills}" if skills else "",
            f"Experience: {experience}" if experience else "",
            f"Summary: {summary}" if summary else "",
            f"Education: {education}" if education else ""
        ])
        
        return self.generate_embedding(combined)


def get_embedding_method(method_id: int, **kwargs):
    """Get embedding method by ID."""
    methods = {
        1: Method1_Baseline,
        2: Method2_Weighted,
        3: Method3_FieldSpecific,
        4: Method4_MultiVector,
        5: Method5_Ensemble
    }
    
    method_class = methods.get(method_id)
    if not method_class:
        raise ValueError(f"Unknown method ID: {method_id}")
    
    return method_class(**kwargs)

