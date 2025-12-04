"""10 Model variations for benchmarking and optimization."""
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


class ModelVariationBase:
    """Base class for model variations."""
    
    def __init__(self, variation_id: int, name: str, model_name: str, **kwargs):
        self.variation_id = variation_id
        self.name = name
        self.model_name = model_name
        self.kwargs = kwargs
        logger.info(f"Loading variation {variation_id}: {name} with model {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.batch_size = kwargs.get('batch_size', 32)
        self.normalize = kwargs.get('normalize', True)
        self.use_tokenization = kwargs.get('use_tokenization', False)
    
    def _tokenize_vietnamese(self, text: str) -> str:
        """Tokenize Vietnamese text if needed."""
        if not self.use_tokenization or not PYVI_AVAILABLE:
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
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            batch_size=1
        )
        
        if self.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding.tolist()
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        
        processed_texts = [self._tokenize_vietnamese(t) if t and t.strip() else "" for t in texts]
        
        embeddings = self.model.encode(
            processed_texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            batch_size=self.batch_size
        )
        
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms
        
        return embeddings.tolist()
    
    def generate_jd_embedding(self, title: str, description: str, requirements: str,
                             skills: str = None, company: str = None) -> List[float]:
        """Generate JD embedding - default implementation."""
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
        """Generate candidate embedding - default implementation."""
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


class Variation1_CurrentModel(ModelVariationBase):
    """Variation 1: Current model (baseline) - VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"""
    def __init__(self):
        super().__init__(
            variation_id=1,
            name="Current_SimCSE_Vietnamese",
            model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
            batch_size=32,
            normalize=True,
            use_tokenization=True
        )


class Variation2_MultilingualMPNet(ModelVariationBase):
    """Variation 2: Multilingual MPNet - paraphrase-multilingual-mpnet-base-v2"""
    def __init__(self):
        super().__init__(
            variation_id=2,
            name="Multilingual_MPNet",
            model_name="paraphrase-multilingual-mpnet-base-v2",
            batch_size=32,
            normalize=True,
            use_tokenization=False
        )


class Variation3_VietnameseSBERT(ModelVariationBase):
    """Variation 3: Vietnamese SBERT - keepitreal/vietnamese-sbert"""
    def __init__(self):
        super().__init__(
            variation_id=3,
            name="Vietnamese_SBERT",
            model_name="keepitreal/vietnamese-sbert",
            batch_size=32,
            normalize=True,
            use_tokenization=False
        )


class Variation4_MiniLM_Multilingual(ModelVariationBase):
    """Variation 4: Fast Multilingual MiniLM - paraphrase-multilingual-MiniLM-L12-v2"""
    def __init__(self):
        super().__init__(
            variation_id=4,
            name="MiniLM_Multilingual",
            model_name="paraphrase-multilingual-MiniLM-L12-v2",
            batch_size=64,  # Larger batch for faster model
            normalize=True,
            use_tokenization=False
        )


class Variation5_MPNet_Base(ModelVariationBase):
    """Variation 5: High quality MPNet - all-mpnet-base-v2"""
    def __init__(self):
        super().__init__(
            variation_id=5,
            name="MPNet_Base",
            model_name="all-mpnet-base-v2",
            batch_size=32,
            normalize=True,
            use_tokenization=False
        )


class Variation6_QA_MPNet(ModelVariationBase):
    """Variation 6: QA-optimized MPNet - multi-qa-mpnet-base-dot-v1"""
    def __init__(self):
        super().__init__(
            variation_id=6,
            name="QA_MPNet",
            model_name="multi-qa-mpnet-base-dot-v1",
            batch_size=32,
            normalize=True,
            use_tokenization=False
        )


class Variation7_SimCSE_LargeBatch(ModelVariationBase):
    """Variation 7: Current model with large batch size for speed"""
    def __init__(self):
        super().__init__(
            variation_id=7,
            name="SimCSE_LargeBatch",
            model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
            batch_size=128,  # Large batch for speed
            normalize=True,
            use_tokenization=True
        )


class Variation8_SimCSE_NoNormalize(ModelVariationBase):
    """Variation 8: Current model without normalization"""
    def __init__(self):
        super().__init__(
            variation_id=8,
            name="SimCSE_NoNormalize",
            model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
            batch_size=32,
            normalize=False,  # No normalization
            use_tokenization=True
        )


class Variation9_Weighted_SimCSE(ModelVariationBase):
    """Variation 9: Weighted embedding approach with SimCSE"""
    def __init__(self):
        super().__init__(
            variation_id=9,
            name="Weighted_SimCSE",
            model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
            batch_size=32,
            normalize=True,
            use_tokenization=True
        )
        self.jd_weights = {'title': 7.0, 'skills': 6.5, 'requirements': 6.0, 'description': 3.0}
        self.candidate_weights = {'skills': 7.0, 'experience': 6.5, 'summary': 3.5, 'education': 3.0}
    
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
        
        weighted_sum = np.zeros(self.dimension)
        total_weight = 0.0
        for field, emb in embeddings.items():
            weight = self.jd_weights.get(field, 1.0)
            weighted_sum += emb * weight
            total_weight += weight
        
        if total_weight > 0:
            result = weighted_sum / total_weight
        else:
            result = np.zeros(self.dimension)
        
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
        
        weighted_sum = np.zeros(self.dimension)
        total_weight = 0.0
        for field, emb in embeddings.items():
            weight = self.candidate_weights.get(field, 1.0)
            weighted_sum += emb * weight
            total_weight += weight
        
        if total_weight > 0:
            result = weighted_sum / total_weight
        else:
            result = np.zeros(self.dimension)
        
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        return result.tolist()


class Variation10_MultiVector_SimCSE(ModelVariationBase):
    """Variation 10: Multi-vector pooling approach with SimCSE"""
    def __init__(self):
        super().__init__(
            variation_id=10,
            name="MultiVector_SimCSE",
            model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
            batch_size=32,
            normalize=True,
            use_tokenization=True
        )
        self.pooling = "weighted"
    
    def generate_jd_embedding(self, title: str, description: str, requirements: str,
                             skills: str = None, company: str = None) -> List[float]:
        """Generate multi-vector JD embedding."""
        vectors = []
        weights = []
        if title:
            vectors.append(np.array(self.generate_embedding(f"Title: {title}")))
            weights.append(3.0)
        if skills:
            vectors.append(np.array(self.generate_embedding(f"Skills: {skills}")))
            weights.append(3.0)
        if requirements:
            vectors.append(np.array(self.generate_embedding(f"Requirements: {requirements}")))
            weights.append(2.0)
        if description:
            vectors.append(np.array(self.generate_embedding(f"Description: {description}")))
            weights.append(1.0)
        
        if not vectors:
            return [0.0] * self.dimension
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        result = np.average(vectors, axis=0, weights=weights)
        
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        return result.tolist()
    
    def generate_candidate_embedding(self, skills: str, experience: str,
                                    education: str = None, summary: str = None) -> List[float]:
        """Generate multi-vector candidate embedding."""
        vectors = []
        weights = []
        if skills:
            vectors.append(np.array(self.generate_embedding(f"Skills: {skills}")))
            weights.append(3.0)
        if experience:
            vectors.append(np.array(self.generate_embedding(f"Experience: {experience}")))
            weights.append(3.0)
        if summary:
            vectors.append(np.array(self.generate_embedding(f"Summary: {summary}")))
            weights.append(2.0)
        if education:
            vectors.append(np.array(self.generate_embedding(f"Education: {education}")))
            weights.append(1.0)
        
        if not vectors:
            return [0.0] * self.dimension
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        result = np.average(vectors, axis=0, weights=weights)
        
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        return result.tolist()


def get_variation(variation_id: int):
    """Get model variation by ID."""
    variations = {
        1: Variation1_CurrentModel,
        2: Variation2_MultilingualMPNet,
        3: Variation3_VietnameseSBERT,
        4: Variation4_MiniLM_Multilingual,
        5: Variation5_MPNet_Base,
        6: Variation6_QA_MPNet,
        7: Variation7_SimCSE_LargeBatch,
        8: Variation8_SimCSE_NoNormalize,
        9: Variation9_Weighted_SimCSE,
        10: Variation10_MultiVector_SimCSE
    }
    
    variation_class = variations.get(variation_id)
    if not variation_class:
        raise ValueError(f"Unknown variation ID: {variation_id}")
    
    return variation_class()


def list_all_variations() -> List[Dict]:
    """List all available variations."""
    return [
        {"id": 1, "name": "Current_SimCSE_Vietnamese", "description": "Baseline: Current SimCSE Vietnamese model"},
        {"id": 2, "name": "Multilingual_MPNet", "description": "Multilingual MPNet model"},
        {"id": 3, "name": "Vietnamese_SBERT", "description": "Vietnamese-specific SBERT model"},
        {"id": 4, "name": "MiniLM_Multilingual", "description": "Fast multilingual MiniLM model"},
        {"id": 5, "name": "MPNet_Base", "description": "High quality MPNet base model"},
        {"id": 6, "name": "QA_MPNet", "description": "QA-optimized MPNet for matching"},
        {"id": 7, "name": "SimCSE_LargeBatch", "description": "SimCSE with large batch size (speed optimization)"},
        {"id": 8, "name": "SimCSE_NoNormalize", "description": "SimCSE without normalization"},
        {"id": 9, "name": "Weighted_SimCSE", "description": "Weighted embedding approach with SimCSE"},
        {"id": 10, "name": "MultiVector_SimCSE", "description": "Multi-vector pooling with SimCSE"}
    ]




