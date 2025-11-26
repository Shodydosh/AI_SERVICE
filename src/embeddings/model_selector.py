"""Embedding model selector with recommended models for job matching."""
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class EmbeddingModelSelector:
    """Provides recommended embedding models for job recommendation tasks."""
    
    # Recommended models for job matching
    RECOMMENDED_MODELS = {
        "all-MiniLM-L6-v2": {
            "name": "all-MiniLM-L6-v2",
            "description": "Fast, lightweight model (384 dimensions). Good balance of speed and quality.",
            "dimensions": 384,
            "max_seq_length": 256,
            "use_case": "General purpose, fast inference",
            "performance": "Good",
            "speed": "Very Fast",
            "size": "Small (~80MB)"
        },
        "all-mpnet-base-v2": {
            "name": "all-mpnet-base-v2",
            "description": "Higher quality model (768 dimensions). Better semantic understanding.",
            "dimensions": 768,
            "max_seq_length": 384,
            "use_case": "High quality semantic matching",
            "performance": "Excellent",
            "speed": "Fast",
            "size": "Medium (~420MB)"
        },
        "multi-qa-mpnet-base-dot-v1": {
            "name": "multi-qa-mpnet-base-dot-v1",
            "description": "Optimized for question-answer matching. Good for job-candidate matching.",
            "dimensions": 768,
            "max_seq_length": 512,
            "use_case": "Question-answer, matching tasks",
            "performance": "Excellent",
            "speed": "Fast",
            "size": "Medium (~420MB)"
        },
        "paraphrase-multilingual-mpnet-base-v2": {
            "name": "paraphrase-multilingual-mpnet-base-v2",
            "description": "Multilingual model supporting 50+ languages including Vietnamese (768 dimensions). Best for Vietnamese text.",
            "dimensions": 768,
            "max_seq_length": 128,
            "use_case": "Multilingual support, Vietnamese text",
            "performance": "Excellent",
            "speed": "Fast",
            "size": "Large (~900MB)"
        },
        "paraphrase-multilingual-MiniLM-L12-v2": {
            "name": "paraphrase-multilingual-MiniLM-L12-v2",
            "description": "Faster multilingual model supporting 50+ languages including Vietnamese (384 dimensions). Good balance for Vietnamese.",
            "dimensions": 384,
            "max_seq_length": 128,
            "use_case": "Multilingual support, Vietnamese text, fast inference",
            "performance": "Very Good",
            "speed": "Very Fast",
            "size": "Medium (~420MB)"
        },
        "keepitreal/vietnamese-sbert": {
            "name": "keepitreal/vietnamese-sbert",
            "description": "Vietnamese-specific SBERT model (768 dimensions). Optimized for Vietnamese text.",
            "dimensions": 768,
            "max_seq_length": 256,
            "use_case": "Vietnamese text only",
            "performance": "Excellent for Vietnamese",
            "speed": "Fast",
            "size": "Medium (~420MB)"
        },
        "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base": {
            "name": "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
            "description": "State-of-the-art Vietnamese SimCSE model based on PhoBERT (768 dimensions). Best for Vietnamese sentence similarity.",
            "dimensions": 768,
            "max_seq_length": 256,
            "use_case": "Vietnamese text, sentence similarity, best performance",
            "performance": "State-of-the-art for Vietnamese",
            "speed": "Fast",
            "size": "Medium (~520MB)",
            "requires_tokenization": True
        },
        "sentence-transformers/all-MiniLM-L6-v2": {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "description": "Same as all-MiniLM-L6-v2 with explicit library prefix.",
            "dimensions": 384,
            "max_seq_length": 256,
            "use_case": "General purpose",
            "performance": "Good",
            "speed": "Very Fast",
            "size": "Small (~80MB)"
        }
    }
    
    @classmethod
    def list_models(cls) -> List[Dict]:
        """List all available recommended models."""
        return list(cls.RECOMMENDED_MODELS.values())
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[Dict]:
        """Get information about a specific model."""
        # Try exact match first
        if model_name in cls.RECOMMENDED_MODELS:
            return cls.RECOMMENDED_MODELS[model_name]
        
        # Try without sentence-transformers prefix
        if model_name.startswith("sentence-transformers/"):
            base_name = model_name.replace("sentence-transformers/", "")
            if base_name in cls.RECOMMENDED_MODELS:
                return cls.RECOMMENDED_MODELS[base_name]
        
        # Try with prefix
        full_name = f"sentence-transformers/{model_name}"
        if full_name in cls.RECOMMENDED_MODELS:
            return cls.RECOMMENDED_MODELS[full_name]
        
        return None
    
    @classmethod
    def recommend_model(cls, use_case: str = "general") -> str:
        """
        Recommend a model based on use case.
        
        Args:
            use_case: "general", "quality", "multilingual", "fast"
        
        Returns:
            Recommended model name
        """
        recommendations = {
            "general": "all-MiniLM-L6-v2",
            "quality": "all-mpnet-base-v2",
            "matching": "multi-qa-mpnet-base-dot-v1",
            "multilingual": "paraphrase-multilingual-mpnet-base-v2",
            "vietnamese": "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
            "fast": "paraphrase-multilingual-MiniLM-L12-v2"
        }
        
        return recommendations.get(use_case.lower(), "all-MiniLM-L6-v2")
    
    @classmethod
    def print_model_options(cls):
        """Print available models in a formatted way."""
        print("\n" + "=" * 100)
        print("AVAILABLE EMBEDDING MODELS FOR JOB RECOMMENDATION")
        print("=" * 100)
        print()
        
        for idx, (key, model) in enumerate(cls.RECOMMENDED_MODELS.items(), 1):
            print(f"{idx}. {model['name']}")
            print(f"   Description: {model['description']}")
            print(f"   Dimensions: {model['dimensions']}")
            print(f"   Performance: {model['performance']} | Speed: {model['speed']} | Size: {model['size']}")
            print(f"   Use Case: {model['use_case']}")
            print()
        
        print("=" * 100)
        print("\nRecommendation:")
        print("  - For general use: all-MiniLM-L6-v2 (fast, good quality)")
        print("  - For best quality: all-mpnet-base-v2 (slower, better results)")
        print("  - For matching tasks: multi-qa-mpnet-base-dot-v1 (optimized for matching)")
        print("  - For multilingual: paraphrase-multilingual-mpnet-base-v2")
        print()

