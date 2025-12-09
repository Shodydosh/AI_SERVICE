"""Utility function for loading SentenceTransformer models with fallback support."""
import logging
import warnings
from sentence_transformers import SentenceTransformer
from typing import Tuple

logger = logging.getLogger(__name__)


def load_embedding_model(
    preferred_model: str = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
    fallback_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
) -> Tuple[SentenceTransformer, str]:
    """
    Load SentenceTransformer model with fallback support.
    
    Args:
        preferred_model: Preferred model name (Vietnamese model)
        fallback_model: Fallback model name if preferred fails
        
    Returns:
        Tuple of (model, actual_model_name_used)
        
    Raises:
        RuntimeError: If both preferred and fallback models fail to load
    """
    # Try preferred model first
    try:
        logger.info(f"Attempting to load preferred model: {preferred_model}")
        
        # Capture warnings during model loading
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = SentenceTransformer(preferred_model)
            
            # Check captured warnings for the "Creating a new one" message
            for warning in w:
                warning_msg = str(warning.message)
                if "Creating a new one with mean pooling" in warning_msg:
                    raise RuntimeError(
                        f"Model '{preferred_model}' not found. "
                        "sentence-transformers tried to create empty model. "
                        "This would break Vietnamese tokenization. Using fallback instead."
                    )
        
        # Verify model is not empty (check if it has proper tokenizer)
        if not hasattr(model, 'tokenizer') or model.tokenizer is None:
            raise RuntimeError("Model loaded but tokenizer is None - model may be empty")
        
        logger.info(f"✓ Successfully loaded preferred model: {preferred_model}")
        print(f"Using embedding model: {preferred_model}")
        return model, preferred_model
        
    except RuntimeError as e:
        # Re-raise RuntimeError if it's about empty model
        error_msg = str(e)
        if "Creating a new one with mean pooling" in error_msg or "tried to create empty model" in error_msg:
            logger.error(f"Preferred model not found: {error_msg}")
            logger.warning(f"Falling back to: {fallback_model}")
        else:
            raise
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"Cannot load {preferred_model}. Error: {error_msg}")
        logger.warning(f"Falling back to: {fallback_model}")
        
    # Try fallback model
    try:
        logger.info(f"Attempting to load fallback model: {fallback_model}")
        
        # Capture warnings during fallback model loading
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = SentenceTransformer(fallback_model)
            
            # Check captured warnings for the "Creating a new one" message
            for warning in w:
                warning_msg = str(warning.message)
                if "Creating a new one with mean pooling" in warning_msg:
                    raise RuntimeError(
                        f"Fallback model '{fallback_model}' also not found. "
                        "sentence-transformers tried to create empty model. "
                        "Please ensure at least one model is properly installed."
                    )
        
        # Verify fallback model is not empty
        if not hasattr(model, 'tokenizer') or model.tokenizer is None:
            raise RuntimeError("Fallback model loaded but tokenizer is None - model may be empty")
        
        logger.warning(f"⚠️  Using fallback model: {fallback_model}")
        logger.warning(f"⚠️  Preferred model {preferred_model} was not available")
        print(f"[WARNING] Using fallback embedding model: {fallback_model}")
        print(f"[WARNING] Preferred model {preferred_model} was not available")
        return model, fallback_model
        
    except RuntimeError as e:
        # Re-raise RuntimeError if it's about empty model
        error_msg = str(e)
        if "Creating a new one with mean pooling" in error_msg or "tried to create empty model" in error_msg:
            raise RuntimeError(
                f"Both preferred ({preferred_model}) and fallback ({fallback_model}) "
                "models failed. sentence-transformers tried to create empty models. "
                "Please ensure at least one model is properly installed."
            )
        else:
            raise
    except Exception as fallback_error:
        error_msg = str(fallback_error)
        logger.error(f"Failed to load fallback model {fallback_model}: {error_msg}")
        raise RuntimeError(
            f"Both preferred ({preferred_model}) and fallback ({fallback_model}) "
            f"models failed to load. Last error: {error_msg}"
        )

