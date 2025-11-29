"""Generate 5 parameter variations for each model."""
from typing import List, Dict, Tuple
import logging
from .model_variations import ModelVariationBase

logger = logging.getLogger(__name__)


# 5 parameter variation configurations
PARAMETER_VARIATIONS = [
    {
        "name_suffix": "v1_bs32_norm",
        "batch_size": 32,
        "normalize": True,
        "use_tokenization": None,  # Will use model default
        "description": "Standard: batch_size=32, normalize=True"
    },
    {
        "name_suffix": "v2_bs64_norm",
        "batch_size": 64,
        "normalize": True,
        "use_tokenization": None,
        "description": "Large batch: batch_size=64, normalize=True"
    },
    {
        "name_suffix": "v3_bs128_norm",
        "batch_size": 128,
        "normalize": True,
        "use_tokenization": None,
        "description": "Very large batch: batch_size=128, normalize=True"
    },
    {
        "name_suffix": "v4_bs32_norm_false",
        "batch_size": 32,
        "normalize": False,
        "use_tokenization": None,
        "description": "No normalization: batch_size=32, normalize=False"
    },
    {
        "name_suffix": "v5_bs16_norm",
        "batch_size": 16,
        "normalize": True,
        "use_tokenization": None,
        "description": "Small batch: batch_size=16, normalize=True"
    }
]


# Base models to test
BASE_MODELS = [
    {
        "model_name": "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
        "base_name": "SimCSE_Vietnamese",
        "default_tokenization": True
    },
    {
        "model_name": "paraphrase-multilingual-mpnet-base-v2",
        "base_name": "Multilingual_MPNet",
        "default_tokenization": False
    },
    {
        "model_name": "keepitreal/vietnamese-sbert",
        "base_name": "Vietnamese_SBERT",
        "default_tokenization": False
    },
    {
        "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
        "base_name": "MiniLM_Multilingual",
        "default_tokenization": False
    },
    {
        "model_name": "all-mpnet-base-v2",
        "base_name": "MPNet_Base",
        "default_tokenization": False
    },
    {
        "model_name": "multi-qa-mpnet-base-dot-v1",
        "base_name": "QA_MPNet",
        "default_tokenization": False
    },
    {
        "model_name": "all-MiniLM-L6-v2",
        "base_name": "MiniLM_L6",
        "default_tokenization": False
    },
    {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "base_name": "MPNet_ST",
        "default_tokenization": False
    },
    {
        "model_name": "paraphrase-MiniLM-L6-v2",
        "base_name": "Paraphrase_MiniLM",
        "default_tokenization": False
    },
    {
        "model_name": "distiluse-base-multilingual-cased",
        "base_name": "DistilUSE_Multilingual",
        "default_tokenization": False
    }
]


class ParameterVariation(ModelVariationBase):
    """A model variation with specific parameters."""
    
    def __init__(self, variation_id: int, model_name: str, base_name: str, 
                 param_config: Dict, default_tokenization: bool):
        # Build full name
        full_name = f"{base_name}_{param_config['name_suffix']}"
        
        # Determine tokenization
        use_tokenization = param_config.get('use_tokenization')
        if use_tokenization is None:
            use_tokenization = default_tokenization
        
        super().__init__(
            variation_id=variation_id,
            name=full_name,
            model_name=model_name,
            batch_size=param_config['batch_size'],
            normalize=param_config['normalize'],
            use_tokenization=use_tokenization
        )
        self.param_config = param_config
        self.base_name = base_name


def generate_all_variations() -> List[ParameterVariation]:
    """Generate all parameter variations for all base models."""
    variations = []
    variation_id = 1
    
    for base_model in BASE_MODELS:
        model_name = base_model["model_name"]
        base_name = base_model["base_name"]
        default_tokenization = base_model["default_tokenization"]
        
        for param_config in PARAMETER_VARIATIONS:
            try:
                variation = ParameterVariation(
                    variation_id=variation_id,
                    model_name=model_name,
                    base_name=base_name,
                    param_config=param_config,
                    default_tokenization=default_tokenization
                )
                variations.append(variation)
                variation_id += 1
            except Exception as e:
                logger.error(f"Failed to create variation {variation_id} for {base_name}: {e}")
                continue
    
    logger.info(f"Generated {len(variations)} total variations ({len(BASE_MODELS)} models × {len(PARAMETER_VARIATIONS)} params)")
    return variations


def get_variation_by_id(variation_id: int) -> ParameterVariation:
    """Get a specific variation by ID."""
    all_variations = generate_all_variations()
    if variation_id < 1 or variation_id > len(all_variations):
        raise ValueError(f"Variation ID {variation_id} out of range (1-{len(all_variations)})")
    return all_variations[variation_id - 1]


def list_all_variations() -> List[Dict]:
    """List all variations with metadata."""
    variations = generate_all_variations()
    return [
        {
            "id": var.variation_id,
            "name": var.name,
            "model_name": var.model_name,
            "base_name": var.base_name,
            "batch_size": var.batch_size,
            "normalize": var.normalize,
            "use_tokenization": var.use_tokenization,
            "dimension": var.dimension,
            "description": var.param_config.get("description", "")
        }
        for var in variations
    ]


def get_variations_by_model(base_name: str) -> List[ParameterVariation]:
    """Get all variations for a specific base model."""
    all_variations = generate_all_variations()
    return [v for v in all_variations if v.base_name == base_name]

