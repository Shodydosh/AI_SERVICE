"""A/B Testing Service: Feature flags cho experimentation."""
from typing import Dict, Optional, Callable, Any
import logging
import hashlib
import random
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class ABTestingService:
    """
    A/B Testing Service với feature flags.
    
    Features:
    - Experiment groups (control/variant)
    - Consistent assignment (based on candidate_id hash)
    - Metrics tracking per experiment
    """
    
    def __init__(
        self,
        experiments: Optional[Dict[str, Dict]] = None
    ):
        """
        Initialize A/B testing service.
        
        Args:
            experiments: Dict of experiment_name -> experiment_config
        """
        self.experiments = experiments or {}
        logger.info(f"ABTestingService initialized with {len(self.experiments)} experiments")
    
    def register_experiment(
        self,
        experiment_name: str,
        control_func: Callable,
        variant_func: Callable,
        split_ratio: float = 0.5,
        enabled: bool = True
    ):
        """
        Register an A/B test experiment.
        
        Args:
            experiment_name: Experiment name
            control_func: Control group function
            variant_func: Variant group function
            split_ratio: Ratio for variant group (0.5 = 50%)
            enabled: Whether experiment is enabled
        """
        self.experiments[experiment_name] = {
            'control_func': control_func,
            'variant_func': variant_func,
            'split_ratio': split_ratio,
            'enabled': enabled,
            'metrics': {
                'control_calls': 0,
                'variant_calls': 0,
                'control_results': [],
                'variant_results': []
            }
        }
        
        logger.info(f"Registered experiment: {experiment_name} (split_ratio={split_ratio})")
    
    def get_experiment_group(
        self,
        experiment_name: str,
        user_id: str
    ) -> str:
        """
        Get experiment group for user (consistent assignment).
        
        Args:
            experiment_name: Experiment name
            user_id: User/Candidate ID
            
        Returns:
            'control' or 'variant'
        """
        if experiment_name not in self.experiments:
            return 'control'
        
        experiment = self.experiments[experiment_name]
        if not experiment['enabled']:
            return 'control'
        
        # Consistent assignment based on hash
        hash_value = int(hashlib.md5(f"{experiment_name}:{user_id}".encode()).hexdigest(), 16)
        assignment = hash_value % 100
        
        split_ratio = experiment['split_ratio']
        threshold = int(split_ratio * 100)
        
        if assignment < threshold:
            return 'variant'
        else:
            return 'control'
    
    def run_experiment(
        self,
        experiment_name: str,
        user_id: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Run experiment và return result.
        
        Args:
            experiment_name: Experiment name
            user_id: User/Candidate ID
            *args: Arguments for experiment functions
            **kwargs: Keyword arguments for experiment functions
            
        Returns:
            Result from control or variant function
        """
        if experiment_name not in self.experiments:
            logger.warning(f"Experiment {experiment_name} not found, using control")
            return None
        
        experiment = self.experiments[experiment_name]
        if not experiment['enabled']:
            return experiment['control_func'](*args, **kwargs)
        
        # Get experiment group
        group = self.get_experiment_group(experiment_name, user_id)
        
        # Run appropriate function
        if group == 'variant':
            result = experiment['variant_func'](*args, **kwargs)
            experiment['metrics']['variant_calls'] += 1
            experiment['metrics']['variant_results'].append({
                'user_id': user_id,
                'result_count': len(result) if isinstance(result, list) else 1,
                'timestamp': datetime.now().isoformat()
            })
        else:
            result = experiment['control_func'](*args, **kwargs)
            experiment['metrics']['control_calls'] += 1
            experiment['metrics']['control_results'].append({
                'user_id': user_id,
                'result_count': len(result) if isinstance(result, list) else 1,
                'timestamp': datetime.now().isoformat()
            })
        
        logger.debug(f"Experiment {experiment_name}: {group} group for user {user_id}")
        
        return result
    
    def get_experiment_metrics(
        self,
        experiment_name: str
    ) -> Dict:
        """
        Get experiment metrics.
        
        Args:
            experiment_name: Experiment name
            
        Returns:
            Metrics dict
        """
        if experiment_name not in self.experiments:
            return {}
        
        experiment = self.experiments[experiment_name]
        metrics = experiment['metrics']
        
        # Calculate statistics
        control_results = metrics['control_results']
        variant_results = metrics['variant_results']
        
        control_avg_results = np.mean([r['result_count'] for r in control_results]) if control_results else 0.0
        variant_avg_results = np.mean([r['result_count'] for r in variant_results]) if variant_results else 0.0
        
        return {
            'experiment_name': experiment_name,
            'control_calls': metrics['control_calls'],
            'variant_calls': metrics['variant_calls'],
            'control_avg_results': control_avg_results,
            'variant_avg_results': variant_avg_results,
            'improvement': (variant_avg_results - control_avg_results) / control_avg_results if control_avg_results > 0 else 0.0,
            'enabled': experiment['enabled']
        }
    
    def get_all_experiment_metrics(self) -> Dict:
        """Get metrics for all experiments."""
        return {
            exp_name: self.get_experiment_metrics(exp_name)
            for exp_name in self.experiments.keys()
        }
    
    def enable_experiment(self, experiment_name: str):
        """Enable an experiment."""
        if experiment_name in self.experiments:
            self.experiments[experiment_name]['enabled'] = True
            logger.info(f"Enabled experiment: {experiment_name}")
    
    def disable_experiment(self, experiment_name: str):
        """Disable an experiment."""
        if experiment_name in self.experiments:
            self.experiments[experiment_name]['enabled'] = False
            logger.info(f"Disabled experiment: {experiment_name}")


# Decorator for A/B testing
def ab_test(
    experiment_name: str,
    control_func: Callable,
    variant_func: Callable,
    split_ratio: float = 0.5
):
    """
    Decorator for A/B testing.
    
    Usage:
        @ab_test('hybrid_search', current_pipeline, new_hybrid_pipeline)
        def recommend_jobs(candidate_id, experiment_group):
            if experiment_group == 'variant':
                return new_hybrid_pipeline(candidate_id)
            return current_pipeline(candidate_id)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get user_id from args or kwargs
            user_id = kwargs.get('user_id') or kwargs.get('candidate_id') or (args[0] if args else None)
            
            if not user_id:
                # Fallback to control
                return control_func(*args, **kwargs)
            
            # Get experiment group
            ab_service = ABTestingService()
            group = ab_service.get_experiment_group(experiment_name, str(user_id))
            
            # Run appropriate function
            if group == 'variant':
                return variant_func(*args, **kwargs)
            else:
                return control_func(*args, **kwargs)
        
        return wrapper
    return decorator

