"""Metrics Dashboard Service: Tracking và monitoring metrics."""
from typing import List, Dict, Optional
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class MetricsDashboardService:
    """
    Metrics Dashboard Service để track:
    - Accuracy metrics (precision@k, recall@k, MRR)
    - Latency metrics (p50, p95, p99)
    - User engagement (CTR, application rate, time to apply)
    - Model drift (embedding distribution, score distribution)
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize metrics dashboard service.
        
        Args:
            storage_path: Path để lưu metrics (optional, default: in-memory)
        """
        self.storage_path = storage_path
        self.metrics_store = defaultdict(list)
        logger.info("MetricsDashboardService initialized")
    
    def track_accuracy_metrics(
        self,
        recommendations: List[Dict],
        ground_truth: List[str],
        k: int = 10
    ) -> Dict:
        """
        Track accuracy metrics.
        
        Args:
            recommendations: List of recommended job IDs
            ground_truth: List of relevant job IDs (ground truth)
            k: Top K for precision/recall
            
        Returns:
            Accuracy metrics dict
        """
        if not recommendations or not ground_truth:
            return {}
        
        # Get top K recommended IDs
        recommended_ids = [r.get('job_id') for r in recommendations[:k]]
        recommended_ids = [rid for rid in recommended_ids if rid]
        
        # Calculate metrics
        relevant_recommended = set(recommended_ids) & set(ground_truth)
        
        precision_at_k = len(relevant_recommended) / len(recommended_ids) if recommended_ids else 0.0
        recall_at_k = len(relevant_recommended) / len(ground_truth) if ground_truth else 0.0
        
        # MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for i, rec_id in enumerate(recommended_ids, 1):
            if rec_id in ground_truth:
                mrr = 1.0 / i
                break
        
        metrics = {
            'precision_at_k': precision_at_k,
            'recall_at_k': recall_at_k,
            'mrr': mrr,
            'k': k,
            'timestamp': datetime.now().isoformat()
        }
        
        self.metrics_store['accuracy'].append(metrics)
        
        return metrics
    
    def track_latency(
        self,
        operation: str,
        latency_ms: float
    ):
        """
        Track latency metrics.
        
        Args:
            operation: Operation name (e.g., 'matching', 'reranking')
            latency_ms: Latency in milliseconds
        """
        metrics = {
            'operation': operation,
            'latency_ms': latency_ms,
            'timestamp': datetime.now().isoformat()
        }
        
        self.metrics_store['latency'].append(metrics)
    
    def get_latency_percentiles(
        self,
        operation: Optional[str] = None,
        time_window_hours: int = 24
    ) -> Dict:
        """
        Get latency percentiles.
        
        Args:
            operation: Operation name (None = all operations)
            time_window_hours: Time window for calculation
            
        Returns:
            Percentiles dict (p50, p95, p99)
        """
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        latencies = []
        for metric in self.metrics_store['latency']:
            if operation and metric['operation'] != operation:
                continue
            
            timestamp = datetime.fromisoformat(metric['timestamp'])
            if timestamp >= cutoff_time:
                latencies.append(metric['latency_ms'])
        
        if not latencies:
            return {'p50': 0.0, 'p95': 0.0, 'p99': 0.0}
        
        latencies = np.array(latencies)
        
        return {
            'p50': float(np.percentile(latencies, 50)),
            'p95': float(np.percentile(latencies, 95)),
            'p99': float(np.percentile(latencies, 99)),
            'mean': float(np.mean(latencies)),
            'count': len(latencies)
        }
    
    def track_user_engagement(
        self,
        candidate_id: str,
        job_id: str,
        action: str,
        metadata: Optional[Dict] = None
    ):
        """
        Track user engagement metrics.
        
        Args:
            candidate_id: Candidate ID
            job_id: Job ID
            action: Action type ('view', 'click', 'apply', etc.)
            metadata: Optional metadata
        """
        engagement = {
            'candidate_id': candidate_id,
            'job_id': job_id,
            'action': action,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.metrics_store['engagement'].append(engagement)
    
    def calculate_engagement_metrics(
        self,
        time_window_hours: int = 24
    ) -> Dict:
        """
        Calculate engagement metrics.
        
        Args:
            time_window_hours: Time window
            
        Returns:
            Engagement metrics dict
        """
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        views = 0
        clicks = 0
        applies = 0
        
        for engagement in self.metrics_store['engagement']:
            timestamp = datetime.fromisoformat(engagement['timestamp'])
            if timestamp >= cutoff_time:
                action = engagement['action']
                if action == 'view':
                    views += 1
                elif action == 'click':
                    clicks += 1
                elif action == 'apply':
                    applies += 1
        
        ctr = clicks / views if views > 0 else 0.0
        application_rate = applies / clicks if clicks > 0 else 0.0
        
        return {
            'views': views,
            'clicks': clicks,
            'applies': applies,
            'ctr': ctr,
            'application_rate': application_rate,
            'time_window_hours': time_window_hours
        }
    
    def track_model_drift(
        self,
        embedding_distribution: Dict[str, float],
        score_distribution: Dict[str, float]
    ):
        """
        Track model drift metrics.
        
        Args:
            embedding_distribution: Embedding distribution stats
            score_distribution: Score distribution stats
        """
        drift_metrics = {
            'embedding_distribution': embedding_distribution,
            'score_distribution': score_distribution,
            'timestamp': datetime.now().isoformat()
        }
        
        self.metrics_store['model_drift'].append(drift_metrics)
    
    def detect_drift(
        self,
        current_distribution: Dict[str, float],
        baseline_distribution: Dict[str, float],
        threshold: float = 0.1
    ) -> bool:
        """
        Detect model drift.
        
        Args:
            current_distribution: Current distribution stats
            baseline_distribution: Baseline distribution stats
            threshold: Drift threshold
            
        Returns:
            True if drift detected
        """
        # Simple drift detection: compare means
        current_mean = current_distribution.get('mean', 0.0)
        baseline_mean = baseline_distribution.get('mean', 0.0)
        
        if baseline_mean == 0:
            return False
        
        drift_ratio = abs(current_mean - baseline_mean) / abs(baseline_mean)
        
        return drift_ratio > threshold
    
    def get_all_metrics(
        self,
        time_window_hours: int = 24
    ) -> Dict:
        """
        Get all metrics summary.
        
        Args:
            time_window_hours: Time window
            
        Returns:
            All metrics dict
        """
        return {
            'accuracy': self.metrics_store.get('accuracy', [])[-10:],  # Last 10
            'latency': self.get_latency_percentiles(time_window_hours=time_window_hours),
            'engagement': self.calculate_engagement_metrics(time_window_hours=time_window_hours),
            'model_drift': self.metrics_store.get('model_drift', [])[-5:]  # Last 5
        }
    
    def save_metrics(self, filepath: Optional[str] = None):
        """Save metrics to file."""
        import json
        
        if filepath is None:
            filepath = self.storage_path or "metrics/metrics.json"
        
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_store, f, indent=2, default=str)
        
        logger.info(f"Metrics saved to {filepath}")
    
    def load_metrics(self, filepath: Optional[str] = None):
        """Load metrics from file."""
        import json
        
        if filepath is None:
            filepath = self.storage_path or "metrics/metrics.json"
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.metrics_store = json.load(f)
            logger.info(f"Metrics loaded from {filepath}")
        except FileNotFoundError:
            logger.warning(f"Metrics file not found: {filepath}")

