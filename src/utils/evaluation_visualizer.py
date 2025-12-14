"""Visualization utilities for embedding evaluation metrics."""
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Try to import matplotlib and seaborn
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("Matplotlib/Seaborn not available. Plotting will be disabled.")


class EvaluationVisualizer:
    """Visualizer for embedding evaluation metrics."""
    
    def __init__(self, output_dir: str = "visualizations/evaluation", style: str = "seaborn-v0_8"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
            style: Matplotlib style
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if PLOTTING_AVAILABLE:
            plt.style.use(style)
            sns.set_palette("husl")
    
    def plot_clustering_metrics(
        self,
        metrics_history: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Plot clustering metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz) over k values.
        
        Args:
            metrics_history: Dictionary with clustering metrics
            save_path: Optional path to save plot
        
        Returns:
            Path to saved plot or None if plotting unavailable
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available. Skipping clustering metrics plot.")
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Clustering Metrics Evaluation', fontsize=16, fontweight='bold')
        
        # Extract data
        k_values = []
        silhouette_scores = []
        davies_bouldin_scores = []
        calinski_harabasz_scores = []
        
        # Handle different input formats
        if 'silhouette_scores' in metrics_history:
            silhouette_dict = metrics_history['silhouette_scores']
            davies_bouldin_dict = metrics_history.get('davies_bouldin_scores', {})
            calinski_harabasz_dict = metrics_history.get('calinski_harabasz_scores', {})
            
            k_values = sorted(silhouette_dict.keys())
            silhouette_scores = [silhouette_dict[k] for k in k_values]
            davies_bouldin_scores = [davies_bouldin_dict.get(k, 0) for k in k_values]
            calinski_harabasz_scores = [calinski_harabasz_dict.get(k, 0) for k in k_values]
        
        # Plot Silhouette Score
        axes[0].plot(k_values, silhouette_scores, marker='o', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[0].set_ylabel('Silhouette Score', fontsize=12)
        axes[0].set_title('Silhouette Score', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([-1, 1])
        
        # Plot Davies-Bouldin Index (lower is better)
        axes[1].plot(k_values, davies_bouldin_scores, marker='s', linewidth=2, markersize=8, color='orange')
        axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[1].set_ylabel('Davies-Bouldin Index', fontsize=12)
        axes[1].set_title('Davies-Bouldin Index (Lower is Better)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Plot Calinski-Harabasz Score
        axes[2].plot(k_values, calinski_harabasz_scores, marker='^', linewidth=2, markersize=8, color='green')
        axes[2].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[2].set_ylabel('Calinski-Harabasz Score', fontsize=12)
        axes[2].set_title('Calinski-Harabasz Score', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "clustering_scores.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Clustering metrics plot saved to: {save_path}")
        return str(save_path)
    
    def plot_adversarial_results(
        self,
        test_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Plot adversarial test results (robustness scores).
        
        Args:
            test_results: Dictionary with adversarial test results
            save_path: Optional path to save plot
        
        Returns:
            Path to saved plot or None if plotting unavailable
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available. Skipping adversarial results plot.")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Adversarial Testing Results', fontsize=16, fontweight='bold')
        
        # Extract data
        test_names = []
        robustness_scores = []
        similarity_scores = []
        
        for test_name, results in test_results.items():
            if isinstance(results, dict):
                test_names.append(test_name.replace('_', ' ').title())
                robustness_scores.append(results.get('robustness_score', 0.0))
                similarity_scores.append(results.get('similarity_score', 0.0))
        
        # Plot 1: Robustness Scores (Bar Chart)
        axes[0, 0].barh(test_names, robustness_scores, color='steelblue', alpha=0.7)
        axes[0, 0].set_xlabel('Robustness Score', fontsize=12)
        axes[0, 0].set_title('Robustness Scores by Test Type', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlim([0, 1.1])
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # Plot 2: Similarity Scores (Bar Chart)
        axes[0, 1].barh(test_names, similarity_scores, color='coral', alpha=0.7)
        axes[0, 1].set_xlabel('Similarity Score', fontsize=12)
        axes[0, 1].set_title('Similarity Scores by Test Type', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlim([0, 1.1])
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # Plot 3: Robustness vs Similarity (Scatter)
        axes[1, 0].scatter(similarity_scores, robustness_scores, s=100, alpha=0.6, c='purple')
        axes[1, 0].set_xlabel('Similarity Score', fontsize=12)
        axes[1, 0].set_ylabel('Robustness Score', fontsize=12)
        axes[1, 0].set_title('Robustness vs Similarity', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim([0, 1.1])
        axes[1, 0].set_ylim([0, 1.1])
        
        # Add diagonal line
        axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
        axes[1, 0].legend()
        
        # Plot 4: Overall Statistics
        if test_results.get('overall_robustness'):
            stats = test_results.get('overall_robustness', {})
            stats_data = {
                'Overall Robustness': stats.get('overall_robustness', 0.0),
                'Avg Similarity': stats.get('avg_similarity', 0.0),
                'Min Similarity': stats.get('min_similarity', 0.0),
                'Max Similarity': stats.get('max_similarity', 0.0)
            }
            
            axes[1, 1].bar(stats_data.keys(), stats_data.values(), color='teal', alpha=0.7)
            axes[1, 1].set_ylabel('Score', fontsize=12)
            axes[1, 1].set_title('Overall Statistics', fontsize=14, fontweight='bold')
            axes[1, 1].set_ylim([0, 1.1])
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        else:
            axes[1, 1].text(0.5, 0.5, 'No overall statistics available', 
                          ha='center', va='center', fontsize=12)
            axes[1, 1].set_title('Overall Statistics', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "adversarial_robustness.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Adversarial results plot saved to: {save_path}")
        return str(save_path)
    
    def plot_semantic_preservation(
        self,
        preservation_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Plot semantic preservation correlation results.
        
        Args:
            preservation_results: Dictionary with semantic preservation metrics
            save_path: Optional path to save plot
        
        Returns:
            Path to saved plot or None if plotting unavailable
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available. Skipping semantic preservation plot.")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Semantic Preservation Analysis', fontsize=16, fontweight='bold')
        
        # Extract data
        similarities = preservation_results.get('similarities', [])
        correlation = preservation_results.get('correlation', 0.0)
        avg_similarity = preservation_results.get('avg_similarity', 0.0)
        
        # Plot 1: Similarity Distribution (Histogram)
        if similarities:
            axes[0].hist(similarities, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
            axes[0].axvline(avg_similarity, color='red', linestyle='--', linewidth=2, 
                           label=f'Mean: {avg_similarity:.3f}')
            axes[0].set_xlabel('Similarity Score', fontsize=12)
            axes[0].set_ylabel('Frequency', fontsize=12)
            axes[0].set_title('Similarity Score Distribution', fontsize=14, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3, axis='y')
        else:
            axes[0].text(0.5, 0.5, 'No similarity data available', 
                        ha='center', va='center', fontsize=12)
            axes[0].set_title('Similarity Score Distribution', fontsize=14, fontweight='bold')
        
        # Plot 2: Correlation Metrics (Bar Chart)
        metrics_data = {
            'Correlation': correlation,
            'Avg Similarity': avg_similarity,
            'Min Similarity': preservation_results.get('min_similarity', 0.0),
            'Max Similarity': preservation_results.get('max_similarity', 0.0)
        }
        
        axes[1].bar(metrics_data.keys(), metrics_data.values(), color='lightcoral', alpha=0.7)
        axes[1].set_ylabel('Score', fontsize=12)
        axes[1].set_title('Semantic Preservation Metrics', fontsize=14, fontweight='bold')
        axes[1].set_ylim([0, 1.1])
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "semantic_preservation.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Semantic preservation plot saved to: {save_path}")
        return str(save_path)
    
    def plot_metrics_tracking(
        self,
        metrics_over_time: Dict[str, List[float]],
        timestamps: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Plot metrics tracking over time.
        
        Args:
            metrics_over_time: Dictionary with metric names as keys and lists of values
            timestamps: Optional list of timestamps for x-axis
            save_path: Optional path to save plot
        
        Returns:
            Path to saved plot or None if plotting unavailable
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available. Skipping metrics tracking plot.")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Metrics Tracking Over Time', fontsize=16, fontweight='bold')
        
        # Prepare x-axis
        if timestamps:
            x_values = timestamps
        else:
            x_values = list(range(len(list(metrics_over_time.values())[0]) if metrics_over_time else 0)))
        
        # Plot each metric
        plot_idx = 0
        colors = ['steelblue', 'coral', 'green', 'purple', 'orange', 'teal']
        
        for metric_name, values in metrics_over_time.items():
            if plot_idx >= 4:  # Only plot first 4 metrics
                break
            
            row = plot_idx // 2
            col = plot_idx % 2
            
            axes[row, col].plot(x_values, values, marker='o', linewidth=2, 
                               markersize=6, color=colors[plot_idx % len(colors)], label=metric_name)
            axes[row, col].set_xlabel('Time', fontsize=12)
            axes[row, col].set_ylabel('Score', fontsize=12)
            axes[row, col].set_title(metric_name.replace('_', ' ').title(), fontsize=14, fontweight='bold')
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].legend()
            
            # Rotate x-axis labels if timestamps
            if timestamps:
                axes[row, col].tick_params(axis='x', rotation=45)
            
            plot_idx += 1
        
        # Hide unused subplots
        for idx in range(plot_idx, 4):
            row = idx // 2
            col = idx % 2
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "metrics_tracking.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Metrics tracking plot saved to: {save_path}")
        return str(save_path)
    
    def plot_comprehensive_evaluation(
        self,
        clustering_results: Dict[str, Any],
        adversarial_results: Dict[str, Any],
        preservation_results: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Create a comprehensive evaluation plot combining all metrics.
        
        Args:
            clustering_results: Clustering evaluation results
            adversarial_results: Adversarial test results
            preservation_results: Optional semantic preservation results
            save_path: Optional path to save plot
        
        Returns:
            Path to saved plot or None if plotting unavailable
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available. Skipping comprehensive evaluation plot.")
            return None
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Comprehensive Embedding Evaluation', fontsize=18, fontweight='bold', y=0.98)
        
        # Plot 1: Clustering Metrics (top left, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        if 'silhouette_scores' in clustering_results:
            k_values = sorted(clustering_results['silhouette_scores'].keys())
            silhouette = [clustering_results['silhouette_scores'][k] for k in k_values]
            ax1.plot(k_values, silhouette, marker='o', linewidth=2, label='Silhouette Score')
            ax1.set_xlabel('Number of Clusters (k)', fontsize=11)
            ax1.set_ylabel('Score', fontsize=11)
            ax1.set_title('Clustering Metrics', fontsize=13, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Robustness Scores (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        if adversarial_results:
            test_names = list(adversarial_results.keys())[:5]  # Top 5
            robustness = [adversarial_results.get(name, {}).get('robustness_score', 0) 
                         for name in test_names]
            ax2.barh(test_names, robustness, color='steelblue', alpha=0.7)
            ax2.set_xlabel('Robustness Score', fontsize=11)
            ax2.set_title('Adversarial Robustness', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
        
        # Plot 3: Semantic Preservation (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        if preservation_results:
            metrics = ['correlation', 'avg_similarity']
            values = [preservation_results.get(m, 0) for m in metrics]
            ax3.bar(metrics, values, color='coral', alpha=0.7)
            ax3.set_ylabel('Score', fontsize=11)
            ax3.set_title('Semantic Preservation', fontsize=13, fontweight='bold')
            ax3.set_ylim([0, 1.1])
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Similarity Distribution (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        if preservation_results and 'similarities' in preservation_results:
            similarities = preservation_results['similarities']
            ax4.hist(similarities, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Similarity Score', fontsize=11)
            ax4.set_ylabel('Frequency', fontsize=11)
            ax4.set_title('Similarity Distribution', fontsize=13, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Overall Summary (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        summary_metrics = {
            'Clustering\nQuality': clustering_results.get('best_silhouette', 0),
            'Robustness': adversarial_results.get('overall_robustness', {}).get('overall_robustness', 0) 
                         if isinstance(adversarial_results.get('overall_robustness'), dict) 
                         else adversarial_results.get('overall_robustness', 0),
            'Semantic\nPreservation': preservation_results.get('correlation', 0) if preservation_results else 0
        }
        ax5.bar(summary_metrics.keys(), summary_metrics.values(), color='teal', alpha=0.7)
        ax5.set_ylabel('Score', fontsize=11)
        ax5.set_title('Overall Summary', fontsize=13, fontweight='bold')
        ax5.set_ylim([0, 1.1])
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Plot 6: Metrics Comparison (bottom, spans all columns)
        ax6 = fig.add_subplot(gs[2, :])
        if clustering_results and adversarial_results:
            comparison_data = {
                'Best Silhouette': clustering_results.get('best_silhouette', 0),
                'Avg Robustness': adversarial_results.get('overall_robustness', {}).get('overall_robustness', 0) 
                                 if isinstance(adversarial_results.get('overall_robustness'), dict)
                                 else adversarial_results.get('overall_robustness', 0),
                'Semantic Correlation': preservation_results.get('correlation', 0) if preservation_results else 0
            }
            ax6.bar(comparison_data.keys(), comparison_data.values(), color='purple', alpha=0.7)
            ax6.set_ylabel('Score', fontsize=11)
            ax6.set_title('Metrics Comparison', fontsize=13, fontweight='bold')
            ax6.set_ylim([0, 1.1])
            ax6.grid(True, alpha=0.3, axis='y')
        
        if save_path is None:
            save_path = self.output_dir / "comprehensive_evaluation.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comprehensive evaluation plot saved to: {save_path}")
        return str(save_path)





