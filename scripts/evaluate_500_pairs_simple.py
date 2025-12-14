"""Simple evaluation script for 500 pairs - no model loading needed."""
import sys
import os
import argparse
import logging
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_ground_truth_csv(csv_path: str) -> pd.DataFrame:
    """Load ground truth pairs from CSV."""
    logger.info(f"Loading ground truth from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    logger.info(f"Loaded {len(df)} pairs")
    logger.info(f"  - High: {len(df[df['similarity_type'] == 'high'])}")
    logger.info(f"  - Medium: {len(df[df['similarity_type'] == 'medium'])}")
    logger.info(f"  - Random: {len(df[df['similarity_type'] == 'random'])}")
    
    return df


def calculate_metrics(df: pd.DataFrame) -> Dict:
    """Calculate evaluation metrics from ground truth."""
    logger.info("\nCalculating metrics...")
    
    # Convert similarity_type to labels
    df['label'] = df['similarity_type'].map({
        'high': 1.0,
        'medium': 0.7,
        'random': 0.0
    })
    
    predictions = df['predicted_similarity'].values
    labels = df['label'].values
    
    # Binary classification (threshold = 0.5)
    binary_preds = (predictions >= 0.5).astype(int)
    binary_labels = (labels >= 0.5).astype(int)
    
    # Confusion matrix
    tp = np.sum((binary_preds == 1) & (binary_labels == 1))
    fp = np.sum((binary_preds == 1) & (binary_labels == 0))
    tn = np.sum((binary_preds == 0) & (binary_labels == 0))
    fn = np.sum((binary_preds == 0) & (binary_labels == 1))
    
    # Classification metrics
    accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # AUC-ROC (simplified - using predicted_similarity as score)
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc_roc = roc_auc_score(binary_labels, predictions)
        auc_pr = average_precision_score(binary_labels, predictions)
    except:
        auc_roc = 0.0
        auc_pr = 0.0
    
    # Analysis by similarity type
    high_df = df[df['similarity_type'] == 'high']
    medium_df = df[df['similarity_type'] == 'medium']
    random_df = df[df['similarity_type'] == 'random']
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'confusion_matrix': {
            'true_positive': int(tp),
            'false_positive': int(fp),
            'true_negative': int(tn),
            'false_negative': int(fn)
        },
        'high_similarity': {
            'count': len(high_df),
            'mean_score': float(high_df['predicted_similarity'].mean()) if len(high_df) > 0 else 0.0,
            'std_score': float(high_df['predicted_similarity'].std()) if len(high_df) > 0 else 0.0,
            'min_score': float(high_df['predicted_similarity'].min()) if len(high_df) > 0 else 0.0,
            'max_score': float(high_df['predicted_similarity'].max()) if len(high_df) > 0 else 0.0
        },
        'medium_similarity': {
            'count': len(medium_df),
            'mean_score': float(medium_df['predicted_similarity'].mean()) if len(medium_df) > 0 else 0.0,
            'std_score': float(medium_df['predicted_similarity'].std()) if len(medium_df) > 0 else 0.0,
            'min_score': float(medium_df['predicted_similarity'].min()) if len(medium_df) > 0 else 0.0,
            'max_score': float(medium_df['predicted_similarity'].max()) if len(medium_df) > 0 else 0.0
        },
        'random_similarity': {
            'count': len(random_df),
            'mean_score': float(random_df['predicted_similarity'].mean()) if len(random_df) > 0 else 0.0,
            'std_score': float(random_df['predicted_similarity'].std()) if len(random_df) > 0 else 0.0,
            'min_score': float(random_df['predicted_similarity'].min()) if len(random_df) > 0 else 0.0,
            'max_score': float(random_df['predicted_similarity'].max()) if len(random_df) > 0 else 0.0
        },
        'score_distribution': {
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'median': float(np.median(predictions)),
            'q25': float(np.percentile(predictions, 25)),
            'q75': float(np.percentile(predictions, 75))
        },
        'label_correlation': float(np.corrcoef(predictions, labels)[0, 1]) if len(np.unique(labels)) > 1 else 0.0
    }
    
    return metrics, predictions, labels


def generate_html_report(
    metrics: Dict,
    predictions: np.ndarray,
    labels: np.ndarray,
    output_path: str
):
    """Generate beautiful HTML report."""
    logger.info(f"\nGenerating HTML report: {output_path}")
    
    total_pairs = len(predictions)
    high_count = metrics.get('high_similarity', {}).get('count', 0)
    medium_count = metrics.get('medium_similarity', {}).get('count', 0)
    random_count = metrics.get('random_similarity', {}).get('count', 0)
    
    # Overall assessment
    auc_roc = metrics.get('auc_roc', 0)
    
    if auc_roc > 0.8:
        overall_status = "EXCELLENT"
        status_color = "#28a745"
        status_icon = "✅"
    elif auc_roc > 0.7:
        overall_status = "GOOD"
        status_color = "#17a2b8"
        status_icon = "✓"
    elif auc_roc > 0.6:
        overall_status = "MODERATE"
        status_color = "#ffc107"
        status_icon = "⚠"
    else:
        overall_status = "NEEDS IMPROVEMENT"
        status_color = "#dc3545"
        status_icon = "❌"
    
    # Confusion matrix
    cm = metrics.get('confusion_matrix', {})
    tp = cm.get('true_positive', 0)
    fp = cm.get('false_positive', 0)
    tn = cm.get('true_negative', 0)
    fn = cm.get('false_negative', 0)
    
    # HTML template (same as before, but simplified)
    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo Cáo Đánh Giá 500 Pairs</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 15px 30px;
            background: {status_color};
            color: white;
            border-radius: 50px;
            font-size: 1.3em;
            font-weight: bold;
            margin: 20px 0;
        }}
        
        .section {{
            margin: 40px 0;
            padding: 30px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 5px solid #667eea;
        }}
        
        .section h2 {{
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .metric-card .label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 10px;
        }}
        
        .metric-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .metric-card .badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.8em;
            margin-top: 10px;
        }}
        
        .badge-excellent {{ background: #28a745; color: white; }}
        .badge-good {{ background: #17a2b8; color: white; }}
        .badge-moderate {{ background: #ffc107; color: #333; }}
        .badge-poor {{ background: #dc3545; color: white; }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 10px;
            overflow: hidden;
        }}
        
        table th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
        }}
        
        table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}
        
        table tr:hover {{
            background: #f8f9fa;
        }}
        
        .confusion-matrix {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            max-width: 500px;
            margin: 20px auto;
        }}
        
        .cm-cell {{
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: white;
            font-weight: bold;
            font-size: 1.2em;
        }}
        
        .cm-tp {{ background: #28a745; }}
        .cm-fp {{ background: #ffc107; }}
        .cm-tn {{ background: #17a2b8; }}
        .cm-fn {{ background: #dc3545; }}
        
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Báo Cáo Đánh Giá 500 Pairs</h1>
            <div class="subtitle">Ground Truth Evaluation Report</div>
            <div class="subtitle" style="margin-top: 10px; font-size: 1em;">
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
        
        <div class="content">
            <div style="text-align: center;">
                <div class="status-badge">
                    {status_icon} {overall_status}
                </div>
            </div>
            
            <!-- Overall Metrics -->
            <div class="section">
                <h2>📈 Tổng Quan Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="label">AUC-ROC</div>
                        <div class="value">{metrics.get('auc_roc', 0):.4f}</div>
                        <span class="badge {'badge-excellent' if metrics.get('auc_roc', 0) > 0.8 else 'badge-good' if metrics.get('auc_roc', 0) > 0.7 else 'badge-moderate' if metrics.get('auc_roc', 0) > 0.6 else 'badge-poor'}">
                            {'Excellent' if metrics.get('auc_roc', 0) > 0.8 else 'Good' if metrics.get('auc_roc', 0) > 0.7 else 'Moderate' if metrics.get('auc_roc', 0) > 0.6 else 'Poor'}
                        </span>
                    </div>
                    <div class="metric-card">
                        <div class="label">Precision</div>
                        <div class="value">{metrics.get('precision', 0):.4f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Recall</div>
                        <div class="value">{metrics.get('recall', 0):.4f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">F1-Score</div>
                        <div class="value">{metrics.get('f1', 0):.4f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Accuracy</div>
                        <div class="value">{metrics.get('accuracy', 0):.4f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">AUC-PR</div>
                        <div class="value">{metrics.get('auc_pr', 0):.4f}</div>
                    </div>
                </div>
            </div>
            
            <!-- Confusion Matrix -->
            <div class="section">
                <h2>🔍 Confusion Matrix</h2>
                <div class="confusion-matrix">
                    <div class="cm-cell cm-tp">
                        <div>True Positive</div>
                        <div style="font-size: 1.5em; margin-top: 10px;">{tp}</div>
                    </div>
                    <div class="cm-cell cm-fp">
                        <div>False Positive</div>
                        <div style="font-size: 1.5em; margin-top: 10px;">{fp}</div>
                    </div>
                    <div class="cm-cell cm-tn">
                        <div>True Negative</div>
                        <div style="font-size: 1.5em; margin-top: 10px;">{tn}</div>
                    </div>
                    <div class="cm-cell cm-fn">
                        <div>False Negative</div>
                        <div style="font-size: 1.5em; margin-top: 10px;">{fn}</div>
                    </div>
                </div>
                <div style="text-align: center; margin-top: 20px;">
                    <p><strong>Precision:</strong> {tp / (tp + fp) if (tp + fp) > 0 else 0:.4f}</p>
                    <p><strong>Recall:</strong> {tp / (tp + fn) if (tp + fn) > 0 else 0:.4f}</p>
                    <p><strong>Specificity:</strong> {tn / (tn + fp) if (tn + fp) > 0 else 0:.4f}</p>
                </div>
            </div>
            
            <!-- Analysis by Similarity Type -->
            <div class="section">
                <h2>📊 Phân Tích Theo Loại Similarity</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Loại</th>
                            <th>Số lượng</th>
                            <th>Mean Score</th>
                            <th>Std</th>
                            <th>Min</th>
                            <th>Max</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>High Similarity</strong></td>
                            <td>{high_count}</td>
                            <td>{metrics.get('high_similarity', {}).get('mean_score', 0):.4f}</td>
                            <td>{metrics.get('high_similarity', {}).get('std_score', 0):.4f}</td>
                            <td>{metrics.get('high_similarity', {}).get('min_score', 0):.4f}</td>
                            <td>{metrics.get('high_similarity', {}).get('max_score', 0):.4f}</td>
                        </tr>
                        <tr>
                            <td><strong>Medium Similarity</strong></td>
                            <td>{medium_count}</td>
                            <td>{metrics.get('medium_similarity', {}).get('mean_score', 0):.4f}</td>
                            <td>{metrics.get('medium_similarity', {}).get('std_score', 0):.4f}</td>
                            <td>{metrics.get('medium_similarity', {}).get('min_score', 0):.4f}</td>
                            <td>{metrics.get('medium_similarity', {}).get('max_score', 0):.4f}</td>
                        </tr>
                        <tr>
                            <td><strong>Random/Low Similarity</strong></td>
                            <td>{random_count}</td>
                            <td>{metrics.get('random_similarity', {}).get('mean_score', 0):.4f}</td>
                            <td>{metrics.get('random_similarity', {}).get('std_score', 0):.4f}</td>
                            <td>{metrics.get('random_similarity', {}).get('min_score', 0):.4f}</td>
                            <td>{metrics.get('random_similarity', {}).get('max_score', 0):.4f}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <!-- Score Distribution -->
            <div class="section">
                <h2>📉 Phân Bố Score</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="label">Mean</div>
                        <div class="value">{metrics.get('score_distribution', {}).get('mean', 0):.4f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Median</div>
                        <div class="value">{metrics.get('score_distribution', {}).get('median', 0):.4f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Std</div>
                        <div class="value">{metrics.get('score_distribution', {}).get('std', 0):.4f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Min</div>
                        <div class="value">{metrics.get('score_distribution', {}).get('min', 0):.4f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Max</div>
                        <div class="value">{metrics.get('score_distribution', {}).get('max', 0):.4f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Q25</div>
                        <div class="value">{metrics.get('score_distribution', {}).get('q25', 0):.4f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Q75</div>
                        <div class="value">{metrics.get('score_distribution', {}).get('q75', 0):.4f}</div>
                    </div>
                </div>
            </div>
            
            <!-- Summary -->
            <div class="section">
                <h2>📋 Tóm Tắt</h2>
                <p><strong>Tổng số pairs đánh giá:</strong> {total_pairs}</p>
                <p><strong>High Similarity pairs:</strong> {high_count} ({high_count/total_pairs*100:.1f}%)</p>
                <p><strong>Medium Similarity pairs:</strong> {medium_count} ({medium_count/total_pairs*100:.1f}%)</p>
                <p><strong>Random/Low Similarity pairs:</strong> {random_count} ({random_count/total_pairs*100:.1f}%)</p>
                <p><strong>Label Correlation:</strong> {metrics.get('label_correlation', 0):.4f}</p>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by AI Service Evaluation System</p>
            <p>Dataset: 500 Pairs Ground Truth</p>
        </div>
    </div>
</body>
</html>"""
    
    # Save HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"✓ HTML report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Simple Evaluation of 500 Pairs with Beautiful HTML Report'
    )
    
    parser.add_argument(
        '--ground-truth-csv',
        type=str,
        default='ground_truth_500_pairs.csv',
        help='Path to ground truth CSV (default: ground_truth_500_pairs.csv)'
    )
    parser.add_argument(
        '--output-html',
        type=str,
        default='reports/evaluation_500_pairs_report.html',
        help='Output HTML report path (default: reports/evaluation_500_pairs_report.html)'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default='reports/evaluation_500_pairs_results.json',
        help='Output JSON results path (default: reports/evaluation_500_pairs_results.json)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_html), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    
    try:
        # Load ground truth
        df = load_ground_truth_csv(args.ground_truth_csv)
        
        # Calculate metrics
        metrics, predictions, labels = calculate_metrics(df)
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"AUC-ROC: {metrics.get('auc_roc', 0):.4f}")
        logger.info(f"Precision: {metrics.get('precision', 0):.4f}")
        logger.info(f"Recall: {metrics.get('recall', 0):.4f}")
        logger.info(f"F1-Score: {metrics.get('f1', 0):.4f}")
        logger.info(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
        
        # Generate HTML report
        generate_html_report(metrics, predictions, labels, args.output_html)
        
        # Save JSON results
        results = {
            'timestamp': datetime.now().isoformat(),
            'ground_truth_csv': args.ground_truth_csv,
            'metrics': metrics,
            'num_samples': len(predictions)
        }
        
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ JSON results saved to: {args.output_json}")
        logger.info(f"\n✅ Evaluation completed!")
        logger.info(f"📄 HTML Report: {args.output_html}")
        logger.info(f"📊 JSON Results: {args.output_json}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()


