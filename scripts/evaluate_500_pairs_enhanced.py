"""Enhanced evaluation script for 500 pairs with beautiful HTML report."""
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

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session
from src.database.connection import SessionLocal
from src.models.two_tower_model import TwoTowerModel
from src.models.training_pipeline import GroundTruthDataset, collate_fn
from src.models.evaluation_metrics import TwoTowerEvaluator
from src.database.multi_field_repository import MultiFieldEmbeddingRepository
from torch.utils.data import DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_ground_truth_csv(csv_path: str) -> List[Dict]:
    """Load ground truth pairs from CSV."""
    logger.info(f"Loading ground truth from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    ground_truth = []
    
    for _, row in df.iterrows():
        similarity_type = str(row.get('similarity_type', 'random')).lower()
        
        # Map similarity type to label
        if similarity_type == 'high':
            label = 1.0
        elif similarity_type == 'medium':
            label = 0.7
        else:  # random
            label = 0.0
        
        ground_truth.append({
            'candidate_id': str(row['candidate_id']),
            'job_id': str(row['job_id']),
            'label': label,
            'similarity_type': similarity_type,
            'predicted_similarity': float(row.get('predicted_similarity', 0.0)),
            'candidate_title': str(row.get('candidate_title', '')),
            'job_title': str(row.get('job_title', '')),
            'candidate_skills': str(row.get('candidate_skills', '')),
            'job_requirements': str(row.get('job_requirements', ''))
        })
    
    logger.info(f"Loaded {len(ground_truth)} pairs")
    logger.info(f"  - High: {sum(1 for p in ground_truth if p['label'] == 1.0)}")
    logger.info(f"  - Medium: {sum(1 for p in ground_truth if p['label'] == 0.7)}")
    logger.info(f"  - Random: {sum(1 for p in ground_truth if p['label'] == 0.0)}")
    
    return ground_truth


def evaluate_model(
    model_path: str,
    ground_truth: List[Dict],
    embedding_dim: int = 768,
    hidden_dims: List[int] = [512, 256],
    output_dim: int = 256,
    batch_size: int = 32,
    device: str = 'cpu'
) -> Tuple[Dict, np.ndarray, np.ndarray, List[Dict]]:
    """Evaluate model and return comprehensive results."""
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATING TWO-TOWER MODEL")
    logger.info("=" * 80)
    
    # Load model
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for model evaluation")
    
    logger.info(f"\n1. Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = TwoTowerModel(
        embedding_dim=embedding_dim,
        candidate_hidden_dims=hidden_dims,
        job_hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout=0.1,
        use_batch_norm=True
    )
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Try loading with strict=False first (in case architecture changed)
        try:
            model.load_state_dict(state_dict, strict=True)
            logger.info("✓ Model loaded successfully (strict mode)")
        except RuntimeError as e:
            logger.warning(f"Strict loading failed, trying flexible loading: {e}")
            # Try flexible loading - only load matching keys
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() 
                             if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            logger.info(f"✓ Model loaded successfully (flexible mode, {len(pretrained_dict)}/{len(state_dict)} keys loaded)")
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    model.to(device)
    model.eval()
    
    # Load data from database
    logger.info("\n2. Loading data from database...")
    db = SessionLocal()
    try:
        repository = MultiFieldEmbeddingRepository(db)
        
        dataset = GroundTruthDataset(ground_truth, repository)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        logger.info(f"✓ Dataset created: {len(dataset)} samples")
        
        # Evaluate
        logger.info("\n3. Computing predictions...")
        evaluator = TwoTowerEvaluator()
        
        all_predictions = []
        all_labels = []
        all_candidate_ids = []
        all_job_ids = []
        all_similarity_types = []
        all_details = []
        
        num_batches = len(dataloader)
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"  Processing batch {batch_idx + 1}/{num_batches}...")
                
                candidate_title = batch['candidate_title'].to(device)
                candidate_skills = batch['candidate_skills'].to(device)
                candidate_experience = batch['candidate_experience'].to(device)
                job_title = batch['job_title'].to(device)
                job_skills = batch['job_skills'].to(device)
                job_requirement = batch['job_requirement'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                candidate_repr, job_repr = model(
                    candidate_title, candidate_skills, candidate_experience,
                    job_title, job_skills, job_requirement
                )
                
                # Compute similarity
                similarity = model.compute_similarity(candidate_repr, job_repr)
                
                # Store results
                predictions_np = similarity.cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                all_predictions.extend(predictions_np)
                all_labels.extend(labels_np)
                all_candidate_ids.extend(batch['candidate_id'])
                all_job_ids.extend(batch['job_id'])
                
                # Store details
                for i, cand_id in enumerate(batch['candidate_id']):
                    job_id = batch['job_id'][i]
                    pair = next(
                        (p for p in ground_truth 
                         if p['candidate_id'] == cand_id and p['job_id'] == job_id),
                        None
                    )
                    
                    sim_type = pair.get('similarity_type', 'unknown') if pair else 'unknown'
                    all_similarity_types.append(sim_type)
                    
                    all_details.append({
                        'candidate_id': cand_id,
                        'job_id': job_id,
                        'prediction': float(predictions_np[i]),
                        'label': float(labels_np[i]),
                        'similarity_type': sim_type,
                        'candidate_title': pair.get('candidate_title', '') if pair else '',
                        'job_title': pair.get('job_title', '') if pair else ''
                    })
        
        logger.info(f"✓ Computed predictions for {len(all_predictions)} pairs")
        
        # Convert to numpy
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        
        # Compute metrics
        logger.info("\n4. Computing evaluation metrics...")
        metrics = evaluator.evaluate(
            predictions, labels, all_candidate_ids, all_job_ids
        )
        
        # Additional analysis
        logger.info("\n5. Additional analysis...")
        
        # Analysis by similarity type
        high_mask = np.array([t == 'high' for t in all_similarity_types])
        medium_mask = np.array([t == 'medium' for t in all_similarity_types])
        random_mask = np.array([t == 'random' for t in all_similarity_types])
        
        if np.any(high_mask):
            metrics['high_similarity'] = {
                'count': int(np.sum(high_mask)),
                'mean_score': float(np.mean(predictions[high_mask])),
                'std_score': float(np.std(predictions[high_mask])),
                'min_score': float(np.min(predictions[high_mask])),
                'max_score': float(np.max(predictions[high_mask]))
            }
        
        if np.any(medium_mask):
            metrics['medium_similarity'] = {
                'count': int(np.sum(medium_mask)),
                'mean_score': float(np.mean(predictions[medium_mask])),
                'std_score': float(np.std(predictions[medium_mask])),
                'min_score': float(np.min(predictions[medium_mask])),
                'max_score': float(np.max(predictions[medium_mask]))
            }
        
        if np.any(random_mask):
            metrics['random_similarity'] = {
                'count': int(np.sum(random_mask)),
                'mean_score': float(np.mean(predictions[random_mask])),
                'std_score': float(np.std(predictions[random_mask])),
                'min_score': float(np.min(predictions[random_mask])),
                'max_score': float(np.max(predictions[random_mask]))
            }
        
        # Confusion matrix (using threshold 0.5)
        binary_preds = (predictions >= 0.5).astype(int)
        binary_labels = (labels >= 0.5).astype(int)
        
        tp = np.sum((binary_preds == 1) & (binary_labels == 1))
        fp = np.sum((binary_preds == 1) & (binary_labels == 0))
        tn = np.sum((binary_preds == 0) & (binary_labels == 0))
        fn = np.sum((binary_preds == 0) & (binary_labels == 1))
        
        metrics['confusion_matrix'] = {
            'true_positive': int(tp),
            'false_positive': int(fp),
            'true_negative': int(tn),
            'false_negative': int(fn)
        }
        
        # Score distribution
        metrics['score_distribution'] = {
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'median': float(np.median(predictions)),
            'q25': float(np.percentile(predictions, 25)),
            'q75': float(np.percentile(predictions, 75))
        }
        
        # Correlation
        if len(np.unique(labels)) > 1:
            correlation = np.corrcoef(predictions, labels)[0, 1]
            metrics['label_correlation'] = (
                float(correlation) if not np.isnan(correlation) else 0.0
            )
        
        return metrics, predictions, labels, all_details
        
    finally:
        db.close()


def generate_html_report(
    metrics: Dict,
    predictions: np.ndarray,
    labels: np.ndarray,
    details: List[Dict],
    output_path: str
):
    """Generate beautiful HTML report."""
    logger.info(f"\n6. Generating HTML report: {output_path}")
    
    # Calculate additional stats
    total_pairs = len(predictions)
    high_count = metrics.get('high_similarity', {}).get('count', 0)
    medium_count = metrics.get('medium_similarity', {}).get('count', 0)
    random_count = metrics.get('random_similarity', {}).get('count', 0)
    
    # Overall assessment
    auc_roc = metrics.get('auc_roc', 0)
    ndcg = metrics.get('ndcg@10', 0)
    
    if auc_roc > 0.8 and ndcg > 0.7:
        overall_status = "EXCELLENT"
        status_color = "#28a745"
        status_icon = "✅"
    elif auc_roc > 0.7 and ndcg > 0.6:
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
    
    # HTML template
    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo Cáo Đánh Giá 500 Pairs - Two-Tower Model</title>
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
        
        .badge-excellent {{
            background: #28a745;
            color: white;
        }}
        
        .badge-good {{
            background: #17a2b8;
            color: white;
        }}
        
        .badge-moderate {{
            background: #ffc107;
            color: #333;
        }}
        
        .badge-poor {{
            background: #dc3545;
            color: white;
        }}
        
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
        
        .cm-tp {{
            background: #28a745;
        }}
        
        .cm-fp {{
            background: #ffc107;
        }}
        
        .cm-tn {{
            background: #17a2b8;
        }}
        
        .cm-fn {{
            background: #dc3545;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #ddd;
        }}
        
        .chart-container {{
            margin: 20px 0;
            padding: 20px;
            background: white;
            border-radius: 10px;
        }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Báo Cáo Đánh Giá 500 Pairs</h1>
            <div class="subtitle">Two-Tower Model Evaluation Report</div>
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
                        <div class="value">{auc_roc:.4f}</div>
                        <span class="badge {'badge-excellent' if auc_roc > 0.8 else 'badge-good' if auc_roc > 0.7 else 'badge-moderate' if auc_roc > 0.6 else 'badge-poor'}">
                            {'Excellent' if auc_roc > 0.8 else 'Good' if auc_roc > 0.7 else 'Moderate' if auc_roc > 0.6 else 'Poor'}
                        </span>
                    </div>
                    <div class="metric-card">
                        <div class="label">NDCG@10</div>
                        <div class="value">{ndcg:.4f}</div>
                        <span class="badge {'badge-excellent' if ndcg > 0.7 else 'badge-good' if ndcg > 0.6 else 'badge-moderate' if ndcg > 0.5 else 'badge-poor'}">
                            {'Excellent' if ndcg > 0.7 else 'Good' if ndcg > 0.6 else 'Moderate' if ndcg > 0.5 else 'Poor'}
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
            
            <!-- Ranking Metrics -->
            <div class="section">
                <h2>🏆 Ranking Metrics</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>NDCG@10</td>
                            <td>{metrics.get('ndcg@10', 0):.4f}</td>
                            <td>{'⭐ Excellent' if metrics.get('ndcg@10', 0) > 0.7 else '✓ Good' if metrics.get('ndcg@10', 0) > 0.6 else '⚠ Moderate'}</td>
                        </tr>
                        <tr>
                            <td>MRR</td>
                            <td>{metrics.get('mrr', 0):.4f}</td>
                            <td>-</td>
                        </tr>
                        <tr>
                            <td>Precision@5</td>
                            <td>{metrics.get('precision@5', 0):.4f}</td>
                            <td>-</td>
                        </tr>
                        <tr>
                            <td>Precision@10</td>
                            <td>{metrics.get('precision@10', 0):.4f}</td>
                            <td>-</td>
                        </tr>
                        <tr>
                            <td>Recall@5</td>
                            <td>{metrics.get('recall@5', 0):.4f}</td>
                            <td>-</td>
                        </tr>
                        <tr>
                            <td>Recall@10</td>
                            <td>{metrics.get('recall@10', 0):.4f}</td>
                            <td>-</td>
                        </tr>
                    </tbody>
                </table>
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
            <p>Model: Two-Tower Architecture | Dataset: 500 Pairs Ground Truth</p>
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
        description='Enhanced Evaluation of Two-Tower Model with Beautiful HTML Report',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='outputs_improved/best_model_improved.pt',
        help='Path to trained model (default: outputs_improved/best_model_improved.pt)'
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
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=768,
        help='Embedding dimension (default: 768)'
    )
    parser.add_argument(
        '--hidden-dims',
        type=int,
        nargs='+',
        default=[512, 256],
        help='Hidden layer dimensions (default: [512, 256])'
    )
    parser.add_argument(
        '--output-dim',
        type=int,
        default=256,
        help='Output dimension (default: 256)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device (default: cpu)'
    )
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and TORCH_AVAILABLE and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_html), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    
    try:
        # Load ground truth
        ground_truth = load_ground_truth_csv(args.ground_truth_csv)
        
        # Evaluate
        metrics, predictions, labels, details = evaluate_model(
            model_path=args.model_path,
            ground_truth=ground_truth,
            embedding_dim=args.embedding_dim,
            hidden_dims=args.hidden_dims,
            output_dim=args.output_dim,
            batch_size=args.batch_size,
            device=args.device
        )
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"AUC-ROC: {metrics.get('auc_roc', 0):.4f}")
        logger.info(f"NDCG@10: {metrics.get('ndcg@10', 0):.4f}")
        logger.info(f"Precision: {metrics.get('precision', 0):.4f}")
        logger.info(f"Recall: {metrics.get('recall', 0):.4f}")
        logger.info(f"F1-Score: {metrics.get('f1', 0):.4f}")
        
        # Generate HTML report
        generate_html_report(metrics, predictions, labels, details, args.output_html)
        
        # Save JSON results
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': args.model_path,
            'ground_truth_csv': args.ground_truth_csv,
            'metrics': metrics,
            'num_samples': len(predictions),
            'config': {
                'embedding_dim': args.embedding_dim,
                'hidden_dims': args.hidden_dims,
                'output_dim': args.output_dim,
                'batch_size': args.batch_size,
                'device': args.device
            }
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

