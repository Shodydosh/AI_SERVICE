# 🏗️ Two-Tower Architecture Documentation

## 📋 Tổng Quan

Kiến trúc **Two-Tower** là một mô hình neural network được thiết kế để học cách kết hợp 3 embedding fields (title, skills, experience/requirement) thành một representation tối ưu cho việc matching candidate với job.

### Kiến Trúc

```
┌─────────────────────────────────────────────────────────┐
│                    CANDIDATE TOWER                      │
│  Input: [title_emb, skills_emb, experience_emb]        │
│  → Dense Layers [512, 256]                              │
│  → Output: candidate_repr [256]                         │
└─────────────────────────────────────────────────────────┘
                          ↓
                    [Similarity]
                          ↓
┌─────────────────────────────────────────────────────────┐
│                      JOB TOWER                          │
│  Input: [title_emb, skills_emb, requirement_emb]       │
│  → Dense Layers [512, 256]                              │
│  → Output: job_repr [256]                               │
└─────────────────────────────────────────────────────────┘
```

## 🎯 Lợi Ích

1. **Học tự động cách kết hợp fields**: Model tự học weights tối ưu cho từng field
2. **Tối ưu end-to-end**: Train trực tiếp cho matching task
3. **Linh hoạt**: Có thể fine-tune với dữ liệu riêng
4. **Cải thiện ranking**: Tối ưu cho ranking metrics (NDCG, MRR)

## 📁 Cấu Trúc Files

```
src/models/
├── __init__.py                    # Module exports
├── two_tower_model.py             # Two-Tower model architecture
├── ground_truth_builder.py        # Build ground truth dataset
├── evaluation_metrics.py          # Evaluation metrics
└── training_pipeline.py           # Training pipeline

scripts/
├── train_two_tower.py             # Training script
└── evaluate_two_tower.py          # Evaluation & comparison script
```

## 🚀 Sử Dụng

### 1. Xây Dựng Ground Truth Dataset

Ground truth dataset được tạo tự động dựa trên:
- **Title similarity threshold**: ≥ 0.6
- **Skills similarity threshold**: ≥ 0.5
- **Experience similarity threshold**: ≥ 0.5
- **Combined threshold**: ≥ 0.55 hoặc ≥ 2 fields pass threshold

```bash
python scripts/train_two_tower.py \
    --build-ground-truth \
    --ground-truth-path data/ground_truth.json \
    --max-candidates 500 \
    --max-jobs 2000
```

### 2. Training Model

```bash
python scripts/train_two_tower.py \
    --ground-truth-path data/ground_truth.json \
    --output-dir models/two_tower \
    --embedding-dim 768 \
    --hidden-dims 512 256 \
    --output-dim 256 \
    --batch-size 32 \
    --num-epochs 10 \
    --learning-rate 0.001 \
    --device cpu
```

**Parameters:**
- `--embedding-dim`: Dimension của input embeddings (768 cho SimCSE-Vietnamese)
- `--hidden-dims`: Kích thước các hidden layers
- `--output-dim`: Dimension của output representation
- `--batch-size`: Batch size cho training
- `--num-epochs`: Số epochs
- `--learning-rate`: Learning rate
- `--device`: `cpu` hoặc `cuda`

### 3. Đánh Giá Model

```bash
python scripts/evaluate_two_tower.py \
    --ground-truth-path data/ground_truth.json \
    --model-path models/two_tower/best_model.pt \
    --embedding-dim 768 \
    --hidden-dims 512 256 \
    --output-dim 256 \
    --device cpu
```

Script này sẽ:
1. Đánh giá baseline method (weighted average)
2. Đánh giá Two-Tower model
3. So sánh và hiển thị improvement

## 📊 Evaluation Metrics

### Classification Metrics
- **Accuracy**: Tỷ lệ dự đoán đúng
- **Precision**: Tỷ lệ positive predictions đúng
- **Recall**: Tỷ lệ positive cases được tìm thấy
- **F1-Score**: Harmonic mean của precision và recall
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under Precision-Recall curve

### Ranking Metrics
- **NDCG@10**: Normalized Discounted Cumulative Gain tại top 10
- **MRR**: Mean Reciprocal Rank
- **Precision@K**: Precision tại top K (K=5, 10, 20)
- **Recall@K**: Recall tại top K (K=5, 10, 20)

### Similarity Metrics
- **Label Correlation**: Correlation giữa predictions và labels
- **Field Correlations**: Correlation với từng field similarity

## 🔧 Ground Truth Building

### Logic Labeling

**Positive Pair** (label=1) nếu:
- Combined similarity ≥ 0.55, HOẶC
- ≥ 2 fields pass individual thresholds

**Negative Pair** (label=0):
- Combined similarity < 0.55 VÀ
- < 2 fields pass thresholds

### Dataset Structure

```json
[
  {
    "candidate_id": "15001",
    "job_id": "JD001",
    "title_similarity": 0.75,
    "skills_similarity": 0.65,
    "experience_similarity": 0.58,
    "combined_similarity": 0.66,
    "label": 1
  },
  ...
]
```

## 🎓 Model Architecture Details

### Candidate Tower
- **Input**: 3 embeddings (title, skills, experience) × 768 dims = 2304 dims
- **Hidden Layers**: [512, 256]
- **Output**: 256-dim representation
- **Activation**: ReLU
- **Normalization**: BatchNorm + L2 normalization

### Job Tower
- **Input**: 3 embeddings (title, skills, requirement) × 768 dims = 2304 dims
- **Hidden Layers**: [512, 256]
- **Output**: 256-dim representation
- **Activation**: ReLU
- **Normalization**: BatchNorm + L2 normalization

### Similarity Computation
- **Method**: Dot product (cosine similarity vì cả 2 đều L2 normalized)
- **Loss**: BCE with Logits Loss

## 📈 Training Process

1. **Data Loading**: Load embeddings từ database
2. **Batch Processing**: Process theo batches
3. **Forward Pass**: 
   - Candidate → Candidate Tower → candidate_repr
   - Job → Job Tower → job_repr
   - Compute similarity
4. **Backward Pass**: Update weights
5. **Validation**: Evaluate trên validation set mỗi epoch
6. **Checkpointing**: Save best model và checkpoints

## 🔍 So Sánh với Baseline

Baseline method sử dụng **weighted average**:
```python
similarity = (
    title_sim * 0.2 +
    skills_sim * 0.4 +
    experience_sim * 0.4
)
```

Two-Tower học weights tự động và có thể capture non-linear relationships.

## 💡 Best Practices

1. **Ground Truth Quality**: Đảm bảo ground truth dataset có đủ positive/negative pairs
2. **Hyperparameter Tuning**: Thử các hidden_dims và output_dim khác nhau
3. **Early Stopping**: Monitor validation metrics để tránh overfitting
4. **Data Augmentation**: Có thể tạo thêm synthetic pairs nếu thiếu data
5. **Regularization**: Sử dụng dropout và weight_decay để tránh overfitting

## 🐛 Troubleshooting

### Out of Memory
- Giảm `batch_size`
- Giảm `max_candidates` và `max_jobs` khi build ground truth

### Poor Performance
- Kiểm tra ground truth quality
- Tăng số epochs
- Thử learning rate khác
- Tăng model capacity (hidden_dims)

### Slow Training
- Sử dụng GPU nếu có (`--device cuda`)
- Tăng batch_size
- Giảm số samples trong ground truth

## 📝 Notes

- Model được train với **BCE with Logits Loss** (binary classification)
- Output representations được **L2 normalized** để similarity = cosine similarity
- Model có thể được fine-tune với dữ liệu riêng của bạn
- Ground truth có thể được cải thiện bằng manual annotations


