# Hướng Dẫn Chạy Evaluation Chi Tiết

## 📋 Yêu Cầu

Trước khi chạy evaluation, cần cài đặt các dependencies:

```bash
pip install -r requirements.txt
```

Hoặc cài đặt thủ công:
```bash
pip install torch numpy sqlalchemy psycopg2-binary pandas scikit-learn tqdm
```

## 🚀 Chạy Evaluation

### Bước 1: Đảm bảo có model đã train

Kiểm tra model có tồn tại:
```bash
# Windows
dir outputs_improved\best_model_improved.pt

# Linux/Mac
ls outputs_improved/best_model_improved.pt
```

### Bước 2: Đảm bảo có ground truth data

File CSV: `ground_truth_500_pairs.csv` (đã có sẵn)

### Bước 3: Chạy evaluation

```bash
python scripts/evaluate_two_tower_detailed.py \
    --model-path outputs_improved/best_model_improved.pt \
    --ground-truth-csv ground_truth_500_pairs.csv \
    --batch-size 16 \
    --device cpu
```

### Bước 4: Xem kết quả

Script sẽ hiển thị:
- ✅ Classification Metrics (Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR)
- ✅ Ranking Metrics (NDCG@10, MRR, Precision@K, Recall@K)
- ✅ Score Distribution (Mean, Std, Min, Max, Median)
- ✅ Analysis by Similarity Type (High/Medium/Random)
- ✅ Correlation Analysis

## 📊 Kết Quả Mẫu

```
================================================================================
DETAILED EVALUATION RESULTS
================================================================================

📊 CLASSIFICATION METRICS
--------------------------------------------------------------------------------
  Accuracy:        0.8500
  Precision:       0.8200
  Recall:          0.8800
  F1-Score:        0.8500
  AUC-ROC:         0.9200  ⭐ Excellent
  AUC-PR:          0.8900

📈 RANKING METRICS
--------------------------------------------------------------------------------
  NDCG@10:         0.8500  ⭐ Excellent
  MRR:             0.9000
  Precision@5:     0.8200
  Precision@10:    0.8000
  Recall@5:        0.7500
  Recall@10:       0.8800

📉 PREDICTION SCORE DISTRIBUTION
--------------------------------------------------------------------------------
  Mean:            0.6500
  Std:             0.2500
  Min:             0.1000
  Max:             0.9800
  Median:          0.7000

🔍 ANALYSIS BY SIMILARITY TYPE
--------------------------------------------------------------------------------
  HIGH Similarity Pairs (167 pairs):
    Mean Score:    0.8500
    Std:           0.1000
    Expected:      0.8-1.0 (High similarity should have high scores)

  MEDIUM Similarity Pairs (167 pairs):
    Mean Score:    0.6500
    Std:           0.1500
    Expected:      0.5-0.8 (Medium similarity should have medium scores)

  RANDOM/LOW Similarity Pairs (166 pairs):
    Mean Score:    0.3500
    Std:           0.2000
    Expected:      0.0-0.4 (Low similarity should have low scores)
```

## 💾 Lưu Kết Quả

Để lưu kết quả vào file JSON:

```bash
python scripts/evaluate_two_tower_detailed.py \
    --model-path outputs_improved/best_model_improved.pt \
    --ground-truth-csv ground_truth_500_pairs.csv \
    --output-file evaluation_results.json
```

## ⚠️ Troubleshooting

### Lỗi: ModuleNotFoundError

**Giải pháp:** Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

### Lỗi: Model file not found

**Giải pháp:** 
1. Train model trước: `python scripts/train_two_tower.py`
2. Hoặc chỉ định đúng path đến model

### Lỗi: Database connection failed

**Giải pháp:**
1. Đảm bảo PostgreSQL đang chạy
2. Kiểm tra connection settings trong `config/settings.py`
3. Đảm bảo embeddings đã được process vào database

### Lỗi: No embeddings found

**Giải pháp:**
1. Chạy lại process embeddings:
   ```bash
   python scripts/process_multi_field_embeddings.py --process-all
   ```

## 📝 Notes

- Evaluation sử dụng CPU mặc định (có thể dùng `--device cuda` nếu có GPU)
- Batch size mặc định là 32 (có thể giảm nếu thiếu memory)
- Kết quả được hiển thị chi tiết với emoji và formatting để dễ đọc














