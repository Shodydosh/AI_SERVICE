# 🚀 Two-Tower Architecture - Quick Start

## Tổng Quan

Kiến trúc Two-Tower học cách kết hợp 3 embedding fields (title, skills, experience/requirement) thành representation tối ưu cho matching.

## Quick Start

### 1. Build Ground Truth

```bash
python scripts/train_two_tower.py \
    --build-ground-truth \
    --ground-truth-path data/ground_truth.json \
    --max-candidates 500 \
    --max-jobs 2000
```

### 2. Train Model

```bash
python scripts/train_two_tower.py \
    --ground-truth-path data/ground_truth.json \
    --output-dir models/two_tower \
    --num-epochs 10 \
    --device cpu
```

### 3. Evaluate & Compare

```bash
python scripts/evaluate_two_tower.py \
    --ground-truth-path data/ground_truth.json \
    --model-path models/two_tower/best_model.pt
```

## Kết Quả

Script evaluation sẽ hiển thị:
- **Baseline metrics**: Weighted average method
- **Two-Tower metrics**: Neural network method
- **Improvement**: So sánh và % cải thiện

## Files Created

- `data/ground_truth.json`: Ground truth dataset
- `models/two_tower/best_model.pt`: Best model checkpoint
- `models/two_tower/training_history.json`: Training history

## Xem Chi Tiết

Xem [TWO_TOWER_ARCHITECTURE.md](./TWO_TOWER_ARCHITECTURE.md) để biết thêm chi tiết.


