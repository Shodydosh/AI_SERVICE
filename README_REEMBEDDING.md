# Re-Embedding Guide

Hướng dẫn re-embedding toàn bộ dữ liệu với phương pháp field-by-field mapping được cải thiện.

## Cải tiến mới

### 1. Improved Field Embedding Generator
- **Better text preprocessing**: Normalize và giới hạn độ dài text
- **Descriptive field labels**: Sử dụng labels rõ ràng hơn (e.g., "Required Skills" thay vì "skills")
- **Batch processing**: Hỗ trợ batch embedding để tăng hiệu suất
- **Improved similarity calculation**: Tính toán chính xác hơn với normalization

### 2. Optimized Weights
- **JD embeddings**: title (25%), requirements (45%), description (30%)
- **Candidate embeddings**: skills (40%), experience (35%), desired_job (25%)

### 3. Performance Improvements
- Batch processing cho database operations
- Progress tracking với rate calculation
- Better error handling và logging

## Cách sử dụng

### Re-embedding toàn bộ dữ liệu

```bash
# Re-embed tất cả (clear existing data)
python scripts/reembed_all_data.py \
    --jd-file data/processed/jd_processed.csv \
    --candidate-file data/processed/candidate_processed.csv \
    --clear-all
```

### Re-embedding từng phần

```bash
# Chỉ re-embed JDs
python scripts/reembed_all_data.py \
    --jd-file data/processed/jd_processed.csv \
    --clear-jd

# Chỉ re-embed candidates
python scripts/reembed_all_data.py \
    --candidate-file data/processed/candidate_processed.csv \
    --clear-candidates

# Clear recommendations (sẽ cần generate lại)
python scripts/reembed_all_data.py \
    --clear-recommendations
```

### Custom batch size

```bash
# Tăng batch size để nhanh hơn (nếu có đủ RAM)
python scripts/reembed_all_data.py \
    --jd-file data/processed/jd_processed.csv \
    --candidate-file data/processed/candidate_processed.csv \
    --batch-size 100 \
    --embedding-batch-size 64
```

## Workflow hoàn chỉnh

### Bước 1: Re-embedding
```bash
python scripts/reembed_all_data.py \
    --jd-file data/processed/jd_processed.csv \
    --candidate-file data/processed/candidate_processed.csv \
    --clear-all
```

### Bước 2: Generate Recommendations
```bash
python scripts/generate_processed_recommendations.py \
    --candidate-file data/processed/candidate_processed.csv \
    --jd-file data/processed/jd_processed.csv \
    --top-k 10
```

### Hoặc chạy full workflow
```bash
python scripts/run_full_workflow.py \
    --candidate-file data/processed/candidate_processed.csv \
    --jd-file data/processed/jd_processed.csv
```

## Field Mappings

Hệ thống sử dụng 3 cặp field mapping:

1. **candidate.skills** → **jd.requirements** (weight: 40%)
2. **candidate.experience** → **jd.requirements** (weight: 35%)
3. **candidate.desired_job** → **jd.title** (weight: 25%)

## Performance

Với cải tiến mới:
- **JD embedding**: ~50-100 JDs/giây (tùy model)
- **Candidate embedding**: ~50-100 candidates/giây (tùy model)
- **Batch processing**: Giảm thời gian xử lý đáng kể

## Monitoring

Script sẽ hiển thị:
- Progress bar với tqdm
- Processing rate (items/second)
- Total time elapsed
- Error logs nếu có

## Troubleshooting

### Lỗi memory
- Giảm `--batch-size` và `--embedding-batch-size`
- Xử lý từng file riêng biệt

### Lỗi database connection
- Kiểm tra database connection trong `config/settings.py`
- Đảm bảo PostgreSQL đang chạy

### Lỗi model loading
- Kiểm tra model name trong `config/settings.py`
- Đảm bảo model đã được download

## Next Steps

Sau khi re-embedding:
1. Generate recommendations: `python scripts/generate_processed_recommendations.py`
2. Test matching: Sử dụng `MatchingService` với `use_processed=True`
3. Benchmark: `python scripts/benchmark_field_mapping.py`

