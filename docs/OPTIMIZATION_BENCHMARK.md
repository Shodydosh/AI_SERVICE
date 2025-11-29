# Model Variations Benchmark & System Optimization

## Tổng Quan

Tài liệu này mô tả quy trình benchmark 10 biến thể model và tối ưu hệ thống dựa trên kết quả.

## 10 Biến Thể Model

### 1. Current_SimCSE_Vietnamese (Baseline)
- **Model**: `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base`
- **Dimension**: 768
- **Batch Size**: 32
- **Normalize**: True
- **Tokenization**: True
- **Mô tả**: Model hiện tại đang sử dụng, làm baseline để so sánh

### 2. Multilingual_MPNet
- **Model**: `paraphrase-multilingual-mpnet-base-v2`
- **Dimension**: 768
- **Batch Size**: 32
- **Normalize**: True
- **Tokenization**: False
- **Mô tả**: Model đa ngôn ngữ hỗ trợ 50+ ngôn ngữ bao gồm tiếng Việt

### 3. Vietnamese_SBERT
- **Model**: `keepitreal/vietnamese-sbert`
- **Dimension**: 768
- **Batch Size**: 32
- **Normalize**: True
- **Tokenization**: False
- **Mô tả**: Model SBERT chuyên cho tiếng Việt

### 4. MiniLM_Multilingual
- **Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Dimension**: 384
- **Batch Size**: 64
- **Normalize**: True
- **Tokenization**: False
- **Mô tả**: Model nhanh, nhẹ, đa ngôn ngữ

### 5. MPNet_Base
- **Model**: `all-mpnet-base-v2`
- **Dimension**: 768
- **Batch Size**: 32
- **Normalize**: True
- **Tokenization**: False
- **Mô tả**: Model chất lượng cao cho semantic matching

### 6. QA_MPNet
- **Model**: `multi-qa-mpnet-base-dot-v1`
- **Dimension**: 768
- **Batch Size**: 32
- **Normalize**: True
- **Tokenization**: False
- **Mô tả**: Model tối ưu cho question-answer matching

### 7. SimCSE_LargeBatch
- **Model**: `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base`
- **Dimension**: 768
- **Batch Size**: 128
- **Normalize**: True
- **Tokenization**: True
- **Mô tả**: Model hiện tại với batch size lớn để tối ưu tốc độ

### 8. SimCSE_NoNormalize
- **Model**: `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base`
- **Dimension**: 768
- **Batch Size**: 32
- **Normalize**: False
- **Tokenization**: True
- **Mô tả**: Model hiện tại không normalize để so sánh

### 9. Weighted_SimCSE
- **Model**: `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base`
- **Dimension**: 768
- **Batch Size**: 32
- **Normalize**: True
- **Tokenization**: True
- **Method**: Weighted embedding
- **Mô tả**: Sử dụng trọng số cho các trường khác nhau (Skills, Experience có trọng số cao hơn)

### 10. MultiVector_SimCSE
- **Model**: `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base`
- **Dimension**: 768
- **Batch Size**: 32
- **Normalize**: True
- **Tokenization**: True
- **Method**: Multi-vector pooling
- **Mô tả**: Tạo nhiều vector cho các trường khác nhau và kết hợp bằng weighted pooling

## Metrics Đánh Giá

### 1. Performance Metrics
- **JD Single Avg Time**: Thời gian trung bình tạo embedding cho 1 JD
- **JD Batch Throughput**: Số lượng embeddings/giây khi xử lý batch
- **Candidate Single Avg Time**: Thời gian trung bình tạo embedding cho 1 candidate
- **Candidate Batch Throughput**: Số lượng embeddings/giây khi xử lý batch candidates

### 2. Quality Metrics
- **Cross Similarity Mean**: Độ tương đồng trung bình giữa JD và Candidate embeddings
- **Cross Similarity Std**: Độ lệch chuẩn của cross similarity
- **Top-5 Similarity Mean**: Độ tương đồng trung bình của top 5 matches
- **JD Self Similarity**: Độ tương đồng giữa các JDs (để đánh giá diversity)
- **Candidate Self Similarity**: Độ tương đồng giữa các candidates

### 3. Resource Metrics
- **Memory Usage**: Bộ nhớ sử dụng (MB)
- **Model Size**: Kích thước model

### 4. Composite Score
Tổng hợp các metrics:
```
Composite Score = (Speed Score × 0.3) + (Quality Score × 0.5) + (Throughput Score × 0.2)
```

## Cách Sử Dụng

### 1. Chạy Benchmark

```bash
# Benchmark tất cả 10 variations
python scripts/benchmark_model_variations.py

# Benchmark với sample size cụ thể
python scripts/benchmark_model_variations.py --sample-size 200

# Benchmark chỉ một số variations cụ thể
python scripts/benchmark_model_variations.py --variations 1 2 3
```

### 2. Xem Kết Quả

Kết quả được lưu trong:
- `reports/benchmark_variations/benchmark_results_*.json`: Dữ liệu chi tiết
- `reports/benchmark_variations/benchmark_report_*.md`: Báo cáo markdown

### 3. Tối Ưu Hệ Thống

```bash
# Tối ưu dựa trên composite score (mặc định)
python scripts/optimize_system.py

# Tối ưu dựa trên tốc độ
python scripts/optimize_system.py --criteria speed

# Tối ưu dựa trên chất lượng
python scripts/optimize_system.py --criteria quality

# Tối ưu cân bằng
python scripts/optimize_system.py --criteria balanced
```

### 4. Áp Dụng Tối Ưu

Sau khi chạy optimization:
1. Xem báo cáo trong `reports/optimization/optimization_report.md`
2. Cập nhật `.env` file với model được đề xuất
3. Regenerate embeddings với model mới
4. Rebuild FAISS indices
5. Re-run pre-computation

## Workflow Hoàn Chỉnh

```bash
# 1. Chạy benchmark
python scripts/benchmark_model_variations.py --sample-size 100

# 2. Xem kết quả
cat reports/benchmark_variations/benchmark_report_*.md

# 3. Tối ưu hệ thống
python scripts/optimize_system.py --criteria composite

# 4. Xem báo cáo optimization
cat reports/optimization/optimization_report.md

# 5. Cập nhật config (nếu cần)
# Edit .env file với model được đề xuất

# 6. Regenerate embeddings
python scripts/rerun_system.py

# 7. Test hệ thống
python scripts/evaluate_system_comprehensive.py
```

## Kết Quả Benchmark

Sau khi chạy benchmark, bạn sẽ có:

1. **Rankings**: Bảng xếp hạng các variations theo composite score
2. **Detailed Metrics**: Metrics chi tiết cho từng variation
3. **Recommendations**: Gợi ý variation tốt nhất cho từng use case

## Lưu Ý

- Benchmark có thể mất thời gian (đặc biệt khi load nhiều models)
- Đảm bảo có đủ RAM (mỗi model có thể tốn 500MB-1GB)
- Kết quả có thể khác nhau tùy vào hardware
- Nên chạy benchmark trên cùng một máy để so sánh công bằng

## Tối Ưu Hệ Thống

Sau khi chọn được variation tốt nhất, script optimization sẽ:

1. Tạo file config tối ưu
2. Cập nhật `.env` file
3. Tạo báo cáo optimization với:
   - Variation được chọn
   - Performance improvements so với baseline
   - Next steps để áp dụng

## Files Liên Quan

- `src/embeddings/model_variations.py`: Định nghĩa 10 variations
- `scripts/benchmark_model_variations.py`: Script benchmark
- `scripts/optimize_system.py`: Script tối ưu hệ thống
- `reports/benchmark_variations/`: Thư mục chứa kết quả benchmark
- `reports/optimization/`: Thư mục chứa kết quả optimization

