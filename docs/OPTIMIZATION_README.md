# Model Optimization Branch - Quick Start

## Mục Đích

Nhánh này được tạo để:
1. Benchmark 10 biến thể model khác nhau
2. Tìm ra model tốt nhất dựa trên performance và quality
3. Tối ưu toàn bộ hệ thống

## 10 Biến Thể Model

1. **Current_SimCSE_Vietnamese** - Model hiện tại (baseline)
2. **Multilingual_MPNet** - Model đa ngôn ngữ MPNet
3. **Vietnamese_SBERT** - SBERT chuyên cho tiếng Việt
4. **MiniLM_Multilingual** - Model nhanh, nhẹ
5. **MPNet_Base** - Model chất lượng cao
6. **QA_MPNet** - Model tối ưu cho matching
7. **SimCSE_LargeBatch** - Tối ưu tốc độ với batch lớn
8. **SimCSE_NoNormalize** - So sánh với/không normalize
9. **Weighted_SimCSE** - Phương pháp weighted embedding
10. **MultiVector_SimCSE** - Phương pháp multi-vector pooling

## Quick Start

### 1. Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

### 2. Chạy Benchmark

```bash
python scripts/benchmark_model_variations.py
```

### 3. Xem Kết Quả

```bash
# Xem báo cáo markdown
cat reports/benchmark_variations/benchmark_report_*.md

# Hoặc xem JSON
cat reports/benchmark_variations/benchmark_results_*.json
```

### 4. Tối Ưu Hệ Thống

```bash
python scripts/optimize_system.py
```

### 5. Xem Báo Cáo Optimization

```bash
cat reports/optimization/optimization_report.md
```

## Files Mới

- `src/embeddings/model_variations.py` - Định nghĩa 10 variations
- `scripts/benchmark_model_variations.py` - Script benchmark
- `scripts/optimize_system.py` - Script tối ưu
- `docs/OPTIMIZATION_BENCHMARK.md` - Tài liệu chi tiết

## Kết Quả

Sau khi chạy, bạn sẽ có:
- Rankings của tất cả variations
- Metrics chi tiết (speed, quality, memory)
- Recommendation cho model tốt nhất
- Optimized configuration

## Next Steps

Sau khi chọn được model tốt nhất:
1. Update `.env` với model mới
2. Regenerate embeddings
3. Rebuild FAISS indices
4. Re-run pre-computation
5. Test hệ thống

Xem `docs/OPTIMIZATION_BENCHMARK.md` để biết chi tiết.

