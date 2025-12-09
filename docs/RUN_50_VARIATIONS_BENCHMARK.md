# 🚀 Hướng Dẫn Chạy Benchmark 50 Bộ Parameter

## 📋 Tổng Quan

Script này sẽ tự động:
1. ✅ Chạy benchmark cho **tất cả 50 bộ parameter** (10 models × 5 configs)
2. ✅ So sánh hiệu năng của tất cả bộ
3. ✅ Tự động chọn và hiển thị **bộ tốt nhất**
4. ✅ Lưu kết quả phân tích chi tiết

## 🎯 50 Bộ Parameter Gồm:

### 10 Base Models:
1. SimCSE_Vietnamese
2. Multilingual_MPNet
3. Vietnamese_SBERT
4. MiniLM_Multilingual
5. MPNet_Base
6. QA_MPNet
7. MiniLM_L6
8. MPNet_ST
9. Paraphrase_MiniLM
10. DistilUSE_Multilingual

### 5 Parameter Configs (mỗi model):
1. **v1_bs32_norm**: batch_size=32, normalize=True
2. **v2_bs64_norm**: batch_size=64, normalize=True
3. **v3_bs128_norm**: batch_size=128, normalize=True
4. **v4_bs32_norm_false**: batch_size=32, normalize=False
5. **v5_bs16_norm**: batch_size=16, normalize=True

## 🚀 Cách Chạy

### Option 1: Sử dụng PowerShell Script (Khuyến nghị - Windows)

```powershell
.\scripts\run_full_benchmark_50.ps1
```

### Option 2: Sử dụng Batch File (Windows CMD)

```cmd
scripts\run_full_benchmark_50.bat
```

### Option 3: Chạy Trực Tiếp Python

```bash
# Chạy tất cả 50 variations
python scripts/run_full_benchmark_50_variations.py \
    --candidate-file "data/raw/candidates_dataset.csv" \
    --jd-file "data/raw/job_data.csv" \
    --sample-size 50

# Test nhanh với 5 variations đầu tiên
python scripts/run_full_benchmark_50_variations.py \
    --candidate-file "data/raw/candidates_dataset.csv" \
    --jd-file "data/raw/job_data.csv" \
    --sample-size 20 \
    --limit 5
```

## ⏱️ Thời Gian Ước Tính

- **Test nhanh** (5 variations, 20 samples): ~5-10 phút
- **Full benchmark** (50 variations, 50 samples): **30-90 phút**
- **Full benchmark** (50 variations, 100 samples): **60-120 phút**

⚠️ **Lưu ý**: Thời gian phụ thuộc vào:
- Cấu hình máy (CPU, RAM, GPU)
- Số lượng samples
- Tốc độ internet (để download models lần đầu)

## 📊 Kết Quả

Sau khi hoàn tất, các file kết quả sẽ được lưu tại:

### 1. Kết Quả Benchmark Chi Tiết:
- `reports/benchmark_csv/benchmark_csv_results_*.json` - Dữ liệu JSON chi tiết
- `reports/benchmark_csv/benchmark_csv_results_*.csv` - Dữ liệu CSV để phân tích

### 2. Phân Tích Tối Ưu:
- `reports/benchmark_variations/optimization_analysis.json` - Phân tích chi tiết
- `reports/benchmark_variations/top_10_optimized.csv` - Top 10 variations

### 3. Summary:
- `reports/benchmark_csv/benchmark_summary.txt` - Tóm tắt kết quả

### 4. Logs:
- `reports/benchmark_csv/logs/full_benchmark_*.log` - Log chi tiết

## 🔍 Xem Kết Quả

### Xem Summary (Nhanh):
```bash
type reports\benchmark_csv\benchmark_summary.txt
```

### Xem Phân Tích Chi Tiết:
```bash
python scripts/analyze_benchmark_results.py --from-csv
```

### Xem CSV Results (Excel/Pandas):
Mở file `reports/benchmark_csv/benchmark_csv_results_*.csv` trong Excel hoặc:
```python
import pandas as pd
df = pd.read_csv('reports/benchmark_csv/benchmark_csv_results_*.csv')
print(df.head(10))
```

## 📈 Metrics Được Đo Lường

Mỗi variation sẽ được đánh giá trên các metrics:

1. **Quality Metrics**:
   - JD-Candidate Similarity (quan trọng nhất)
   - JD Self-Similarity
   - Candidate Self-Similarity

2. **Performance Metrics**:
   - Embedding Generation Time
   - Batch Processing Throughput
   - Memory Usage

3. **Optimization Score** (tổng hợp):
   - Quality: 60%
   - Speed: 25%
   - Memory: 15%

## 🏆 Bộ Parameter Tốt Nhất

Sau khi benchmark hoàn tất, script sẽ tự động:
1. ✅ Tính optimization score cho mỗi variation
2. ✅ Sắp xếp theo score
3. ✅ Hiển thị top 10 variations
4. ✅ Chọn và highlight bộ tốt nhất

## 💡 Tips

### Test Nhanh Trước:
```bash
# Test với 5 variations đầu tiên
python scripts/run_full_benchmark_50_variations.py --limit 5 --sample-size 20
```

### Chỉ Chạy Một Số Variations:
```bash
# Chỉ chạy variations 1-10 (2 models đầu tiên)
python scripts/run_full_benchmark_50_variations.py \
    --variations 1 2 3 4 5 6 7 8 9 10 \
    --sample-size 50
```

### Tăng Sample Size để Chính Xác Hơn:
```bash
# Sample size 100 (chính xác hơn nhưng lâu hơn)
python scripts/run_full_benchmark_50_variations.py --sample-size 100
```

## ⚠️ Lưu Ý

1. **Memory Usage**: Mỗi model cần ~500MB-1GB RAM. Có thể chạy từng model một nếu RAM hạn chế.

2. **Internet Connection**: Lần đầu chạy cần download models (có thể vài GB).

3. **Thời Gian**: Full benchmark có thể mất 1-2 giờ. Nên chạy vào lúc không dùng máy.

4. **Kết Quả**: Script sẽ tự động lưu kết quả, không cần lo mất dữ liệu nếu dừng giữa chừng (có thể resume từ variation chưa chạy).

## 🆘 Troubleshooting

### Lỗi: "File not found"
- Kiểm tra đường dẫn file CSV có đúng không
- Đảm bảo files `candidates_dataset.csv` và `job_data.csv` tồn tại

### Lỗi: "Out of memory"
- Giảm `--sample-size` xuống 20 hoặc 10
- Chạy từng nhóm variations nhỏ hơn

### Lỗi: "Model download failed"
- Kiểm tra kết nối internet
- Có thể cần proxy nếu trong mạng nội bộ

## 📞 Hỗ Trợ

Nếu gặp vấn đề, kiểm tra:
1. Logs trong `reports/benchmark_csv/logs/`
2. File `benchmark_summary.txt` để xem tiến độ
3. Đảm bảo virtual environment đã được activate

