# Benchmark với Parameter Variations

## Tổng Quan

Hệ thống benchmark hiện tại hỗ trợ **5 biến thể tham số cho mỗi model**, tạo ra tổng cộng **50 variations** (10 models × 5 parameter configs).

## 5 Parameter Variations

Mỗi model sẽ được test với 5 cấu hình tham số khác nhau:

1. **v1_bs32_norm**: Standard - batch_size=32, normalize=True
2. **v2_bs64_norm**: Large batch - batch_size=64, normalize=True  
3. **v3_bs128_norm**: Very large batch - batch_size=128, normalize=True
4. **v4_bs32_norm_false**: No normalization - batch_size=32, normalize=False
5. **v5_bs16_norm**: Small batch - batch_size=16, normalize=True

## 10 Base Models

1. **SimCSE_Vietnamese** - `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base`
2. **Multilingual_MPNet** - `paraphrase-multilingual-mpnet-base-v2`
3. **Vietnamese_SBERT** - `keepitreal/vietnamese-sbert`
4. **MiniLM_Multilingual** - `paraphrase-multilingual-MiniLM-L12-v2`
5. **MPNet_Base** - `all-mpnet-base-v2`
6. **QA_MPNet** - `multi-qa-mpnet-base-dot-v1`
7. **MiniLM_L6** - `all-MiniLM-L6-v2`
8. **MPNet_ST** - `sentence-transformers/all-mpnet-base-v2`
9. **Paraphrase_MiniLM** - `paraphrase-MiniLM-L6-v2`
10. **DistilUSE_Multilingual** - `distiluse-base-multilingual-cased`

## Cách Chạy Benchmark

### Option 1: Sử dụng helper script (Windows)

```bash
# PowerShell
.\scripts\run_benchmark.ps1

# Hoặc CMD
scripts\run_benchmark.bat
```

### Option 2: Chạy trực tiếp

```bash
# Activate virtual environment trước
# Windows PowerShell
venv\Scripts\Activate.ps1

# Hoặc CMD
venv\Scripts\activate.bat

# Sau đó chạy benchmark
python scripts/benchmark_model_variations.py --sample-size 50 --use-param-variations
```

### Option 3: Chạy chỉ một số variations

```bash
# Chạy chỉ 5 variations đầu tiên (model 1 với 5 parameter configs)
python scripts/benchmark_model_variations.py --sample-size 50 --variations 1 2 3 4 5

# Chạy tất cả variations của một model cụ thể
# Ví dụ: SimCSE_Vietnamese (variations 1-5)
python scripts/benchmark_model_variations.py --sample-size 50 --variations 1 2 3 4 5

# Ví dụ: Multilingual_MPNet (variations 6-10)
python scripts/benchmark_model_variations.py --sample-size 50 --variations 6 7 8 9 10
```

## Tham Số Command Line

- `--sample-size`: Số lượng samples để test (mặc định: 100)
- `--variations`: Danh sách variation IDs cụ thể (mặc định: tất cả)
- `--use-param-variations`: Sử dụng parameter variations (mặc định: True)
- `--use-original`: Sử dụng 10 original variations thay vì parameter variations

## Kết Quả

Sau khi chạy, kết quả sẽ được lưu trong:
- `reports/benchmark_variations/benchmark_results_*.json`: Dữ liệu chi tiết
- `reports/benchmark_variations/benchmark_report_*.md`: Báo cáo markdown

## Mapping Variations

Với 10 models và 5 parameter configs mỗi model:

- Variations 1-5: SimCSE_Vietnamese (5 configs)
- Variations 6-10: Multilingual_MPNet (5 configs)
- Variations 11-15: Vietnamese_SBERT (5 configs)
- Variations 16-20: MiniLM_Multilingual (5 configs)
- Variations 21-25: MPNet_Base (5 configs)
- Variations 26-30: QA_MPNet (5 configs)
- Variations 31-35: MiniLM_L6 (5 configs)
- Variations 36-40: MPNet_ST (5 configs)
- Variations 41-45: Paraphrase_MiniLM (5 configs)
- Variations 46-50: DistilUSE_Multilingual (5 configs)

## Lưu Ý

- Benchmark 50 variations có thể mất **rất nhiều thời gian** (có thể vài giờ)
- Mỗi model cần được load vào memory (~500MB-1GB mỗi model)
- Nên chạy với sample size nhỏ hơn (50) để test trước
- Có thể chạy từng model một để tránh out of memory

## Tối Ưu Hệ Thống

Sau khi có kết quả benchmark:

```bash
python scripts/optimize_system.py --criteria composite
```

Script sẽ tự động:
1. Load kết quả benchmark mới nhất
2. Chọn variation tốt nhất
3. Tạo optimized configuration
4. Cập nhật .env file

