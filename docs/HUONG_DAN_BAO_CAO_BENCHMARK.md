# Hướng Dẫn Sử Dụng Dữ Liệu Benchmark Để Viết Báo Cáo

## Tổng Quan

Sau khi chạy benchmark, bạn sẽ có đầy đủ dữ liệu để viết báo cáo chi tiết. Tất cả dữ liệu được lưu tự động với logging chi tiết.

## Files Được Tạo Sau Benchmark

### 1. Log Files (Chi Tiết)
**Location**: `reports/benchmark_variations/logs/benchmark_*.log`

**Nội dung**:
- Tất cả các bước thực hiện
- Thời gian cho từng variation
- Lỗi nếu có (với stack trace)
- Metrics chi tiết cho từng variation
- Thông tin về model loading, memory usage

**Sử dụng**: Để hiểu chi tiết quá trình benchmark, debug lỗi, và phân tích performance.

### 2. JSON Results (Dữ Liệu Chi Tiết)
**Location**: `reports/benchmark_variations/benchmark_results_*.json`

**Nội dung**: Tất cả metrics dưới dạng JSON structured data

**Cấu trúc dữ liệu**:
```json
{
  "variation_id": 1,
  "variation_name": "SimCSE_Vietnamese_v1_bs32_norm",
  "model_name": "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
  "base_name": "SimCSE_Vietnamese",
  "dimension": 768,
  "batch_size": 32,
  "normalize": true,
  "use_tokenization": true,
  "jd_single_avg_time": 0.1234,
  "jd_batch_throughput": 45.67,
  "cross_similarity_mean": 0.789,
  "memory_usage_mb": 512.34,
  ...
}
```

**Sử dụng**: 
- Import vào Python/Pandas để phân tích
- Tạo visualizations
- So sánh metrics giữa các variations

### 3. CSV Results (Dễ Phân Tích)
**Location**: `reports/benchmark_variations/benchmark_results_*.csv`

**Nội dung**: Cùng dữ liệu như JSON nhưng dạng CSV

**Sử dụng**:
- Mở trong Excel/Google Sheets
- Filter và sort dễ dàng
- Tạo charts trực tiếp

### 4. Markdown Report (Báo Cáo Tự Động)
**Location**: `reports/benchmark_variations/benchmark_report_*.md`

**Nội dung**:
- Rankings theo composite score
- Detailed metrics theo từng model
- Recommendations
- Summary statistics

**Sử dụng**: 
- Copy vào báo cáo của bạn
- Làm base để viết báo cáo chi tiết hơn

## Các Metrics Có Sẵn

### Performance Metrics
- `jd_single_avg_time`: Thời gian trung bình tạo embedding cho 1 JD (giây)
- `jd_single_min_time`: Thời gian tối thiểu
- `jd_single_max_time`: Thời gian tối đa
- `jd_single_std_time`: Độ lệch chuẩn
- `jd_batch_time`: Thời gian xử lý batch
- `jd_batch_throughput`: Số embeddings/giây (batch)
- `candidate_single_avg_time`: Tương tự cho candidate
- `candidate_batch_throughput`: Tương tự cho candidate

### Quality Metrics
- `cross_similarity_mean`: Độ tương đồng trung bình JD-Candidate
- `cross_similarity_std`: Độ lệch chuẩn
- `cross_similarity_min/max`: Min/Max values
- `top_5_similarity_mean`: Độ tương đồng trung bình của top 5 matches
- `jd_self_similarity_mean`: Độ tương đồng giữa các JDs (diversity)
- `candidate_self_similarity_mean`: Độ tương đồng giữa các candidates

### Embedding Quality Metrics
- `embedding_mean_norm`: Norm trung bình của embeddings
- `embedding_std_norm`: Độ lệch chuẩn của norm
- `embedding_mean_variance`: Variance trung bình
- `embedding_min/max/mean`: Min/Max/Mean values

### Resource Metrics
- `memory_usage_mb`: Bộ nhớ sử dụng (MB)
- `memory_after_mb`: Tổng bộ nhớ sau khi load model

### Composite Score
- `composite_score`: Tổng hợp (speed × 0.3 + quality × 0.5 + throughput × 0.2)

## Cách Phân Tích Dữ Liệu

### 1. Sử dụng Python/Pandas

```python
import pandas as pd
import json

# Load CSV
df = pd.read_csv('reports/benchmark_variations/benchmark_results_*.csv')

# Hoặc load JSON
with open('reports/benchmark_variations/benchmark_results_*.json') as f:
    data = json.load(f)
df = pd.DataFrame(data)

# Phân tích theo model
by_model = df.groupby('base_name').agg({
    'jd_single_avg_time': 'mean',
    'cross_similarity_mean': 'mean',
    'jd_batch_throughput': 'mean'
})

# Tìm best variation cho mỗi model
best_by_model = df.loc[df.groupby('base_name')['composite_score'].idxmax()]

# So sánh parameter variations
param_comparison = df.groupby(['base_name', 'batch_size', 'normalize']).agg({
    'jd_single_avg_time': 'mean',
    'cross_similarity_mean': 'mean'
})
```

### 2. Sử dụng Excel/Google Sheets

1. Mở file CSV
2. Tạo Pivot Tables để phân tích:
   - Group by `base_name` và `batch_size`
   - Tính average của các metrics
3. Tạo Charts:
   - Bar chart: So sánh throughput giữa các models
   - Scatter plot: Quality vs Speed
   - Line chart: Batch size vs Performance

### 3. Tạo Visualizations

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Quality vs Speed scatter plot
plt.scatter(df['jd_single_avg_time'], df['cross_similarity_mean'])
plt.xlabel('Time (s)')
plt.ylabel('Similarity')
plt.title('Quality vs Speed')

# Batch size comparison
sns.boxplot(data=df, x='batch_size', y='jd_batch_throughput')
```

## Cấu Trúc Báo Cáo Đề Xuất

### 1. Executive Summary
- Tổng số variations được benchmark
- Best overall variation
- Key findings

### 2. Methodology
- Sample size
- Test data description
- Metrics được đo
- Benchmark process

### 3. Results by Model
- So sánh 10 base models
- Best parameter config cho mỗi model
- Performance comparison

### 4. Results by Parameters
- Ảnh hưởng của batch size
- Ảnh hưởng của normalization
- Trade-offs giữa speed và quality

### 5. Detailed Analysis
- Top performers
- Worst performers
- Anomalies và explanations

### 6. Recommendations
- Best variation cho production
- Best variation cho speed
- Best variation cho quality
- Configuration recommendations

### 7. Appendices
- Full results table
- Log excerpts
- Code snippets

## Ví Dụ Phân Tích

### So Sánh Batch Sizes

```python
# Tìm ảnh hưởng của batch size
batch_analysis = df.groupby('batch_size').agg({
    'jd_batch_throughput': 'mean',
    'jd_single_avg_time': 'mean',
    'memory_usage_mb': 'mean'
})
```

### So Sánh Normalization

```python
# So sánh normalize vs không normalize
norm_analysis = df.groupby('normalize').agg({
    'cross_similarity_mean': 'mean',
    'jd_single_avg_time': 'mean'
})
```

### Best Configuration per Model

```python
# Tìm best config cho mỗi model
best_configs = df.loc[df.groupby('base_name')['composite_score'].idxmax()]
```

## Tips Viết Báo Cáo

1. **Sử dụng Visualizations**: Charts giúp dễ hiểu hơn text
2. **Highlight Key Findings**: Làm nổi bật những phát hiện quan trọng
3. **Explain Trade-offs**: Giải thích trade-offs giữa speed và quality
4. **Provide Context**: So sánh với baseline (variation 1)
5. **Include Recommendations**: Đưa ra recommendations cụ thể
6. **Show Data**: Include tables với key metrics
7. **Discuss Limitations**: Nêu limitations của benchmark

## Next Steps

Sau khi có dữ liệu:
1. Phân tích dữ liệu theo các sections trên
2. Tạo visualizations
3. Viết báo cáo với structure đề xuất
4. Run optimization script để chọn best variation
5. Apply best variation vào production

## Files Reference

- Logs: `reports/benchmark_variations/logs/benchmark_*.log`
- JSON: `reports/benchmark_variations/benchmark_results_*.json`
- CSV: `reports/benchmark_variations/benchmark_results_*.csv`
- Report: `reports/benchmark_variations/benchmark_report_*.md`

