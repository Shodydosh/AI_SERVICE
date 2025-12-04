# Title Matching Validator

## 📋 Tổng Quan

`TitleMatchingValidator` là một service để kiểm tra và tăng cường mối quan hệ giữa **desired job** của candidate và **JD title** được đề xuất.

## 🎯 Mục Đích

- ✅ Đảm bảo các JD được đề xuất có title tương tự với desired job của candidate
- ✅ Boost score cho các JD có title matching cao
- ✅ Filter out các JD có title matching quá thấp

## 🔧 Cách Sử Dụng

### 1. Khởi Tạo

```python
from src.services.title_matching_validator import TitleMatchingValidator

validator = TitleMatchingValidator(
    min_title_similarity=0.4,      # Minimum threshold để giữ lại JD
    boost_threshold=0.6,            # Threshold để boost score
    boost_factor=1.2                 # Factor boost (1.2 = tăng 20%)
)
```

### 2. Validate và Filter

```python
# Filter các JD có title similarity < min_title_similarity
validated_matches = validator.validate_and_filter(
    job_matches=job_matches,
    candidate_title_emb=candidate_title_embedding,
    job_title_embeddings=job_title_embeddings_dict
)
```

### 3. Boost Title Matches

```python
# Boost score cho các JD có title similarity >= boost_threshold
boosted_matches = validator.boost_title_matches(
    job_matches=job_matches,
    candidate_title_emb=candidate_title_embedding,
    job_title_embeddings=job_title_embeddings_dict
)
```

### 4. Validate và Boost (Kết Hợp)

```python
# Kết hợp filter + boost
final_matches = validator.validate_and_boost(
    job_matches=job_matches,
    candidate_title_emb=candidate_title_embedding,
    job_title_embeddings=job_title_embeddings_dict
)
```

### 5. Lấy Thống Kê

```python
stats = validator.get_title_matching_stats(job_matches)
print(f"Average title similarity: {stats['avg_title_similarity']:.4f}")
print(f"Boosted jobs: {stats['boosted_count']}")
```

## 🔗 Tích Hợp Với MultiFilterMatchingService

`TitleMatchingValidator` đã được tích hợp vào `MultiFilterMatchingService`:

```python
from src.services.multi_filter_matching_service import MultiFilterMatchingService

# Khởi tạo với title validation enabled
matching_service = MultiFilterMatchingService(
    db=db,
    use_faiss=True,
    enable_title_validation=True,      # Enable title validation
    min_title_similarity=0.4,          # Minimum threshold
    title_boost_threshold=0.6,         # Boost threshold
    title_boost_factor=1.2             # Boost factor
)

# Tự động validate và boost khi tìm jobs
results = matching_service.find_jobs_for_candidate(
    candidate_id="candidate_123",
    top_k=10
)
```

## 📊 Parameters

| Parameter | Type | Default | Mô Tả |
|-----------|------|---------|-------|
| `min_title_similarity` | float | 0.4 | Minimum title similarity threshold (0-1). JD có similarity < threshold sẽ bị loại bỏ |
| `boost_threshold` | float | 0.6 | Title similarity threshold để boost score (0-1). JD có similarity >= threshold sẽ được boost |
| `boost_factor` | float | 1.2 | Factor để boost score (ví dụ: 1.2 = tăng 20%) |

## 🎯 Workflow

```
Job Matches (từ matching service)
    ↓
[Validate] → Filter JD có title similarity < min_title_similarity
    ↓
[Boost] → Boost score cho JD có title similarity >= boost_threshold
    ↓
Final Results (đã được validate và boost)
```

## 📈 Ví Dụ

### Input:
```python
job_matches = [
    {
        "job_id": "jd_1",
        "title": "Software Engineer",
        "similarity_score": 0.75,
        "field_similarities": {"title": 0.65}  # High title similarity
    },
    {
        "job_id": "jd_2",
        "title": "Marketing Manager",
        "similarity_score": 0.70,
        "field_similarities": {"title": 0.35}  # Low title similarity
    }
]
```

### Sau khi validate và boost:
```python
# JD 1: Title similarity 0.65 >= 0.6 → Boosted (0.75 * 1.2 = 0.90)
# JD 2: Title similarity 0.35 < 0.4 → Filtered out

final_matches = [
    {
        "job_id": "jd_1",
        "title": "Software Engineer",
        "similarity_score": 0.90,  # Boosted!
        "field_similarities": {"title": 0.65},
        "title_boosted": True,
        "title_boost_factor": 1.2
    }
]
```

## 🔍 Methods

### `normalize_title(title: str) -> str`
Normalize title text: lowercase, remove extra spaces.

### `calculate_title_similarity(candidate_title_emb, jd_title_emb) -> float`
Tính title similarity giữa candidate desired job và JD title (cosine similarity).

### `validate_and_filter(...) -> List[Dict]`
Filter job matches dựa trên title similarity threshold.

### `boost_title_matches(...) -> List[Dict]`
Boost score cho các job matches có title similarity cao.

### `validate_and_boost(...) -> List[Dict]`
Kết hợp validate và boost: filter theo min threshold, sau đó boost các matches cao.

### `get_title_matching_stats(...) -> Dict`
Lấy thống kê về title matching trong job matches.

## 💡 Best Practices

1. **Set min_title_similarity phù hợp**: 
   - Quá cao (0.7+) → Có thể filter quá nhiều, ít kết quả
   - Quá thấp (0.2-) → Không filter được JD không liên quan
   - Khuyến nghị: 0.4-0.5

2. **Set boost_threshold hợp lý**:
   - Nên cao hơn min_title_similarity
   - Khuyến nghị: 0.6-0.7

3. **Boost factor**:
   - 1.1-1.2: Boost nhẹ
   - 1.3-1.5: Boost mạnh
   - Khuyến nghị: 1.2 (tăng 20%)

4. **Monitor stats**:
   - Kiểm tra `get_title_matching_stats()` để điều chỉnh parameters
   - Đảm bảo có đủ kết quả sau khi filter

## 🚀 Tích Hợp Vào API

Khi sử dụng API, title validation sẽ tự động được áp dụng nếu `enable_title_validation=True`:

```python
# API endpoint sẽ tự động sử dụng title validator
POST /api/v1/match/candidate-id
{
    "candidate_id": "candidate_123",
    "limit": 10
}
```

Kết quả sẽ bao gồm:
- Jobs đã được filter (chỉ giữ lại title similarity >= threshold)
- Jobs đã được boost (title similarity cao)
- Thống kê title matching trong logs

