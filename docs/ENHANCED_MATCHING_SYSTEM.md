# Enhanced Matching System - Tài Liệu

## 📋 Tổng Quan

Hệ thống matching đã được cải tiến với nhiều tính năng mới để tăng độ chính xác và hiệu suất.

## 🚀 Tính Năng Mới

### 1. Hybrid Search
- **Kết hợp semantic + keyword matching**
- Boost score dựa trên keyword matches
- Configurable keyword boost factor (default: 15%)

### 2. Reranking Layer
- **Cross-encoder reranking** sau FAISS search
- Chính xác hơn bi-encoder cho reranking
- Rerank top 100-1000 results từ FAISS

### 3. Dynamic Filtering
- **Điều chỉnh số lượng filter** theo data quality
- Tự động tăng/giảm filter sizes dựa trên:
  - Embedding quality (norm)
  - Text completeness
  - Result diversity

### 4. Contextual Embeddings
- **Composite embeddings** với prompt engineering
- Format: "[Title] working on [Skills] with [Requirements]"
- Tăng context awareness

### 5. Negative Signals
- **Deal-breakers và penalties**:
  - Salary mismatch
  - Location constraints
  - Industry mismatch
  - Seniority level misalignment

### 6. Caching Strategy
- **Redis cache** cho frequent queries
- Cache candidate recommendations (TTL: 1 hour)
- Cache FAISS search results (TTL: 30 minutes)
- In-memory fallback nếu không có Redis

### 7. Async Processing
- **Async operations** cho I/O
- Match multiple candidates concurrently
- Non-blocking operations

### 8. Database Optimization
- **Indexes**: Composite indexes, GIN indexes
- **Materialized Views**: Pre-computed scores
- **Table Analysis**: Query planner optimization

## 🔧 Sử Dụng

### Enhanced Matching Service

```python
from sqlalchemy.orm import Session
from src.services.enhanced_multi_filter_matching_service import EnhancedMultiFilterMatchingService
from src.database.connection import get_db

db: Session = next(get_db())

# Initialize với tất cả features
service = EnhancedMultiFilterMatchingService(
    db=db,
    use_faiss=True,
    use_hybrid_search=True,
    use_reranking=True,
    use_dynamic_filtering=True,
    use_contextual_embeddings=True,
    use_negative_signals=True,
    use_caching=True
)

# Find jobs for candidate
results = service.find_jobs_for_candidate(
    candidate_id="15001",
    top_k=10
)
```

### Tắt/Bật Features

```python
# Chỉ dùng hybrid search và caching
service = EnhancedMultiFilterMatchingService(
    db=db,
    use_hybrid_search=True,
    use_reranking=False,
    use_dynamic_filtering=False,
    use_contextual_embeddings=False,
    use_negative_signals=False,
    use_caching=True
)
```

## 📊 Pipeline Mới

```
1. Check Cache
   ↓
2. Title Matching (1000 jobs)
   - Hybrid Search (semantic + keyword)
   - Contextual Embeddings
   ↓
3. Skills Matching (100 jobs)
   - Dynamic Filtering (adjust size)
   ↓
4. Experience Matching (10 jobs)
   ↓
5. Reranking (Cross-Encoder)
   ↓
6. Negative Signals (Penalties)
   ↓
7. Title Validation & Boost
   ↓
8. Cache Results
   ↓
9. Return Top K
```

## ⚙️ Configuration

### Hybrid Search
- `keyword_boost`: 0.15 (15% boost)

### Reranking
- `top_k_rerank`: 100 (rerank top 100 từ FAISS)
- `model_name`: "cross-encoder/ms-marco-MiniLM-L-6-v2"

### Dynamic Filtering
- `min_quality_threshold`: 0.3
- `diversity_threshold`: 0.1

### Negative Signals
- `salary_mismatch_penalty`: 0.5
- `location_mismatch_penalty`: 0.3
- `industry_mismatch_penalty`: 0.4
- `seniority_mismatch_penalty`: 0.3

### Caching
- `default_ttl`: 3600 (1 hour)
- `faiss_cache_ttl`: 1800 (30 minutes)

## 🗄️ Database Optimization

Chạy script để tạo indexes và optimize:

```bash
python scripts/database_optimization.py
```

Indexes được tạo:
- `idx_jd_industry_location`: Composite index
- `idx_candidate_skills_gin`: GIN index cho skills
- `idx_jd_title_trgm`: Trigram index cho title search
- `idx_candidate_title_trgm`: Trigram index cho candidate title

## 📈 Performance

### Expected Improvements:
- **Accuracy**: +15-20% với hybrid search + reranking
- **Speed**: +30-40% với caching
- **Quality**: +10-15% với negative signals filtering

## 🔄 Migration

Để migrate từ `MultiFilterMatchingService` sang `EnhancedMultiFilterMatchingService`:

1. Update imports
2. Initialize với feature flags
3. Test với sample data
4. Monitor performance
5. Gradually enable features

## 📝 Notes

- Redis là optional (fallback to in-memory cache)
- Cross-encoder reranking chậm hơn nhưng chính xác hơn
- Dynamic filtering tự động adjust, không cần config
- Negative signals chỉ apply nếu có đủ data (salary, location, etc.)

