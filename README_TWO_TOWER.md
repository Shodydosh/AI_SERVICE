# 🏗️ Two-Tower Architecture - Quick Start Guide

## ✅ Đã Hoàn Thành

Hệ thống đã được chuyển đổi hoàn toàn sang **Two-Tower Architecture** với:

- ✅ **Job Tower Encoder** - Encode jobs thành 3 embeddings (title, skills, requirement)
- ✅ **Candidate Tower Encoder** - Encode candidates thành 3 embeddings (title, skills, experience)
- ✅ **TwoTowerFAISSManager** - Quản lý 6 FAISS indices riêng biệt
- ✅ **TwoTowerMatchingService** - 3-stage matching pipeline
- ✅ **API v2** - Endpoints mới cho Two-Tower

## 🚀 Quick Start

### 1. Tạo Database Schema

```bash
python scripts/migrate_to_two_tower_schema.py
```

### 2. Index Dữ Liệu

**Index một job:**
```bash
curl -X POST "http://localhost:8000/api/v2/index/job" \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "JD001",
    "title": "Senior Software Engineer",
    "skills": "Python, FastAPI, PostgreSQL",
    "requirement": "5+ years experience",
    "company": "Tech Corp",
    "location": "Ho Chi Minh City"
  }'
```

**Index một candidate:**
```bash
curl -X POST "http://localhost:8000/api/v2/index/candidate" \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_id": "CAND001",
    "title": "Software Engineer",
    "skills": "Python, FastAPI, PostgreSQL",
    "experience": "5 years at Company X",
    "name": "Nguyen Van A",
    "email": "a@example.com"
  }'
```

### 3. Build FAISS Indices

```bash
python scripts/batch_reindex_two_tower.py
```

### 4. Tìm Jobs cho Candidate

```bash
curl -X POST "http://localhost:8000/api/v2/search/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_id": "CAND001",
    "top_k": 10,
    "weights": {
      "title": 0.2,
      "skills": 0.4,
      "experience": 0.4
    }
  }'
```

### 5. Tìm Candidates cho Job

```bash
curl -X POST "http://localhost:8000/api/v2/search/candidates" \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "JD001",
    "top_k": 10
  }'
```

### 6. Health Check

```bash
curl http://localhost:8000/api/v2/health
```

## 📁 Cấu Trúc Files Mới

```
src/
├── embeddings/
│   ├── job_tower_encoder.py          # Job Tower Encoder
│   └── candidate_tower_encoder.py    # Candidate Tower Encoder
├── vector_search/
│   └── two_tower_faiss_manager.py   # FAISS Manager (6 indices)
├── services/
│   └── two_tower_matching_service.py # Matching Service (3-stage)
├── database/
│   ├── models.py                      # Two-Tower models
│   └── two_tower_repository.py        # Repository
└── api/
    ├── two_tower_routes.py           # API Routes v2
    └── two_tower_schemas.py          # Pydantic schemas
```

## 🔄 Migration từ Old System

Nếu bạn có data trong old tables (`job_description_multi_embeddings`, `candidate_multi_embeddings`), có thể migrate:

```python
# Script để migrate (cần implement)
from src.database.models import JobDescriptionMultiEmbedding, CandidateMultiEmbedding
from src.database.models import JobDescriptionTwoTower, CandidateTwoTower

# Copy data từ old tables sang new tables
# (embeddings giữ nguyên, chỉ đổi table name)
```

## 📊 API Endpoints

### v2 Endpoints

- `POST /api/v2/search/jobs` - Tìm jobs cho candidate
- `POST /api/v2/search/candidates` - Tìm candidates cho job
- `POST /api/v2/index/job` - Index một job
- `POST /api/v2/index/candidate` - Index một candidate
- `POST /api/v2/reindex` - Reindex toàn bộ
- `GET /api/v2/health` - Health check

### Legacy v1 Endpoints

Đã được comment out trong `src/api/main.py`. Có thể xóa hoàn toàn nếu không cần.

## 🎯 Next Steps

1. **Migrate existing data** từ old tables sang new tables
2. **Build FAISS indices** với data mới
3. **Test matching** với sample queries
4. **Monitor performance** (latency, recall@10)
5. **Tune weights** dựa trên evaluation

## 📝 Notes

- **Model**: Sử dụng `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base` (768-dim)
- **FAISS**: HNSW index với M=32, ef_search=128
- **Weights**: Default title=0.2, skills=0.4, experience=0.4
- **Indices location**: `indices/two_tower/`

## 🐛 Troubleshooting

**FAISS indices not found:**
- Chạy `python scripts/batch_reindex_two_tower.py` để build indices

**No results from search:**
- Kiểm tra xem đã index jobs/candidates chưa
- Kiểm tra embeddings có valid không (không phải zero vectors)

**Import errors:**
- Đảm bảo đã xóa các file cũ và không còn import từ old modules


