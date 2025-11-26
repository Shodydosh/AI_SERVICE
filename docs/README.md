# Tài Liệu Hệ Thống AI Job Recommendation Service

**Phiên bản:** 2.0.0  
**Cập nhật:** 2024-01-01

---

## 📑 Mục Lục

1. [Tổng Quan](#tổng-quan)
2. [Cài Đặt và Setup](#cài-đặt-và-setup)
3. [Kiến Trúc Hệ Thống](#kiến-trúc-hệ-thống)
4. [API Documentation](#api-documentation)
5. [FAISS Vector Search](#faiss-vector-search)
6. [Scheduled Jobs](#scheduled-jobs)
7. [Troubleshooting](#troubleshooting)
8. [Performance và Tối Ưu](#performance-và-tối-ưu)

> **📖 Xem thêm:** 
> - [Tóm Tắt Kiến Trúc](TOM_TAT_KIEN_TRUC.md) - Bản tóm tắt ngắn gọn về kiến trúc hệ thống
> - [Kiến Trúc Hệ Thống Chi Tiết](KIEN_TRUC_HE_THONG.md) - Tài liệu kiến trúc đầy đủ với sơ đồ và giải thích chi tiết từng thành phần

---

## Tổng Quan

Hệ thống **AI Job Recommendation Service** là một dịch vụ khuyến nghị việc làm thông minh sử dụng công nghệ **Semantic Search** và **Vector Similarity** để tự động kết nối ứng viên với các công việc phù hợp nhất.

### ⚡ Tính Năng Nổi Bật

- **FAISS HNSW**: Tìm kiếm vector siêu nhanh (10-100x nhanh hơn PostgreSQL)
- **Pre-computed Recommendations**: Query cực nhanh (< 10ms) với top 10 jobs đã tính sẵn
- **Scheduled Jobs**: Tự động regenerate embeddings và pre-compute mỗi 12 giờ
- **Cross-encoder Re-ranking**: Độ chính xác 90%+
- **Vietnamese SimCSE Model**: Tối ưu cho tiếng Việt (768 dimensions)

### 📊 Performance

| Method | Response Time | Use Case |
|--------|---------------|----------|
| Pre-computed (IDs only) | < 10ms | ⚡ Fastest - Chỉ cần job IDs |
| Pre-computed (with metadata) | ~50ms | Fast - Cần full job details |
| FAISS HNSW | 10-50ms | Fast - Fallback khi chưa pre-compute |
| PostgreSQL | 500-2000ms | Slow - Fallback cuối cùng |

---

## Cài Đặt và Setup

### 1. Yêu Cầu Hệ Thống

- Python 3.8+
- PostgreSQL 11+ với pgvector extension
- 4GB+ RAM (khuyến nghị 8GB+)
- 10GB+ disk space

### 2. Setup Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup PostgreSQL

#### Windows
1. Download PostgreSQL từ [postgresql.org](https://www.postgresql.org/download/windows/)
2. Cài đặt với default settings (port 5432)
3. Nhớ password cho user `postgres`

#### macOS
```bash
brew install postgresql@15
brew services start postgresql@15
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
```

#### Cài pgvector Extension

**Windows:**
1. Download pgvector từ [GitHub](https://github.com/pgvector/pgvector/releases)
2. Extract và compile (hoặc dùng pre-built binaries)
3. Copy vào PostgreSQL lib directory

**macOS:**
```bash
brew install pgvector
```

**Linux:**
```bash
git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

#### Tạo Database và Enable Extension

```bash
# Tạo database
createdb job_recommendation_db

# Kết nối và enable extension
psql job_recommendation_db -c "CREATE EXTENSION vector;"
```

### 5. Cấu Hình Environment

Tạo file `.env` trong project root:

```env
# Database
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=job_recommendation_db

# Embedding Model
EMBEDDING_MODEL=VoVanPhuc/sup-SimCSE-VietNamese-phobert-base
EMBEDDING_DIMENSION=768

# API
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO
```

### 6. Khởi Tạo Database

```bash
python scripts/init_db.py
```

### 7. Quick Start

```bash
# 1. Process datasets
python scripts/data_pipeline.py --file data/raw/job_data.csv --type jd
python scripts/data_pipeline.py --file data/raw/candidates_dataset.csv --type candidate

# 2. Generate embeddings
python scripts/generate_embeddings.py \
    --jd-file data/processed/jd_processed.csv \
    --candidate-file data/processed/candidate_processed.csv

# 3. Start service
python main.py
```

**Scheduler tự động start và chạy mỗi 12 giờ!**

---

## Kiến Trúc Hệ Thống

### 🏗️ Kiến Trúc Tổng Thể

```
┌─────────────────────────────────────────┐
│         CLIENT (Web/Mobile/API)         │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│      API LAYER (FastAPI)               │
│  - RESTful Endpoints                   │
│  - Request Validation                  │
│  - Scheduler Integration               │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│      SERVICE LAYER                     │
│  - MatchingService (Tìm kiếm)          │
│  - PrecomputeService (Pre-compute)     │
│  - SchedulerService (Scheduled jobs)  │
│  - EmbeddingService (Generate vectors) │
│  - RerankingService (Cải thiện độ chính xác)│
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│   VECTOR SEARCH (FAISS)                │
│  - HNSW Index (Fast Search)             │
│  - Vector Normalization                 │
│  - ID Mapping                           │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│      DATA LAYER                        │
│  - PostgreSQL (Embeddings + Metadata)  │
│  - Processed Recommendations Table    │
│  - FAISS Indices (Disk)               │
└─────────────────────────────────────────┘
```

### 📦 Các Thành Phần Chính

#### 1. API Layer (`src/api/`)
- FastAPI application với RESTful endpoints
- Request/response validation (Pydantic)
- Scheduler integration

#### 2. Service Layer (`src/services/`)
- **MatchingService**: Tìm kiếm việc làm, ưu tiên pre-computed
- **PrecomputeService**: Pre-compute top 10 jobs cho tất cả candidates
- **SchedulerService**: Quản lý scheduled jobs (12h interval)
- **EmbeddingService**: Generate embeddings từ text
- **RerankingService**: Cross-encoder re-ranking

#### 3. Vector Search (`src/vector_search/`)
- **FAISSIndexManager**: Quản lý FAISS indices (HNSW)
- Index persistence (save/load từ disk)
- Vector normalization cho cosine similarity

#### 4. Database (`src/database/`)
- **Models**: SQLAlchemy models cho embeddings và recommendations
- **Repository**: CRUD operations và batch processing
- **Connection**: Session management và connection pooling

### 🔄 Luồng Dữ Liệu

#### Luồng Query (Real-time)
```
User Request → API → Check Pre-computed?
    ├─ YES → Return (< 10ms) ⚡
    └─ NO → FAISS Search (10-50ms) → Re-ranking → Response
```

#### Luồng Scheduled Job (Background - 12h)
```
Every 12 Hours:
    1. Regenerate Embeddings
    2. Rebuild FAISS Indices
    3. Pre-compute Top 10 Jobs
    4. Save to processed_candidate_recommendations
```

---

## API Documentation

### Base URL
```
http://localhost:8000
```

### Interactive Docs
Visit `http://localhost:8000/docs` for Swagger UI.

### Matching Endpoints

#### 1. Get Job IDs (Fastest) ⚡
```bash
POST /api/v1/jobs/ids
Content-Type: application/json

{
  "candidate_id": "candidate_123",
  "limit": 10
}
```

**Response:** `< 10ms`
```json
{
  "candidate_id": "candidate_123",
  "job_ids": ["job_1", "job_2", "job_3", ...],
  "total": 10
}
```

#### 2. Match Candidate by ID
```bash
POST /api/v1/match/candidate-id
Content-Type: application/json

{
  "candidate_id": "candidate_123",
  "limit": 15
}
```

**Response:** `~50ms` (với pre-computed) hoặc `10-50ms` (FAISS fallback)

#### 3. Match Candidate by Text
```bash
POST /api/v1/match/candidate-text
Content-Type: application/json

{
  "candidate_text": "Python developer with 5 years experience...",
  "limit": 50
}
```

#### 4. Match Candidate with Detailed Input
```bash
POST /api/v1/match/candidate
Content-Type: application/json

{
  "name": "John Doe",
  "skills": "Python, Machine Learning, TensorFlow",
  "experience": "5 years in ML",
  "education": "Computer Science",
  "summary": "Experienced ML engineer",
  "limit": 15
}
```

### Processing Endpoints

#### Process JD Dataset
```bash
POST /api/v1/process/jd-dataset
Content-Type: application/json

{
  "file_path": "data/processed/jd_processed.csv",
  "file_type": "csv"
}
```

#### Process Candidate Dataset
```bash
POST /api/v1/process/candidate-dataset
Content-Type: application/json

{
  "file_path": "data/processed/candidate_processed.csv",
  "file_type": "csv"
}
```

### Scheduler Endpoints

#### Get Scheduler Status
```bash
GET /api/v1/scheduler/status
```

**Response:**
```json
{
  "is_running": true,
  "jobs": [
    {
      "id": "regeneration_job",
      "name": "Regenerate embeddings and recompute recommendations",
      "next_run_time": "2024-01-01T12:00:00",
      "trigger": "interval[0:12:00]"
    }
  ]
}
```

#### Trigger Pre-compute Manually
```bash
POST /api/v1/scheduler/precompute
```

### Utility Endpoints

#### Get Candidates List
```bash
GET /api/v1/candidates?limit=1000
```

#### Get Candidate Details
```bash
GET /api/v1/candidates/{candidate_id}
```

#### Health Check
```bash
GET /api/v1/health
```

---

## FAISS Vector Search

### Tổng Quan

FAISS (Facebook AI Similarity Search) được sử dụng để tăng tốc độ tìm kiếm từ **500-2000ms** (PostgreSQL) xuống còn **10-50ms** (FAISS HNSW).

### Các Loại Index

1. **Flat Index** - Chính xác 100%, chậm với dataset lớn (< 10K vectors)
2. **IVF Index** - Gần đúng, nhanh hơn (10K - 100K vectors)
3. **HNSW Index** ⭐ - Rất nhanh, được sử dụng (> 100K vectors)

### Cấu Hình HNSW

```python
{
    "M": 32,                    # Số kết nối mỗi node
    "ef_construction": 200,    # Độ chính xác khi xây dựng
    "ef_search": 64           # Độ chính xác khi tìm kiếm
}
```

### Sử Dụng

#### Build Index
```bash
python scripts/manage_faiss.py build --dataset-type jd --index-type HNSW
```

#### Load Index
```bash
python scripts/manage_faiss.py load --dataset-type jd --index-path indices/jd_index.faiss
```

#### Rebuild Index
```bash
python scripts/manage_faiss.py rebuild --dataset-type jd
```

### Performance

| Dataset Size | Search Time | Index Size |
|--------------|-------------|------------|
| 1K jobs | ~5ms | ~3 MB |
| 10K jobs | ~10-50ms | ~30 MB |
| 100K jobs | ~20-100ms | ~300 MB |

---

## Scheduled Jobs

### Tổng Quan

Hệ thống tự động chạy scheduled jobs mỗi **12 giờ** để:
1. Regenerate embeddings cho tất cả data
2. Rebuild FAISS indices
3. Pre-compute top 10 jobs cho tất cả candidates

### Tự Động Chạy

Scheduler tự động start khi FastAPI app khởi động:

```bash
python main.py
```

### Monitor

```bash
# Check status
GET /api/v1/scheduler/status

# Trigger manually
POST /api/v1/scheduler/precompute
```

### Performance

| Candidates | Pre-compute Time |
|------------|------------------|
| 1K | 3-5 phút |
| 5K | 15-25 phút |
| 10K | 30-60 phút |

### Cấu Hình

Để thay đổi interval, sửa trong `src/api/main.py`:

```python
scheduler_service.add_regeneration_job(
    hours=24,  # Thay đổi từ 12 sang 24 giờ
    ...
)
```

---

## Troubleshooting

### 1. Scheduler Không Chạy

**Kiểm tra:**
```bash
# Check status
curl http://localhost:8000/api/v1/scheduler/status

# Check logs
python main.py
```

**Giải pháp:**
- Đảm bảo APScheduler đã cài: `pip install apscheduler`
- Kiểm tra logs khi app start
- Restart app

### 2. Processed Recommendations Rỗng

**Giải pháp:**
```bash
# Trigger pre-compute manually
POST /api/v1/scheduler/precompute
```

### 3. FAISS Index Không Tìm Thấy

**Giải pháp:**
```bash
# Build index
python scripts/manage_faiss.py build --dataset-type jd
```

### 4. Database Connection Error

**Kiểm tra:**
- PostgreSQL đang chạy
- Database credentials trong `.env` đúng
- pgvector extension đã enable

**Giải pháp:**
```bash
# Test connection
python test_db_connection.py

# Enable extension
psql job_recommendation_db -c "CREATE EXTENSION vector;"
```

### 5. Memory Error

**Giải pháp:**
- Giảm batch_size trong PrecomputeService
- Sử dụng IVF index thay vì HNSW cho dataset rất lớn
- Tăng RAM hoặc sử dụng swap

### 6. Embedding Model Download Error

**Giải pháp:**
```bash
# Download model manually
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')"
```

---

## Performance và Tối Ưu

### Query Performance

**Với Pre-computed:**
- Job IDs only: **< 10ms**
- With metadata: **~50ms**

**Với FAISS:**
- HNSW: **10-50ms**
- PostgreSQL fallback: **500-2000ms**

### Memory Usage

- FAISS Index: ~30 MB per 10K vectors
- Processed Recommendations: ~1 MB per 1K candidates
- Total: ~60-100 MB cho dataset 10K

### Tối Ưu Hóa

1. **Sử dụng pre-computed recommendations** khi có thể
2. **FAISS HNSW** cho dataset > 10K vectors
3. **Batch processing** cho embeddings và pre-computation
4. **Database connection pooling**
5. **Index caching** (load từ disk)

### Best Practices

1. **Monitor performance** - Track response times
2. **Backup database** - Trước khi regenerate
3. **Test với sample data** - Trước khi scale
4. **Schedule jobs vào giờ thấp điểm**
5. **Regular maintenance** - Rebuild indices khi cần

---

## 📚 Tài Liệu Tham Khảo

- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [APScheduler Documentation](https://apscheduler.readthedocs.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [PostgreSQL pgvector](https://github.com/pgvector/pgvector)

---

## ✅ Kết Luận

Hệ thống **AI Job Recommendation Service** cung cấp:

- ⚡ **Performance**: Query cực nhanh (< 10ms với pre-computed)
- 🔄 **Automation**: Tự động regenerate và pre-compute mỗi 12 giờ
- 🎯 **Accuracy**: 90%+ với cross-encoder re-ranking
- 🇻🇳 **Vietnamese**: Tối ưu cho tiếng Việt
- 📈 **Scalable**: Xử lý hàng trăm nghìn jobs và candidates

**Phiên bản:** 2.0.0  
**Cập nhật:** 2024-01-01
