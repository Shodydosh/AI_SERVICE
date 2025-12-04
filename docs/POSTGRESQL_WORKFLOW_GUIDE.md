# 🐘 Hướng Dẫn Workflow với PostgreSQL

## ✅ Xác Nhận PostgreSQL Setup

Database đang sử dụng: **PostgreSQL**

## 🔍 Kiểm Tra PostgreSQL Setup

Trước khi chạy workflow, kiểm tra PostgreSQL đã sẵn sàng:

```bash
python scripts/check_postgresql_setup.py
```

Script này sẽ kiểm tra:
1. ✅ Python modules (psycopg2, SQLAlchemy)
2. ✅ Database configuration
3. ✅ Database connection
4. ✅ PostgreSQL extensions
5. ✅ Existing tables

## 📋 Quy Trình Đầy Đủ

### Bước 1: Kiểm Tra PostgreSQL Connection

```bash
python scripts/check_postgresql_setup.py
```

**Kết quả mong đợi:**
```
✓ Connected successfully
✓ PostgreSQL version: PostgreSQL 14.x
✓ Current database: job_recommendation_db
```

### Bước 2: Khởi Tạo Database Tables

```bash
python scripts/init_multi_field_tables.py
```

Script này sẽ:
- Kết nối đến PostgreSQL
- Tạo 2 bảng:
  - `job_description_multi_embeddings`
  - `candidate_multi_embeddings`
- Sử dụng PostgreSQL ARRAY type cho embeddings
- Tạo GIN indexes cho tìm kiếm nhanh

**Kết quả mong đợi:**
```
✓ Detected PostgreSQL database
✓ Multi-field embedding tables created/verified successfully!
```

### Bước 3: Process Job Descriptions

```bash
python scripts/process_multi_field_embeddings.py --jd-file data/raw/job_data.csv --batch-size 50
```

### Bước 4: Process Candidates

```bash
python scripts/process_multi_field_embeddings.py --candidate-file data/raw/candidates_dataset.csv --batch-size 50
```

### Bước 5: Test Matching

```bash
python scripts/test_multi_filter_matching.py --candidate-id "15001" --top-k 10
```

## 🗄️ Cấu Trúc Bảng PostgreSQL

### Bảng: job_description_multi_embeddings

```sql
CREATE TABLE job_description_multi_embeddings (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(100) UNIQUE NOT NULL,
    title VARCHAR(500) NOT NULL,
    skills TEXT,
    requirement TEXT,
    company VARCHAR(200),
    location VARCHAR(200),
    title_embedding FLOAT[] NOT NULL,      -- PostgreSQL ARRAY
    skills_embedding FLOAT[] NOT NULL,     -- PostgreSQL ARRAY
    requirement_embedding FLOAT[] NOT NULL, -- PostgreSQL ARRAY
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE
);
```

**Indexes:**
- `idx_job_multi_title_embedding` (GIN index on title_embedding)
- `idx_job_multi_skills_embedding` (GIN index on skills_embedding)
- `idx_job_multi_requirement_embedding` (GIN index on requirement_embedding)

### Bảng: candidate_multi_embeddings

```sql
CREATE TABLE candidate_multi_embeddings (
    id SERIAL PRIMARY KEY,
    candidate_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(200),
    email VARCHAR(200),
    title VARCHAR(500),
    skills TEXT,
    experience TEXT,
    title_embedding FLOAT[] NOT NULL,       -- PostgreSQL ARRAY
    skills_embedding FLOAT[] NOT NULL,      -- PostgreSQL ARRAY
    experience_embedding FLOAT[] NOT NULL,  -- PostgreSQL ARRAY
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE
);
```

**Indexes:**
- `idx_candidate_multi_title_embedding` (GIN index)
- `idx_candidate_multi_skills_embedding` (GIN index)
- `idx_candidate_multi_experience_embedding` (GIN index)

## 🔧 PostgreSQL-Specific Features

### 1. ARRAY Type

Hệ thống sử dụng PostgreSQL `FLOAT[]` (ARRAY) để lưu embeddings:

```python
# Embeddings được lưu dưới dạng ARRAY
title_embedding = Column(ARRAY(Float), nullable=False)
```

**Ưu điểm:**
- Native PostgreSQL support
- Efficient storage
- GIN index hỗ trợ tìm kiếm nhanh
- Không cần extension bổ sung

### 2. GIN Indexes

GIN (Generalized Inverted Index) được sử dụng cho ARRAY columns:

```python
Index('idx_job_multi_title_embedding', 'title_embedding', postgresql_using='gin')
```

**Lợi ích:**
- Tìm kiếm nhanh trong arrays
- Hỗ trợ similarity search hiệu quả
- Tối ưu cho vector operations

### 3. Cosine Similarity

Similarity được tính bằng cosine similarity trên PostgreSQL arrays:

```python
# Trong repository
similarity = np.dot(query_vec, job_emb)
```

## 📊 Kiểm Tra Database

### Kiểm Tra Số Lượng Records

```sql
-- Số lượng jobs
SELECT COUNT(*) FROM job_description_multi_embeddings;

-- Số lượng candidates
SELECT COUNT(*) FROM candidate_multi_embeddings;

-- Xem một vài records
SELECT job_id, title, 
       array_length(title_embedding, 1) as embedding_dim
FROM job_description_multi_embeddings 
LIMIT 5;
```

### Kiểm Tra Indexes

```sql
-- Liệt kê indexes
SELECT 
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
AND tablename LIKE '%multi_embeddings%';
```

### Kiểm Tra Index Usage

```sql
-- Xem index size
SELECT 
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
AND tablename LIKE '%multi_embeddings%';
```

## 🚀 Chạy Toàn Bộ Workflow

### Cách 1: Batch File (Windows)

```bash
.\run_workflow.bat
```

### Cách 2: Python Script

```bash
python scripts/execute_workflow.py
```

### Cách 3: Từng Bước

```bash
# 1. Check PostgreSQL
python scripts/check_postgresql_setup.py

# 2. Init tables
python scripts/init_multi_field_tables.py

# 3. Process data
python scripts/process_multi_field_embeddings.py --jd-file data/raw/job_data.csv --batch-size 50
python scripts/process_multi_field_embeddings.py --candidate-file data/raw/candidates_dataset.csv --batch-size 50

# 4. Test
python scripts/test_multi_filter_matching.py --candidate-id "15001"
```

## ⚠️ Troubleshooting PostgreSQL

### Lỗi: "could not connect to server"

**Nguyên nhân:** PostgreSQL service không chạy

**Giải pháp:**
```bash
# Windows
Get-Service -Name postgresql*
Start-Service postgresql-x64-14  # Thay version của bạn

# Linux/Mac
sudo systemctl start postgresql
```

### Lỗi: "database does not exist"

**Giải pháp:**
```sql
-- Tạo database
CREATE DATABASE job_recommendation_db;

-- Hoặc từ command line
createdb job_recommendation_db
```

### Lỗi: "permission denied for schema public"

**Giải pháp:**
```sql
-- Grant permissions
GRANT ALL ON SCHEMA public TO your_user;
GRANT ALL PRIVILEGES ON DATABASE job_recommendation_db TO your_user;
```

### Lỗi: "relation already exists"

**Giải pháp:** Bảng đã tồn tại, có thể bỏ qua hoặc xóa và tạo lại:
```sql
DROP TABLE IF EXISTS job_description_multi_embeddings CASCADE;
DROP TABLE IF EXISTS candidate_multi_embeddings CASCADE;
```

### Kiểm Tra Connection String

Xem file `.env` hoặc `config/settings.py`:

```python
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=job_recommendation_db
```

## 📈 Performance Tips

1. **Batch Size:**
   - Nhỏ (10-50): An toàn, dễ debug
   - Vừa (100-200): Cân bằng
   - Lớn (500+): Nhanh nhưng cần RAM nhiều

2. **PostgreSQL Configuration:**
   - Tăng `shared_buffers` nếu có nhiều RAM
   - Tăng `work_mem` cho sorting/indexing
   - Enable `parallel_workers` nếu có multiple cores

3. **Index Maintenance:**
   - Chạy `VACUUM ANALYZE` sau khi insert nhiều data
   - Monitor index usage với `pg_stat_user_indexes`

## 🔗 Tài Liệu Liên Quan

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [PostgreSQL ARRAY Types](https://www.postgresql.org/docs/current/arrays.html)
- [GIN Indexes](https://www.postgresql.org/docs/current/gin.html)

## ✅ Checklist

Trước khi chạy workflow:

- [ ] PostgreSQL đang chạy
- [ ] Database `job_recommendation_db` đã tạo
- [ ] User có quyền CREATE TABLE
- [ ] Connection string đúng trong `.env` hoặc `config/settings.py`
- [ ] Python packages đã cài: `pip install psycopg2-binary sqlalchemy`
- [ ] Đã chạy `python scripts/check_postgresql_setup.py` thành công



