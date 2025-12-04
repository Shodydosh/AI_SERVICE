# 🚀 Hướng Dẫn Chạy Toàn Bộ Workflow và Debug

## 📋 Quy Trình Đầy Đủ

### Bước 1: Khởi Tạo Database Tables

```bash
python scripts/init_multi_field_tables.py
```

**Kết quả mong đợi:**
```
================================================================================
INITIALIZING MULTI-FIELD EMBEDDING TABLES
================================================================================
Connecting to database: localhost:5432/database_name (user: username)

Creating tables:
  - job_description_multi_embeddings
  - candidate_multi_embeddings

✓ Multi-field embedding tables created successfully!
```

**Nếu gặp lỗi:**
- Kiểm tra PostgreSQL đang chạy
- Kiểm tra cấu hình database trong `config/settings.py`
- Kiểm tra quyền CREATE TABLE của user

### Bước 2: Process Job Descriptions

```bash
python scripts/process_multi_field_embeddings.py --jd-file data/raw/job_data.csv --batch-size 50
```

**Kết quả mong đợi:**
```
================================================================================
INITIALIZING DATABASE TABLES
================================================================================
✓ Multi-field embedding tables created/verified successfully!

================================================================================
PROCESSING JOB DESCRIPTIONS
================================================================================
Processing batch 1 (50 records)...
Generating multi-field embeddings...
✓ Saved batch 1: 50 records
...
```

### Bước 3: Process Candidates

```bash
python scripts/process_multi_field_embeddings.py --candidate-file data/raw/candidates_dataset.csv --batch-size 50
```

**Kết quả mong đợi:**
```
================================================================================
PROCESSING CANDIDATES
================================================================================
Processing batch 1 (50 records)...
Generating multi-field embeddings...
✓ Saved batch 1: 50 records
...
```

### Bước 4: Test Matching

```bash
python scripts/test_multi_filter_matching.py --candidate-id "15001" --top-k 10
```

**Kết quả mong đợi:**
```
================================================================================
TESTING MULTI-FILTER MATCHING for Candidate ID: 15001
================================================================================
Step 1: Finding 1000 jobs by experience/requirement similarity...
✓ Step 1: Found 1000 jobs
Step 2: Filtering to 100 jobs by skills similarity...
✓ Step 2: Filtered to 100 jobs
Step 3: Filtering to top 10 jobs by title similarity...
✓ Step 3: Filtered to 10 final jobs

Found 10 matching jobs:
1. Job ID: 12345
   Title: Nhân Viên Kế Toán
   Similarity Score: 0.8750
...
```

## 🔄 Chạy Toàn Bộ Trong Một Lần

### Cách 1: Sử dụng Script Workflow

```bash
python scripts/run_full_workflow_3_fields.py --batch-size 50
```

Script này sẽ tự động:
1. Khởi tạo database tables
2. Process cả job descriptions và candidates
3. Test matching với candidate đầu tiên

### Cách 2: Sử dụng Script Debug (Có Output File)

```bash
python scripts/run_and_debug.py
```

Script này sẽ:
- Chạy toàn bộ workflow
- Lưu output vào file `workflow_debug_output.txt`
- Tự động skip processing nếu đã có data

Sau khi chạy, kiểm tra file:
```bash
type workflow_debug_output.txt  # Windows
cat workflow_debug_output.txt   # Linux/Mac
```

### Cách 3: Chạy Từng Bước (Khuyến Nghị)

```bash
# Bước 1: Init database
python scripts/init_multi_field_tables.py

# Bước 2: Process jobs
python scripts/process_multi_field_embeddings.py --jd-file data/raw/job_data.csv --batch-size 50

# Bước 3: Process candidates (lấy mẫu nhỏ trước)
python scripts/process_multi_field_embeddings.py --candidate-file data/raw/candidates_dataset.csv --batch-size 50

# Bước 4: Test matching
python scripts/test_multi_filter_matching.py --candidate-id "15001"
```

## 🐛 Debug Các Lỗi Thường Gặp

### Lỗi 1: "relation does not exist"

**Triệu chứng:**
```
sqlalchemy.exc.ProgrammingError: relation "candidate_multi_embeddings" does not exist
```

**Giải pháp:**
```bash
python scripts/init_multi_field_tables.py
```

### Lỗi 2: "No module named 'src'"

**Triệu chứng:**
```
ModuleNotFoundError: No module named 'src'
```

**Giải pháp:**
Đảm bảo đang ở thư mục gốc của project:
```bash
cd E:\4. CODE\AI_SERVICE
python scripts/init_multi_field_tables.py
```

### Lỗi 3: Database Connection Failed

**Triệu chứng:**
```
psycopg2.OperationalError: could not connect to server
```

**Giải pháp:**
1. Kiểm tra PostgreSQL đang chạy:
   ```bash
   # Windows
   Get-Service -Name postgresql*
   
   # Linux/Mac
   sudo systemctl status postgresql
   ```

2. Kiểm tra cấu hình trong `config/settings.py`:
   ```python
   DATABASE_URL = "postgresql://user:password@localhost:5432/database"
   ```

### Lỗi 4: "No matching jobs found"

**Triệu chứng:**
```
No jobs found in step 1
```

**Giải pháp:**
1. Kiểm tra đã process job descriptions chưa:
   ```bash
   python scripts/process_multi_field_embeddings.py --jd-file data/raw/job_data.csv
   ```

2. Kiểm tra số lượng jobs trong database:
   ```python
   from src.database.connection import SessionLocal
   from src.database.models import JobDescriptionMultiEmbedding
   
   db = SessionLocal()
   count = db.query(JobDescriptionMultiEmbedding).count()
   print(f"Jobs in DB: {count}")
   ```

### Lỗi 5: Candidate Not Found

**Triệu chứng:**
```
Candidate 15001 not found in database
```

**Giải pháp:**
1. Kiểm tra candidate ID có đúng không
2. Kiểm tra đã process candidates chưa:
   ```bash
   python scripts/process_multi_field_embeddings.py --candidate-file data/raw/candidates_dataset.csv
   ```

3. Liệt kê các candidate IDs có sẵn:
   ```python
   from src.database.connection import SessionLocal
   from src.database.models import CandidateMultiEmbedding
   
   db = SessionLocal()
   candidates = db.query(CandidateMultiEmbedding).limit(10).all()
   for c in candidates:
       print(c.candidate_id)
   ```

## 🔍 Kiểm Tra Trạng Thái Hệ Thống

### Kiểm Tra Database Tables

```python
from src.database.connection import SessionLocal
from src.database.models import JobDescriptionMultiEmbedding, CandidateMultiEmbedding

db = SessionLocal()

# Đếm số records
jd_count = db.query(JobDescriptionMultiEmbedding).count()
cand_count = db.query(CandidateMultiEmbedding).count()

print(f"Job descriptions: {jd_count}")
print(f"Candidates: {cand_count}")

# Xem một vài records
if jd_count > 0:
    first_job = db.query(JobDescriptionMultiEmbedding).first()
    print(f"First job: {first_job.job_id} - {first_job.title}")

if cand_count > 0:
    first_cand = db.query(CandidateMultiEmbedding).first()
    print(f"First candidate: {first_cand.candidate_id} - {first_cand.title}")

db.close()
```

### Kiểm Tra Embeddings

```python
from src.database.connection import SessionLocal
from src.database.models import CandidateMultiEmbedding

db = SessionLocal()
candidate = db.query(CandidateMultiEmbedding).first()

if candidate:
    print(f"Title embedding length: {len(candidate.title_embedding)}")
    print(f"Skills embedding length: {len(candidate.skills_embedding)}")
    print(f"Experience embedding length: {len(candidate.experience_embedding)}")

db.close()
```

## 📝 Checklist Trước Khi Chạy

- [ ] PostgreSQL đang chạy
- [ ] Database connection config đúng trong `config/settings.py`
- [ ] CSV files tồn tại:
  - [ ] `data/raw/job_data.csv`
  - [ ] `data/raw/candidates_dataset.csv`
- [ ] Đã cài đặt tất cả dependencies: `pip install -r requirements.txt`
- [ ] Đang ở thư mục gốc của project

## 🚀 Quick Start Command

Chạy lệnh này để khởi tạo và test nhanh:

```bash
# Windows PowerShell
cd "E:\4. CODE\AI_SERVICE"
python scripts/init_multi_field_tables.py
python scripts/process_multi_field_embeddings.py --jd-file data/raw/job_data.csv --batch-size 10
python scripts/test_multi_filter_matching.py --candidate-id "15001" --top-k 5
```

## 📊 Monitor Progress

Khi process datasets lớn, có thể theo dõi progress bằng cách:

1. Kiểm tra số records đã lưu:
   ```sql
   SELECT COUNT(*) FROM job_description_multi_embeddings;
   SELECT COUNT(*) FROM candidate_multi_embeddings;
   ```

2. Xem log files (nếu có) trong thư mục `logs/`

3. Chạy với batch size nhỏ hơn để thấy progress thường xuyên hơn:
   ```bash
   python scripts/process_multi_field_embeddings.py --batch-size 10
   ```

## ⚡ Performance Tips

1. **Batch Size**: 
   - Nhỏ (10-50): Tốn thời gian nhưng dễ debug
   - Vừa (100-200): Cân bằng
   - Lớn (500+): Nhanh nhưng cần RAM nhiều

2. **FAISS Index**: 
   - Sau khi process data, có thể build FAISS index để search nhanh hơn
   - Chỉ cần thiết khi có nhiều data (>10k records)

3. **Process từng phần**:
   - Test với sample nhỏ trước
   - Sau đó process toàn bộ dataset

## 🆘 Cần Trợ Giúp?

Nếu vẫn gặp lỗi, kiểm tra:
1. File log nếu có
2. Python version (nên dùng Python 3.8+)
3. Dependencies đã cài đầy đủ
4. Database connection settings


