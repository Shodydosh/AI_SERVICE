# Hướng Dẫn Sử Dụng Multi-Filter Matching với 3 Trường

## 📋 Tổng Quan

Hệ thống matching đã được implement lại để sử dụng **chỉ 3 trường** và **multi-filter pipeline**:

### 3 Trường Embedding

#### Đối với Job Description:
- **title**: Tiêu đề công việc
- **skills**: Kỹ năng yêu cầu
- **requirement**: Yêu cầu công việc

#### Đối với Candidate (CV):
- **title**: Vị trí mong muốn (desired_job)
- **skills**: Kỹ năng ứng viên
- **experience**: Kinh nghiệm làm việc

### Pipeline Multi-Filter

Pipeline matching được thực hiện qua 3 bước:

```
BƯỚC 1: Experience (CV) vs Requirement (Job)
    ↓ Tìm 1000 jobs khớp nhất
    
BƯỚC 2: Skills Matching
    ↓ Lọc xuống 100 jobs
    
BƯỚC 3: Title Matching
    ↓ Lọc top 10 jobs cuối cùng
```

## 🚀 Cách Sử Dụng

### 1. Process Dataset và Tạo Embeddings

Để xử lý dataset và tạo multi-field embeddings:

```bash
# Process job descriptions
python scripts/process_multi_field_embeddings.py --jd-file data/raw/job_data.csv --batch-size 100

# Process candidates
python scripts/process_multi_field_embeddings.py --candidate-file data/raw/candidates_dataset.csv --batch-size 100

# Process cả hai
python scripts/process_multi_field_embeddings.py --process-all --batch-size 100
```

### 2. Test Matching

#### Test với Candidate ID:

```bash
python scripts/test_multi_filter_matching.py --candidate-id "15001" --top-k 10
```

#### Test với Text Input:

```bash
python scripts/test_multi_filter_matching.py \
    --title "Nhân Viên Kế Toán" \
    --skills "Excel, Kế toán, Báo cáo tài chính" \
    --experience "5 năm kinh nghiệm làm kế toán tại công ty lớn" \
    --top-k 10
```

### 3. Sử Dụng trong Code

#### Tìm jobs cho candidate từ database:

```python
from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.services.multi_filter_matching_service import MultiFilterMatchingService

db: Session = next(get_db())
service = MultiFilterMatchingService(db, use_faiss=True)

# Tìm top 10 jobs cho candidate
results = service.find_jobs_for_candidate(candidate_id="15001", top_k=10)

for result in results:
    print(f"Job: {result['title']}")
    print(f"Similarity: {result['similarity_score']}")
    print(f"Experience: {result['field_similarities']['experience']}")
    print(f"Skills: {result['field_similarities']['skills']}")
    print(f"Title: {result['field_similarities']['title']}")
```

#### Tìm jobs cho candidate mới (từ text):

```python
results = service.find_jobs_for_candidate_text(
    title="Nhân Viên Kế Toán",
    skills="Excel, Kế toán, Báo cáo tài chính",
    experience="5 năm kinh nghiệm làm kế toán",
    top_k=10
)
```

## 📁 Cấu Trúc File

### Core Components

1. **`src/utils/three_field_extractor.py`**
   - `ThreeFieldExtractor`: Extract chính xác 3 trường từ CSV
   - Methods: `extract_candidate_fields()`, `extract_job_fields()`

2. **`src/embeddings/multi_field_generator.py`**
   - `MultiFieldEmbeddingGenerator`: Tạo 3 embeddings riêng biệt
   - Methods: `generate_job_embeddings()`, `generate_candidate_embeddings()`

3. **`src/services/multi_filter_matching_service.py`**
   - `MultiFilterMatchingService`: Service chính cho multi-filter matching
   - Pipeline: 1000 → 100 → 10

4. **`src/services/multi_field_embedding_service.py`**
   - `MultiFieldEmbeddingService`: Service để process và lưu embeddings

5. **`src/database/multi_field_repository.py`**
   - `MultiFieldEmbeddingRepository`: Repository cho multi-field embeddings

## 🔍 Chi Tiết Pipeline

### Bước 1: Experience vs Requirement (1000 jobs)

So sánh embedding của:
- **Candidate experience** với **Job requirement**

Tìm 1000 jobs có similarity cao nhất về experience/requirement.

### Bước 2: Skills Matching (100 jobs)

Từ 1000 jobs ở bước 1, so sánh:
- **Candidate skills** với **Job skills**

Lọc xuống 100 jobs có similarity cao nhất về skills.

### Bước 3: Title Matching (10 jobs)

Từ 100 jobs ở bước 2, so sánh:
- **Candidate title** (desired_job) với **Job title**

Lọc xuống top 10 jobs có similarity cao nhất về title.

### Kết Quả

Kết quả trả về bao gồm:
- `job_id`: ID công việc
- `title`: Tiêu đề
- `company`: Công ty
- `location`: Địa điểm
- `similarity_score`: Điểm tổng hợp (weighted average)
- `field_similarities`: Điểm chi tiết từng trường
  - `experience`: Điểm experience/requirement
  - `skills`: Điểm skills
  - `title`: Điểm title

## ⚙️ Cấu Hình

### Batch Size

Khi process dataset, có thể điều chỉnh batch size:

```bash
python scripts/process_multi_field_embeddings.py --process-all --batch-size 50
```

Batch size nhỏ hơn sẽ:
- Tiêu tốn ít bộ nhớ hơn
- Xử lý chậm hơn
- Phù hợp với máy có RAM thấp

Batch size lớn hơn sẽ:
- Xử lý nhanh hơn
- Tiêu tốn nhiều bộ nhớ hơn
- Phù hợp với máy có RAM cao

### FAISS Index

Service hỗ trợ FAISS index để tăng tốc search:

```python
service = MultiFilterMatchingService(db, use_faiss=True)  # Sử dụng FAISS
service = MultiFilterMatchingService(db, use_faiss=False)  # Sử dụng database search
```

## 📊 Ví Dụ Kết Quả

```
MULTI-FILTER MATCHING for Candidate: 15001
================================================================================
Step 1: Finding 1000 jobs by experience/requirement similarity...
✓ Step 1: Found 1000 jobs
Step 2: Filtering to 100 jobs by skills similarity...
✓ Step 2: Filtered to 100 jobs
Step 3: Filtering to top 10 jobs by title similarity...
✓ Step 3: Filtered to 10 final jobs

Found 10 matching jobs:

1. Job ID: 12345
   Title: Nhân Viên Kế Toán Tổng Hợp
   Company: Công ty ABC
   Location: Hà Nội
   Similarity Score: 0.8750
   Field Similarities:
     - Experience: 0.9200
     - Skills: 0.8500
     - Title: 0.8800
```

## 🔧 Troubleshooting

### Lỗi: Candidate không tìm thấy trong database

**Nguyên nhân**: Candidate chưa được process và lưu vào database.

**Giải pháp**: Chạy script process candidate dataset trước:
```bash
python scripts/process_multi_field_embeddings.py --candidate-file data/raw/candidates_dataset.csv
```

### Lỗi: Không có job nào được tìm thấy

**Nguyên nhân**: 
- Job dataset chưa được process
- Pipeline filter quá strict

**Giải pháp**: 
1. Kiểm tra đã process job dataset chưa
2. Kiểm tra candidate có đủ thông tin (experience, skills, title)

### Lỗi: Out of memory

**Nguyên nhân**: Batch size quá lớn.

**Giải pháp**: Giảm batch size:
```bash
python scripts/process_multi_field_embeddings.py --process-all --batch-size 50
```

## 📝 Notes

- Hệ thống chỉ sử dụng **3 trường**: title, skills, experience/requirement
- Pipeline luôn thực hiện **3 bước**: 1000 → 100 → 10
- Embeddings được normalize để tính cosine similarity
- Service hỗ trợ cả FAISS index và database search



