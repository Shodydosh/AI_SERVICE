# HƯỚNG DẪN DEMO TẤT CẢ FEATURES

## TỔNG QUAN

Tài liệu này hướng dẫn demo tất cả features của hệ thống AI Job Recommendation Service. Tất cả thông tin được lấy từ codebase hiện tại.

---

## 1. KHỞI ĐỘNG HỆ THỐNG

### 1.1. Khởi động API Server

**File:** `main.py`

```bash
python main.py
```

Hoặc sử dụng uvicorn trực tiếp:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Thông tin:**
- API sẽ chạy tại: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- Version: 2.0.0

**File:** `src/api/main.py`
- FastAPI application với CORS middleware
- Routes được include từ `src.api.two_tower_routes` với prefix `/api/v2`
- Static files được mount tại `/static`

---

## 2. API ENDPOINTS

### 2.1. Health Check

**Endpoint:** `GET /api/v2/health`

**File:** `src/api/two_tower_routes.py` (line 216-258)

**Cách test:**
```bash
curl http://localhost:8000/api/v2/health
```

**Response:**
```json
{
    "status": "healthy",
    "version": "2.0.0",
    "indices": {
        "job_title_index": "loaded" | "not_found",
        "job_skills_index": "loaded" | "not_found",
        "job_requirement_index": "loaded" | "not_found",
        "candidate_title_index": "loaded" | "not_found",
        "candidate_skills_index": "loaded" | "not_found",
        "candidate_experience_index": "loaded" | "not_found"
    },
    "database": "connected",
    "total_jobs": <int>,
    "total_candidates": <int>
}
```

---

### 2.2. Index Job

**Endpoint:** `POST /api/v2/index/job`

**File:** `src/api/two_tower_routes.py` (line 106-143)

**Request Schema:** `IndexJobRequest`
- `job_id`: str (required)
- `title`: str (required)
- `skills`: Optional[str]
- `requirement`: Optional[str]
- `company`: Optional[str]
- `location`: Optional[str]

**Cách test:**
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

**Response:**
```json
{
    "status": "success",
    "job_id": "JD001",
    "message": "Job indexed successfully"
}
```

**Chức năng:**
- Encode job thành 3 embeddings: title, skills, requirement
- Lưu vào database (JobDescriptionTwoTower table)
- Sử dụng `JobTowerEncoder` để tạo embeddings

---

### 2.3. Index Candidate

**Endpoint:** `POST /api/v2/index/candidate`

**File:** `src/api/two_tower_routes.py` (line 146-183)

**Request Schema:** `IndexCandidateRequest`
- `candidate_id`: str (required)
- `title`: Optional[str]
- `skills`: Optional[str]
- `experience`: Optional[str]
- `name`: Optional[str]
- `email`: Optional[str]

**Cách test:**
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

**Response:**
```json
{
    "status": "success",
    "candidate_id": "CAND001",
    "message": "Candidate indexed successfully"
}
```

**Chức năng:**
- Encode candidate thành 3 embeddings: title, skills, experience
- Lưu vào database (CandidateTwoTower table)
- Sử dụng `CandidateTowerEncoder` để tạo embeddings

---

### 2.4. Search Jobs for Candidate

**Endpoint:** `POST /api/v2/search/jobs`

**File:** `src/api/two_tower_routes.py` (line 33-67)

**Request Schema:** `JobSearchRequest`
- `candidate_id`: str (required)
- `top_k`: int (default: 10, range: 1-100)

**Cách test:**
```bash
curl -X POST "http://localhost:8000/api/v2/search/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_id": "CAND001",
    "top_k": 10
  }'
```

**Response:**
```json
{
    "total_matches": 10,
    "matches": [
        {
            "job_id": "JD001",
            "title": "Senior Software Engineer",
            "company": "Tech Corp",
            "location": "Ho Chi Minh City",
            "score": 0.8523
        },
        ...
    ]
}
```

**Chức năng:**
- Lấy candidate từ database
- Build candidate text: "Title: {title} | Skills: {skills} | Experience: {experience}"
- Encode candidate bằng TwoTowerModel
- Tính similarity với tất cả jobs
- Trả về top K jobs với score

---

### 2.5. Search Candidates for Job

**Endpoint:** `POST /api/v2/search/candidates`

**File:** `src/api/two_tower_routes.py` (line 70-103)

**Request Schema:** `CandidateSearchRequest`
- `job_id`: str (required)
- `top_k`: int (default: 10, range: 1-100)

**Cách test:**
```bash
curl -X POST "http://localhost:8000/api/v2/search/candidates" \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "JD001",
    "top_k": 10
  }'
```

**Response:**
```json
{
    "total_matches": 10,
    "matches": [
        {
            "candidate_id": "CAND001",
            "name": "Nguyen Van A",
            "email": "a@example.com",
            "score": 0.8523
        },
        ...
    ]
}
```

**Chức năng:**
- Lấy job từ database
- Build job text: "Title: {title} | Skills: {skills} | Requirements: {requirement}"
- Encode job bằng TwoTowerModel
- Tính similarity với tất cả candidates
- Trả về top K candidates với score

---

### 2.6. Reindex

**Endpoint:** `POST /api/v2/reindex`

**File:** `src/api/two_tower_routes.py` (line 186-213)

**Request Schema:** `ReindexRequest`
- `reindex_type`: str (default: "full", pattern: "^(full|incremental|job|candidate)$")
- `force`: bool (default: False)

**Cách test:**
```bash
curl -X POST "http://localhost:8000/api/v2/reindex" \
  -H "Content-Type: application/json" \
  -d '{
    "reindex_type": "full",
    "force": false
  }'
```

**Response:**
```json
{
    "status": "accepted",
    "reindex_id": 1,
    "message": "Reindex job started",
    "estimated_time_minutes": 30
}
```

**Lưu ý:** Hiện tại chỉ tạo tracking record, chưa chạy reindex thực tế (TODO trong code).

---

## 3. SCRIPTS DEMO

### 3.1. Test Two-Tower với Precomputed Embeddings

**File:** `scripts/test_two_tower_precomputed.py`

**Cách chạy:**
```bash
python scripts/test_two_tower_precomputed.py \
    --max-candidates 5 \
    --top-k 5 \
    --output two_tower_recommendations_5x5_final.txt
```

**Parameters:**
- `--max-candidates`: Số lượng candidates để test (default: 5)
- `--top-k`: Số lượng jobs đề xuất cho mỗi candidate (default: 10)
- `--output`: File output (default: "two_tower_precomputed_test.txt")

**Chức năng:**
- Load precomputed embeddings từ database (MultiFieldEmbeddingRepository)
- Tính combined embedding (average của title, skills, experience/requirement)
- Tính cosine similarity
- Apply rule matching (RuleMatcher)
- In kết quả với đầy đủ thông tin

**Output format:**
- Candidate info
- Top K job recommendations với:
  - Similarity score
  - Rule 1 (Title) score và status
  - Rule 2 (Skill) score và status
  - Final Decision (OK/NG)

---

### 3.2. Recommend Jobs for Candidates

**File:** `scripts/recommend_jobs_for_candidates.py`

**Cách chạy:**
```bash
python scripts/recommend_jobs_for_candidates.py \
    --max-candidates 10 \
    --top-k 10 \
    --output job_recommendations.txt \
    --model-path outputs_improved/best_model_improved.pt
```

**Parameters:**
- `--max-candidates`: Maximum number of candidates to process (default: 10)
- `--top-k`: Number of top jobs to recommend per candidate (default: 10)
- `--output`: Output file path (default: "job_recommendations.txt")
- `--model-path`: Path to model checkpoint (default: "outputs_improved/best_model_improved.pt")

**Chức năng:**
- Load TwoTowerModel từ checkpoint
- Initialize RuleMatcher
- Load candidates và jobs từ database
- Pre-compute job embeddings (tất cả jobs)
- Với mỗi candidate:
  - Encode candidate
  - Tính similarity với tất cả jobs
  - Lấy top K jobs
  - Apply rule matching
  - In kết quả

**Model:**
- `candidate_model_name`: "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
- `job_model_name`: "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
- `output_dim`: 768

---

### 3.3. Test Two-Tower với Vietnamese

**File:** `scripts/test_two_tower_with_vietnamese.py`

**Cách chạy:**
```bash
python scripts/test_two_tower_with_vietnamese.py \
    --max-candidates 5 \
    --top-k 10 \
    --output two_tower_test_output.txt
```

**Chức năng:**
- Tương tự như `test_two_tower_precomputed.py` nhưng sử dụng TwoTowerModel để encode real-time
- Hỗ trợ tiếng Việt

---

### 3.4. Process Multi-Field Embeddings

**File:** `scripts/process_multi_field_embeddings.py`

**Cách chạy:**
```bash
python scripts/process_multi_field_embeddings.py \
    --jd-file data/filtered/jds_with_skills.csv \
    --candidate-file data/filtered/candidates_with_skills.csv
```

**Chức năng:**
- Process CSV files và tạo embeddings
- Lưu vào database (MultiFieldEmbeddingRepository)
- Tạo 3 embeddings cho mỗi job/candidate: title, skills, requirement/experience

---

### 3.5. Batch Reindex Two-Tower

**File:** `scripts/batch_reindex_two_tower.py`

**Cách chạy:**
```bash
python scripts/batch_reindex_two_tower.py
```

**Chức năng:**
- Reindex tất cả jobs và candidates trong database
- Build FAISS indices nếu cần

---

### 3.6. Migrate to Two-Tower Schema

**File:** `scripts/migrate_to_two_tower_schema.py`

**Cách chạy:**
```bash
python scripts/migrate_to_two_tower_schema.py
```

**Chức năng:**
- Tạo database schema cho Two-Tower tables
- Migrate data từ old tables nếu có

---

### 3.7. Initialize Multi-Field Tables

**File:** `scripts/init_multi_field_tables.py`

**Cách chạy:**
```bash
python scripts/init_multi_field_tables.py
```

**Chức năng:**
- Khởi tạo database tables cho multi-field embeddings

---

### 3.8. Filter Data with Skills

**File:** `scripts/filter_data_with_skills.py`

**Cách chạy:**
```bash
python scripts/filter_data_with_skills.py
```

**Chức năng:**
- Lọc data chỉ giữ lại records có skills
- Output: `data/filtered/jds_with_skills.csv` và `data/filtered/candidates_with_skills.csv`

---

### 3.9. Train Two-Tower Model

**File:** `scripts/train_two_tower.py`

**Cách chạy:**
```bash
python scripts/train_two_tower.py
```

**Chức năng:**
- Train TwoTowerModel với training data
- Save model checkpoint

---

### 3.10. Evaluate Two-Tower

**File:** `scripts/evaluate_two_tower.py`

**Cách chạy:**
```bash
python scripts/evaluate_two_tower.py
```

**Chức năng:**
- Evaluate model performance
- Tính các metrics: recall@10, NDCG@10, precision@K, recall@K

---

### 3.11. Test JD to JD Matching

**File:** `scripts/test_jd_to_jd_matching.py`

**Cách chạy:**
```bash
python scripts/test_jd_to_jd_matching.py
```

**Chức năng:**
- Tìm similar jobs cho một job
- Sử dụng combined score: Title 50% + Skills 35% + Requirement 15%

---

### 3.12. Check Database Status

**File:** `scripts/check_database_status.py`

**Cách chạy:**
```bash
python scripts/check_database_status.py
```

**Chức năng:**
- Kiểm tra trạng thái database
- Đếm số lượng jobs và candidates

---

### 3.13. Visualize Embeddings (t-SNE)

**File:** `scripts/visualize_embeddings_tsne.py`

**Cách chạy:**
```bash
python scripts/visualize_embeddings_tsne.py
```

**Chức năng:**
- Visualize embeddings bằng t-SNE
- Tạo visualization files

---

### 3.14. Print Recommendations Details

**File:** `scripts/print_recommendations_details.py`

**Cách chạy:**
```bash
python scripts/print_recommendations_details.py
```

**Chức năng:**
- In chi tiết recommendations cho một candidate

---

### 3.15. Show 10 Samples with Recommendations

**File:** `scripts/show_10_samples_with_recommendations.py`

**Cách chạy:**
```bash
python scripts/show_10_samples_with_recommendations.py
```

**Chức năng:**
- Hiển thị 10 samples với recommendations

---

### 3.16. Generate Ground Truth Pairs

**File:** `scripts/generate_ground_truth_500_pairs.py`

**Cách chạy:**
```bash
python scripts/generate_ground_truth_500_pairs.py
```

**Chức năng:**
- Generate ground truth pairs cho evaluation
- Output: `ground_truth_500_pairs.csv`

---

### 3.17. Run Embedding Scheduler

**File:** `scripts/run_embedding_scheduler.py`

**Cách chạy:**
```bash
python scripts/run_embedding_scheduler.py
```

**Chức năng:**
- Chạy scheduler để regenerate embeddings tự động (mỗi 12 giờ)

---

## 4. FEATURES CHÍNH

### 4.1. Two-Tower Architecture

**Files:**
- `two_tower/model.py` - TwoTowerModel architecture
- `src/embeddings/job_tower_encoder.py` - Job Tower Encoder
- `src/embeddings/candidate_tower_encoder.py` - Candidate Tower Encoder

**Tính năng:**
- Encode jobs thành 3 embeddings: title, skills, requirement
- Encode candidates thành 3 embeddings: title, skills, experience
- Tính cosine similarity giữa candidate và job embeddings

---

### 4.2. Rule-Based Matching

**File:** `src/utils/rule_matcher.py`

**Tính năng:**
- **Rule 1: Title Match**
  - Threshold: 0.60 (60%)
  - Tính similarity bằng: token Jaccard, sequence similarity, TF-IDF, semantic similarity
  - Final score = max của tất cả metrics

- **Rule 2: Skill Match**
  - Threshold: 0.8
  - Match types: exact (+1.0), synonym (+0.8), pattern (+0.6), partial (+0.5), category (+0.7)
  - Hỗ trợ Vietnamese-English translation

- **Final Decision:**
  - OK nếu: title_score >= 0.60 HOẶC skill_score >= 0.8
  - NG nếu: cả 2 đều không đạt

---

### 4.3. Database Operations

**Files:**
- `src/database/two_tower_repository.py` - TwoTowerRepository
- `src/database/multi_field_repository.py` - MultiFieldEmbeddingRepository

**Tính năng:**
- CRUD operations cho jobs và candidates
- Lưu embeddings vào PostgreSQL
- Query với embeddings

---

### 4.4. Matching Service

**File:** `src/services/two_tower_matching_service.py`

**Class:** `TwoTowerMatchingService`

**Tính năng:**
- `find_jobs_for_candidate(candidate_id, top_k)` - Tìm jobs cho candidate
- `find_candidates_for_job(job_id, top_k)` - Tìm candidates cho job
- Sử dụng TwoTowerModel để encode
- Tính cosine similarity
- Trả về top K results

---

### 4.5. Embedding Service

**File:** `src/services/embedding_service.py`

**Class:** `OptimizedEmbeddingService`

**Tính năng:**
- Cache embeddings (12 giờ TTL)
- Batch processing
- Lazy load encoders
- `get_candidate_embedding()` - Lấy embedding cho candidate
- `get_job_embedding()` - Lấy embedding cho job

---

### 4.6. FAISS Index

**File:** `two_tower/inference.py`

**Class:** `JobRecommender`

**Tính năng:**
- Build FAISS index từ precomputed embeddings
- HNSW index hoặc Flat index
- Fast similarity search

**Lưu ý:** FAISS index không được sử dụng trong `TwoTowerMatchingService` hiện tại, chỉ có trong `JobRecommender` class.

---

## 5. DEMO WORKFLOW

### Workflow 1: Index và Search

1. **Khởi động API:**
   ```bash
   python main.py
   ```

2. **Index một job:**
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

3. **Index một candidate:**
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

4. **Search jobs cho candidate:**
   ```bash
   curl -X POST "http://localhost:8000/api/v2/search/jobs" \
     -H "Content-Type: application/json" \
     -d '{
       "candidate_id": "CAND001",
       "top_k": 10
     }'
   ```

5. **Search candidates cho job:**
   ```bash
   curl -X POST "http://localhost:8000/api/v2/search/candidates" \
     -H "Content-Type: application/json" \
     -d '{
       "job_id": "JD001",
       "top_k": 10
     }'
   ```

---

### Workflow 2: Batch Processing

1. **Filter data:**
   ```bash
   python scripts/filter_data_with_skills.py
   ```

2. **Process embeddings:**
   ```bash
   python scripts/process_multi_field_embeddings.py \
     --jd-file data/filtered/jds_with_skills.csv \
     --candidate-file data/filtered/candidates_with_skills.csv
   ```

3. **Test với precomputed embeddings:**
   ```bash
   python scripts/test_two_tower_precomputed.py \
     --max-candidates 5 \
     --top-k 5 \
     --output two_tower_recommendations_5x5_final.txt
   ```

---

### Workflow 3: Recommendation với Rule Matching

1. **Recommend jobs cho candidates:**
   ```bash
   python scripts/recommend_jobs_for_candidates.py \
     --max-candidates 10 \
     --top-k 10 \
     --output job_recommendations.txt \
     --model-path outputs_improved/best_model_improved.pt
   ```

2. **Output sẽ bao gồm:**
   - Candidate info
   - Top K job recommendations
   - Two-Tower similarity score
   - Rule 1 (Title) score và status
   - Rule 2 (Skill) score và status
   - Final Decision (OK/NG)

---

## 6. CẤU TRÚC DỮ LIỆU

### 6.1. Request/Response Schemas

**File:** `src/api/two_tower_schemas.py`

**Các schemas:**
- `JobSearchRequest` - Request để search jobs
- `CandidateSearchRequest` - Request để search candidates
- `IndexJobRequest` - Request để index job
- `IndexCandidateRequest` - Request để index candidate
- `ReindexRequest` - Request để reindex
- `JobSearchResponse` - Response cho job search
- `CandidateSearchResponse` - Response cho candidate search
- `IndexResponse` - Response cho indexing
- `ReindexResponse` - Response cho reindex
- `HealthResponse` - Response cho health check
- `JobMatch` - Job match result
- `CandidateMatch` - Candidate match result

---

### 6.2. Database Models

**File:** `src/database/models.py`

**Models:**
- `JobDescriptionTwoTower` - Job với 3 embeddings
- `CandidateTwoTower` - Candidate với 3 embeddings
- `JobDescriptionMultiEmbedding` - Job với multi-field embeddings (old)
- `CandidateMultiEmbedding` - Candidate với multi-field embeddings (old)

---

## 7. CONFIGURATION

### 7.1. Settings

**File:** `config/settings.py`

**Các settings:**
- `API_HOST` - API host
- `API_PORT` - API port
- `EMBEDDING_MODEL` - Embedding model name
- `EMBEDDING_DIMENSION` - Embedding dimension
- `LOG_LEVEL` - Log level

---

## 8. LƯU Ý

- Tất cả thông tin trong document này được lấy từ codebase
- Không có thông tin nào được sáng tạo thêm
- Các scripts có thể có thêm parameters, xem help bằng `--help`
- Một số features có thể chưa hoàn thiện (có TODO trong code)

---

## 9. TROUBLESHOOTING

### Lỗi: Database connection failed
- Kiểm tra PostgreSQL đã chạy chưa
- Kiểm tra connection string trong `config/settings.py`

### Lỗi: Model not found
- Kiểm tra model checkpoint file tồn tại
- Default path: `outputs_improved/best_model_improved.pt`

### Lỗi: No candidates/jobs found
- Chạy script để process data: `scripts/process_multi_field_embeddings.py`
- Hoặc index qua API: `POST /api/v2/index/job` và `POST /api/v2/index/candidate`

### Lỗi: FAISS indices not found
- Chạy script để build indices: `scripts/batch_reindex_two_tower.py`
- Hoặc sử dụng API search (không cần FAISS)

---

## 10. TÀI LIỆU THAM KHẢO

- `README_TWO_TOWER.md` - Quick start guide
- `report_architecture.md` - Kiến trúc hệ thống
- `report_rules.md` - Rules matching
- `report_recommendation.md` - Recommendation system
- `docs/` - Các tài liệu khác

