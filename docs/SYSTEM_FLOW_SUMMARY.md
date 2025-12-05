# 🔄 TÓM TẮT LUỒNG HOẠT ĐỘNG HỆ THỐNG

## 📋 TỔNG QUAN

**AI Job Recommendation Service** - Hệ thống khuyến nghị việc làm thông minh sử dụng semantic embeddings để match Job Descriptions (JD) với Candidates theo pipeline 3 bước.

---

## 🏗️ KIẾN TRÚC TỔNG QUAN

```
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                      │
│  - RESTful endpoints                                        │
│  - Scheduler service (chạy mỗi 12 giờ)                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Service Layer (Business Logic)                 │
│  - Multi-Filter Matching Service (3-step pipeline)          │
│  - Embedding Service                                        │
│  - Precompute Service                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│          Embedding Layer (Sentence Transformers)            │
│  - Multi-field embeddings (Title, Skills, Requirements)     │
│  - 50 Parameter Variations (10 models × 5 configs)         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│            Database Layer (PostgreSQL + FAISS)              │
│  - Multi-field embedding storage                            │
│  - Vector similarity search                                │
│  - FAISS indices for fast search                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 5 LUỒNG CHÍNH

### 1. 📥 LUỒNG XỬ LÝ DỮ LIỆU (Data Processing)

```
Raw CSV Files (data/raw/)
    ↓
[Validation & Preprocessing]
    ↓ Kiểm tra format, columns, data quality
    ↓ Làm sạch, normalize, handle NaN
[Filtering]
    ↓ Loại bỏ records không có skills/title
[Column Mapping]
    ↓ Tự động map columns từ nhiều format
    ↓
Processed CSV Files (data/processed/)
    ↓
Filtered CSV Files (data/filtered/)
    - jds_with_skills.csv
    - candidates_with_skills.csv
```

**Scripts liên quan:**
- `src/data_processing/jd_processor.py` - Xử lý JD data
- `src/data_processing/candidate_processor.py` - Xử lý Candidate data
- `scripts/filter_data_with_skills.py` - Lọc records có skills

---

### 2. 🎯 LUỒNG TẠO EMBEDDING (Embedding Generation)

```
Processed CSV
    ↓
[Load Data]
    ↓ JDProcessor / CandidateProcessor
[Generate Multi-Field Embeddings]
    ↓
    ├─ Title Embedding (JD Title / Candidate Desired Job)
    ├─ Skills Embedding (JD Skills / Candidate Skills)
    └─ Requirement Embedding (JD Requirements / Candidate Experience)
    ↓
[Store in PostgreSQL]
    ↓
Database Tables:
  - job_description_multi_embeddings
    ├─ job_title_emb
    ├─ job_skills_emb
    └─ job_requirements_emb
  - candidate_multi_embeddings
    ├─ candidate_title_emb
    ├─ candidate_skills_emb
    └─ candidate_experience_emb
```

**Scripts:**
- `scripts/process_multi_field_embeddings.py` - Generate và store embeddings
- `src/embeddings/multi_field_generator.py` - Multi-field embedding generator

---

### 3. 🔍 LUỒNG MATCHING (Multi-Filter Pipeline) - QUAN TRỌNG NHẤT

Đây là luồng chính khi người dùng yêu cầu tìm việc làm:

```
Candidate ID hoặc Candidate Text
    ↓
[Load Candidate Embeddings from DB]
    ↓
    ├─ candidate_experience_emb
    ├─ candidate_skills_emb
    └─ candidate_title_emb
    ↓
┌─────────────────────────────────────────────────────────┐
│ BƯỚC 1: Experience/Requirement Matching                │
│   - Tìm 1000 jobs khớp nhất về experience/requirement   │
│   - Input: candidate_experience_emb                     │
│   - Search: JD requirement embeddings                   │
│   - Output: Top 1000 jobs                               │
│   - Weight: 40% trong combined score                    │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ BƯỚC 2: Skills Matching                                 │
│   - Từ 1000 jobs trên, tìm 100 jobs khớp nhất về skills │
│   - Input: candidate_skills_emb                         │
│   - Search: JD skills embeddings (từ top 1000)          │
│   - Output: Top 100 jobs                                │
│   - Weight: 40% trong combined score                    │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ BƯỚC 3: Title Matching                                  │
│   - Từ 100 jobs trên, tìm 10 jobs khớp nhất về title    │
│   - Input: candidate_title_emb                          │
│   - Search: JD title embeddings (từ top 100)            │
│   - Output: Top 10 jobs                                 │
│   - Weight: 20% trong combined score                    │
└─────────────────────────────────────────────────────────┘
    ↓
[Calculate Combined Score]
    ↓ Weighted: Experience 40% + Skills 40% + Title 20%
[Title Validation & Boosting]
    ↓ Nếu title similarity > 0.6 → boost score lên 20%
[Sort by Combined Score]
    ↓
Top 10 Job Recommendations
```

**Service:**
- `src/services/multi_filter_matching_service.py` - Multi-filter matching logic

**Tính năng đặc biệt:**
- Fallback logic nếu embedding invalid
- Dynamic weights dựa trên fields valid
- Title validation và boosting
- FAISS hoặc PostgreSQL search

**Công thức Combined Score:**
```python
combined_score = (
    exp_sim * 0.4 +      # Experience/Requirement (40%)
    skills_sim * 0.4 +   # Skills (40%)
    title_sim * 0.2      # Title (20%)
)
```

---

### 4. 🚀 LUỒNG API (API Workflow)

Khi người dùng gọi API:

```
HTTP Request
    ↓
[FastAPI Routes] (src/api/routes.py)
    ↓
    ├─ POST /api/v1/multi-filter/match/candidate
    │   └─ → MultiFilterMatchingService.find_jobs_for_candidate_text()
    │
    ├─ POST /api/v1/multi-filter/recommend/jobs
    │   └─ → MultiFilterMatchingService.find_jobs_for_candidate()
    │
    ├─ POST /api/v1/multi-filter/process/jd-dataset
    │   └─ → MultiFieldEmbeddingService.process_jd_dataset()
    │
    ├─ POST /api/v1/multi-filter/process/candidate-dataset
    │   └─ → MultiFieldEmbeddingService.process_candidate_dataset()
    │
    └─ GET /api/v1/scheduler/status
        └─ → Kiểm tra trạng thái scheduler
    ↓
[Service Layer]
    ↓ Multi-filter pipeline
[Database/FAISS Search]
    ↓
[Response]
    ↓ JSON với job recommendations
```

**Entry Point:**
- `main.py` → Chạy uvicorn server
- `src/api/main.py` → FastAPI application với scheduler

**Scheduler tự động:**
- Chạy mỗi 12 giờ
- Regenerate embeddings và recompute recommendations
- Đảm bảo dữ liệu luôn cập nhật

---

### 5. 📊 LUỒNG BENCHMARK (Benchmark Workflow)

Để tối ưu và so sánh models:

```
50 Parameter Variations
    ↓
[For each variation:]
    ↓
[Load Test Data]
    ↓ Filtered (chỉ records có skills), sample_size=5000
[Generate Embeddings]
    ↓ JD embeddings + Candidate embeddings
[Calculate Metrics]
    ↓
    ├─ Quality Metrics (30%):
    │   ├─ JD-Candidate similarity
    │   ├─ JD self-similarity
    │   └─ Candidate self-similarity
    │
    ├─ Skill Matching (50% - QUAN TRỌNG NHẤT):
    │   ├─ Skill matching percentage
    │   └─ Skill similarity statistics
    │
    ├─ Title Matching:
    │   ├─ Title matching percentage
    │   └─ Title similarity statistics
    │
    ├─ Speed Metrics (15%):
    │   ├─ Generation time
    │   └─ Batch processing speed
    │
    └─ Memory Metrics (5%):
        └─ Memory usage
    ↓
[Calculate Optimization Score]
    ↓ Quality 30% + Skill Matching 50% + Speed 15% + Memory 5%
[Save Results]
    ↓
benchmark_results_comparison.csv
```

**Scripts:**
- `scripts/run_full_optimization_benchmark.py` - Full benchmark runner
- `scripts/analyze_all_variations.py` - Detailed analysis

---

## 🔄 LUỒNG KHỞI ĐỘNG HỆ THỐNG

Khi chạy `python main.py`:

```
1. Start FastAPI Application
   ↓
2. Initialize Scheduler Service
   ↓
3. Start Scheduler
   ↓
4. Add Regeneration Job (chạy mỗi 12 giờ)
   ↓
5. Load FAISS Indices (nếu có)
   ↓
6. API sẵn sàng nhận requests
```

**Scheduler tự động:**
- Mỗi 12 giờ → Regenerate embeddings và recompute recommendations
- Đảm bảo dữ liệu luôn mới nhất

---

## 📈 QUY TRÌNH SETUP HOÀN CHỈNH

### Bước 1: Khởi tạo Database
```bash
python scripts/init_multi_field_tables.py
```

### Bước 2: Xử lý dữ liệu
```bash
# Lọc records có skills
python scripts/filter_data_with_skills.py

# Hoặc xử lý từ raw data
python scripts/data_pipeline.py --file data/raw/job_data.csv --type jd
```

### Bước 3: Tạo Embeddings
```bash
# Tạo embeddings cho JD
python scripts/process_multi_field_embeddings.py \
    --jd-file data/filtered/jds_with_skills.csv \
    --batch-size 50

# Tạo embeddings cho Candidate
python scripts/process_multi_field_embeddings.py \
    --candidate-file data/filtered/candidates_with_skills.csv \
    --batch-size 50
```

### Bước 4: Build FAISS Indices (tùy chọn)
```bash
# FAISS sẽ tự động build khi matching service khởi động
# Hoặc build thủ công bằng script
```

### Bước 5: Chạy API
```bash
python main.py
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### Bước 6: Test Matching
```bash
# Test với candidate ID
python scripts/test_multi_filter_matching.py \
    --candidate-id "15001" \
    --top-k 10
```

---

## 🎯 ĐIỂM QUAN TRỌNG

1. **Multi-Filter Pipeline**: Hệ thống sử dụng 3 bước lọc để tăng độ chính xác:
   - Bước 1: Lọc 1000 jobs (Experience)
   - Bước 2: Lọc 100 jobs (Skills)
   - Bước 3: Lọc 10 jobs (Title)

2. **Scheduler tự động**: Chạy mỗi 12 giờ để regenerate embeddings

3. **FAISS Index**: Tăng tốc độ search đáng kể

4. **Title Validation**: Tự động boost score nếu title khớp tốt

5. **Weighted Scoring**: Experience 40%, Skills 40%, Title 20%

---

## 🔧 CÁC COMPONENT CHÍNH

### 1. Data Processing
- **JDProcessor**: Xử lý JD data
- **CandidateProcessor**: Xử lý Candidate data
- **ColumnMapper**: Tự động map columns

### 2. Embedding Generation
- **MultiFieldEmbeddingGenerator**: Tạo 3 embeddings riêng biệt
- **ParameterVariation**: 50 variations để benchmark

### 3. Matching Service
- **MultiFilterMatchingService**: 3-step filter pipeline
- **FAISS Manager**: Fast vector search
- **Repository**: Database operations

### 4. Scheduler Service
- **SchedulerService**: Quản lý scheduled jobs
- Tự động regenerate embeddings mỗi 12 giờ

---

**Version**: 1.0.0
**Last Updated**: 2025-01-XX


