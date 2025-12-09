# 🔄 LUỒNG HOẠT ĐỘNG CỦA PROJECT

## 📋 TỔNG QUAN

**AI Job Recommendation Service** - Hệ thống khuyến nghị việc làm thông minh sử dụng semantic embeddings để match Job Descriptions (JD) với Candidates.

---

## 🏗️ KIẾN TRÚC TỔNG QUAN

```
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                      │
│  - RESTful endpoints                                        │
│  - Request/Response handling                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Service Layer (Business Logic)                 │
│  - Matching Service (Multi-Filter Pipeline)                 │
│  - Embedding Service                                        │
│  - Precompute Service                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│          Embedding Layer (Sentence Transformers)            │
│  - Multi-field embeddings (Title, Skills, Requirements)     │
│  - 50 Parameter Variations (10 models × 5 configs)         │
│  - Model selection & optimization                           │
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

## 🔄 CÁC LUỒNG CHÍNH

### 1. 📥 DATA PROCESSING WORKFLOW

```
Raw CSV Files
    ↓
[Validation]
    ↓ Kiểm tra format, columns, data quality
[Preprocessing]
    ↓ Làm sạch, normalize, handle NaN
[Filtering]
    ↓ Loại bỏ records không có skills/title
[Column Mapping]
    ↓ Tự động map columns từ nhiều format
    ↓
Processed CSV Files
    ↓
data/processed/
  - jd_processed.csv
  - candidates_dataset.csv
```

**Scripts:**
- `src/data_processing/jd_processor.py` - Xử lý JD data
- `src/data_processing/candidate_processor.py` - Xử lý Candidate data
- `scripts/filter_data_with_skills.py` - Lọc records có skills

---

### 2. 🎯 EMBEDDING GENERATION WORKFLOW

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
  - candidate_multi_embeddings
```

**Scripts:**
- `scripts/process_multi_field_embeddings.py` - Generate và store embeddings
- `src/embeddings/multi_field_generator.py` - Multi-field embedding generator

**Fields:**
- **JD**: title, skills, requirements
- **Candidate**: title (desired_job), skills, experience

---

### 3. 🔍 MATCHING WORKFLOW (Multi-Filter Pipeline)

```
Candidate ID
    ↓
[Load Candidate Embeddings from DB]
    ↓
    ├─ candidate_experience_emb
    ├─ candidate_skills_emb
    └─ candidate_title_emb
    ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 1: Experience/Requirement Matching                 │
│   - Filter 1000 jobs by experience/requirement similarity│
│   - Input: candidate_experience_emb                     │
│   - Output: Top 1000 jobs                               │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 2: Skills Matching                                 │
│   - Filter 100 jobs by skills similarity                │
│   - Input: candidate_skills_emb                         │
│   - Output: Top 100 jobs                                │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 3: Title Matching                                  │
│   - Filter top 10 jobs by title similarity              │
│   - Input: candidate_title_emb                          │
│   - Output: Top 10 jobs                                 │
└─────────────────────────────────────────────────────────┘
    ↓
[Calculate Combined Score]
    ↓ Weighted: Experience 40%, Skills 40%, Title 20%
[Sort by Combined Score]
    ↓
Top 10 Job Recommendations
```

**Service:**
- `src/services/multi_filter_matching_service.py` - Multi-filter matching logic

**Features:**
- Fallback logic nếu embedding invalid
- Dynamic weights dựa trên fields valid
- FAISS hoặc PostgreSQL search

---

### 4. 📊 BENCHMARK WORKFLOW

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
    ├─ Quality Metrics:
    │   ├─ JD-Candidate similarity
    │   ├─ JD self-similarity
    │   └─ Candidate self-similarity
    │
    ├─ Skill Matching (50% weight):
    │   ├─ Skill matching percentage
    │   └─ Skill similarity statistics
    │
    ├─ Title Matching (NEW):
    │   ├─ Title matching percentage
    │   └─ Title similarity statistics
    │
    ├─ Speed Metrics:
    │   ├─ Generation time
    │   └─ Batch processing speed
    │
    └─ Memory Metrics:
        └─ Memory usage
    ↓
[Calculate Optimization Score]
    ↓ Quality 30% + Skill Matching 50% + Speed 15% + Memory 5%
[Save Results]
    ↓
benchmark_csv_results_*.csv
    ↓
[Update Comparison CSV]
    ↓
benchmark_results_comparison.csv
    ↓
[Analysis & Recommendations]
    ↓
improvement_recommendations_*.json
```

**Scripts:**
- `scripts/benchmark_from_csv.py` - Core benchmark logic
- `scripts/run_full_optimization_benchmark.py` - Full benchmark runner
- `scripts/run_benchmark_minimal.py` - Minimal logging version
- `scripts/update_comparison_csv.py` - Update comparison CSV
- `scripts/analyze_benchmark_results.py` - Analyze results
- `scripts/analyze_all_variations.py` - Detailed analysis
- `scripts/analyze_improvements.py` - Improvement recommendations

**Metrics:**
- **Quality**: 30% weight
- **Skill Matching**: 50% weight (QUAN TRỌNG NHẤT)
- **Speed**: 15% weight
- **Memory**: 5% weight

---

### 5. 🚀 API WORKFLOW

```
HTTP Request
    ↓
[FastAPI Routes]
    ↓
    ├─ POST /api/match/candidate/{candidate_id}
    │   └─ → MultiFilterMatchingService.find_jobs_for_candidate()
    │
    ├─ POST /api/match/candidate-text
    │   └─ → MultiFilterMatchingService.find_jobs_for_candidate_text()
    │
    └─ GET /api/health
        └─ → Health check
    ↓
[Service Layer]
    ↓ Multi-filter pipeline
[Database/FAISS Search]
    ↓
[Response]
    ↓ JSON với job recommendations
```

**Files:**
- `src/api/main.py` - FastAPI application
- `src/api/routes.py` - API endpoints
- `src/api/schemas.py` - Request/Response schemas

---

## 📈 QUY TRÌNH HOÀN CHỈNH

### Setup & Initialization

```bash
# 1. Initialize Database Tables
python scripts/init_multi_field_tables.py

# 2. Process Data
python scripts/filter_data_with_skills.py  # Lọc records có skills

# 3. Generate Embeddings
python scripts/process_multi_field_embeddings.py \
    --jd-file data/filtered/jds_with_skills.csv \
    --batch-size 50

python scripts/process_multi_field_embeddings.py \
    --candidate-file data/filtered/candidates_with_skills.csv \
    --batch-size 50
```

### Benchmark & Optimization

```bash
# 1. Run Benchmark
python scripts/run_full_optimization_benchmark.py \
    --candidate-file data/filtered/candidates_with_skills.csv \
    --jd-file data/filtered/jds_with_skills.csv \
    --sample-size 5000

# 2. Analyze Results
python scripts/analyze_all_variations.py
python scripts/analyze_improvements.py

# 3. Monitor Progress
python scripts/monitor_benchmark_progress.py
```

### Matching & Recommendations

```bash
# Test Matching
python scripts/test_multi_filter_matching.py \
    --candidate-id "15001" \
    --top-k 10

# Test với Best Variation
python scripts/test_best_variation_samples.py --samples 5
```

---

## 🔑 CÁC COMPONENT CHÍNH

### 1. Data Processing
- **JDProcessor**: Xử lý JD data, validation, preprocessing
- **CandidateProcessor**: Xử lý Candidate data, validation, preprocessing
- **ColumnMapper**: Tự động map columns từ nhiều format

### 2. Embedding Generation
- **MultiFieldEmbeddingGenerator**: Tạo 3 embeddings riêng biệt
- **ParameterVariation**: 50 variations (10 models × 5 configs)
- **Model Variations**: 10 base models với different configs

### 3. Matching Service
- **MultiFilterMatchingService**: 3-step filter pipeline
- **FAISS Manager**: Fast vector search
- **Repository**: Database operations

### 4. Benchmark System
- **CSVBenchmark**: Benchmark từ CSV data
- **Analysis Tools**: Phân tích và đề xuất cải thiện
- **Comparison CSV**: So sánh tất cả variations

---

## 📊 DATA FLOW

```
┌──────────────┐
│  Raw CSV     │
│  (data/raw/) │
└──────┬───────┘
       │
       ↓
┌──────────────────┐
│  Data Processing │
│  - Validation    │
│  - Preprocessing │
│  - Filtering     │
└──────┬───────────┘
       │
       ↓
┌──────────────────┐
│  Processed CSV   │
│  (data/processed/)│
└──────┬───────────┘
       │
       ↓
┌──────────────────┐
│  Generate        │
│  Embeddings      │
│  (Multi-field)   │
└──────┬───────────┘
       │
       ↓
┌──────────────────┐
│  PostgreSQL      │
│  - Store         │
│  - Search        │
└──────┬───────────┘
       │
       ↓
┌──────────────────┐
│  Matching        │
│  (3-step filter) │
└──────┬───────────┘
       │
       ↓
┌──────────────────┐
│  Recommendations │
│  (Top 10 Jobs)   │
└──────────────────┘
```

---

## 🎯 MATCHING PIPELINE CHI TIẾT

### Step 1: Experience/Requirement Matching
- **Input**: Candidate experience embedding
- **Search**: JD requirement embeddings
- **Method**: Cosine similarity search
- **Output**: Top 1000 jobs
- **Weight**: 40% trong combined score

### Step 2: Skills Matching
- **Input**: Candidate skills embedding
- **Search**: JD skills embeddings (từ top 1000 của Step 1)
- **Method**: Cosine similarity search
- **Output**: Top 100 jobs
- **Weight**: 40% trong combined score

### Step 3: Title Matching
- **Input**: Candidate title embedding (desired_job)
- **Search**: JD title embeddings (từ top 100 của Step 2)
- **Method**: Cosine similarity search
- **Output**: Top 10 jobs
- **Weight**: 20% trong combined score

### Combined Score Calculation
```python
combined_score = (
    exp_sim * 0.4 +      # Experience/Requirement
    skills_sim * 0.4 +   # Skills
    title_sim * 0.2      # Title
)
```

---

## 🔧 BENCHMARK METRICS

### Quality Metrics (30% weight)
- JD-Candidate similarity mean/std/max/min
- JD self-similarity mean/std
- Candidate self-similarity mean/std

### Skill Matching (50% weight) - QUAN TRỌNG NHẤT
- Skill matching percentage (0-100%)
- Skill similarity mean/std/max/min

### Title Matching (NEW)
- Title matching percentage (0-100%)
- Title similarity mean/std/max/min

### Speed Metrics (15% weight)
- Avg generation time per text (ms)
- Batch time per text (ms)
- Embeddings per second

### Memory Metrics (5% weight)
- Memory usage (MB)

### Optimization Score
```
Score = Quality × 0.30 + 
        Skill Matching × 0.50 + 
        Speed × 0.15 + 
        Memory × 0.05
```

---

## 📁 FILE STRUCTURE

```
AI_SERVICE/
├── src/
│   ├── api/                    # FastAPI application
│   ├── data_processing/        # Data processors
│   ├── embeddings/            # Embedding generators
│   ├── services/              # Business logic
│   ├── database/              # Database layer
│   └── vector_search/         # FAISS managers
│
├── scripts/
│   ├── benchmark_from_csv.py           # Core benchmark
│   ├── run_full_optimization_benchmark.py  # Full benchmark
│   ├── process_multi_field_embeddings.py   # Generate embeddings
│   ├── test_multi_filter_matching.py       # Test matching
│   ├── analyze_benchmark_results.py        # Analyze results
│   └── [100+ utility scripts]
│
├── data/
│   ├── raw/                    # Raw CSV files
│   ├── processed/              # Processed CSV files
│   └── filtered/               # Filtered data (có skills)
│
├── reports/
│   ├── benchmark_csv/          # Benchmark results
│   └── benchmark_variations/   # Variation analysis
│
└── docs/                       # Documentation
```

---

## 🚀 QUICK START WORKFLOW

### 1. Setup
```bash
# Initialize database
python scripts/init_multi_field_tables.py
```

### 2. Process Data
```bash
# Filter data (chỉ records có skills)
python scripts/filter_data_with_skills.py

# Process embeddings
python scripts/process_multi_field_embeddings.py \
    --jd-file data/filtered/jds_with_skills.csv \
    --candidate-file data/filtered/candidates_with_skills.csv
```

### 3. Benchmark
```bash
# Run benchmark
python scripts/run_full_optimization_benchmark.py \
    --sample-size 5000
```

### 4. Test Matching
```bash
# Test với candidate ID
python scripts/test_multi_filter_matching.py \
    --candidate-id "15001"

# Test với best variation
python scripts/test_best_variation_samples.py --samples 5
```

### 5. Start API
```bash
python main.py
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

---

## 📝 LƯU Ý QUAN TRỌNG

1. **Data Quality**: Luôn filter records có skills trước khi benchmark
2. **Title Matching**: Cần có desired_job hoặc title trong candidate data
3. **Skill Matching**: Sử dụng Job Requirements làm skills nếu JD không có skills riêng
4. **Optimization Score**: Skill matching có trọng số cao nhất (50%)
5. **Multi-Filter Pipeline**: 3-step filtering để tăng độ chính xác

---

**Generated**: 2025-12-03
**Version**: 2.0.0 (với Skill Matching 50% và Title Matching)

