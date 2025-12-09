# BÁO CÁO HỆ THỐNG AI SERVICE

## MỤC LỤC

1. [TỔNG QUAN HỆ THỐNG](#1-tổng-quan-hệ-thống)
   1.1. Giới thiệu hệ thống
   1.2. Mục tiêu và phạm vi
   1.3. Công nghệ sử dụng

2. [KIẾN TRÚC HỆ THỐNG](#2-kiến-trúc-hệ-thống)
   2.1. Kiến trúc Two-Tower Architecture
   2.2. Cấu trúc phân lớp (Layered Architecture)
   2.3. Entry Point và Application Bootstrap

3. [MÔ HÌNH VÀ EMBEDDINGS](#3-mô-hình-và-embeddings)
   3.1. Two-Tower Model Architecture
   3.2. Embedding Encoders
   3.3. Embedding Model Selection
   3.4. Embedding Service và Caching

4. [HỆ THỐNG MATCHING VÀ TÌM KIẾM](#4-hệ-thống-matching-và-tìm-kiếm)
   4.1. Two-Tower Matching Service
   4.2. Vector Search với FAISS
   4.3. Rule-based Matching

5. [XỬ LÝ DỮ LIỆU](#5-xử-lý-dữ-liệu)
   5.1. Data Processing Layer
   5.2. Data Preprocessing Utilities
   5.3. Data Quality và Validation

6. [LỚP DATABASE](#6-lớp-database)
   6.1. Database Connection và Configuration
   6.2. Database Models
   6.3. Database Repositories
   6.4. Database Migrations

7. [API LAYER](#7-api-layer)
   7.1. FastAPI Application
   7.2. API Routes
   7.3. API Schemas

8. [SERVICES VÀ BUSINESS LOGIC](#8-services-và-business-logic)
   8.1. Matching Services
   8.2. Embedding Services
   8.3. Scheduler Services

9. [UTILITIES VÀ HELPERS](#9-utilities-và-helpers)
   9.1. Text Processing Utilities
   9.2. Embedding Utilities
   9.3. Explanation Utilities
   9.4. Data Utilities
   9.5. Logging Utilities

10. [TRAINING VÀ EVALUATION](#10-training-và-evaluation)
    10.1. Training Pipeline
    10.2. Evaluation Metrics
    10.3. Ground Truth Building

11. [KIẾN THỨC BỔ SUNG VÀ ĐỀ XUẤT CẢI TIẾN](#11-kiến-thức-bổ-sung-và-đề-xuất-cải-tiến)
    11.1. Lý thuyết về Two-Tower Architecture
    11.2. Lý thuyết về FAISS và Vector Search
    11.3. Lý thuyết về Hybrid Retrieval Systems
    11.4. Đề xuất cải tiến

12. [KẾT LUẬN](#12-kết-luận)

---

## 1. TỔNG QUAN HỆ THỐNG

### 1.1. Giới thiệu hệ thống

**Theo codebase hiện tại**, hệ thống AI Service là một dịch vụ khuyến nghị việc làm thông minh sử dụng kiến trúc Two-Tower Architecture để match Job Descriptions (JD) với Candidates. Hệ thống được xây dựng bằng Python với FastAPI framework, sử dụng PostgreSQL để lưu trữ dữ liệu và embeddings, FAISS cho vector search, và PyTorch cho neural network models.

**Entry Point**: File `main.py` khởi động FastAPI server với cấu hình từ `config/settings.py`:
- API Host: `0.0.0.0`
- API Port: `8000`
- Embedding Model: `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base`
- Embedding Dimension: `768`

### 1.2. Mục tiêu và phạm vi

Hệ thống cung cấp các chức năng chính:
- **Indexing**: Index jobs và candidates với 3 embeddings riêng biệt (title, skills, requirement/experience)
- **Search**: Tìm jobs cho candidates và candidates cho jobs
- **Matching**: Sử dụng Two-Tower model để tính similarity giữa candidates và jobs
- **Explainability**: Cung cấp giải thích cho kết quả matching (5 levels)

### 1.3. Công nghệ sử dụng

**Theo codebase hiện tại**, các công nghệ chính:
- **FastAPI**: Web framework cho API
- **PostgreSQL**: Database với ARRAY support cho embeddings
- **SQLAlchemy**: ORM cho database operations
- **PyTorch**: Deep learning framework
- **FAISS**: Vector similarity search library
- **Sentence Transformers**: Embedding generation
- **Vietnamese NLP**: pyvi cho tokenization tiếng Việt

---

## 2. KIẾN TRÚC HỆ THỐNG

### 2.1. Kiến trúc Two-Tower Architecture

**Theo codebase hiện tại**, hệ thống sử dụng Two-Tower Architecture với:

#### 2.1.1. Job Tower Encoder
- **File**: `src/embeddings/job_tower_encoder.py`
- **Chức năng**: Encode jobs thành 3 embeddings riêng biệt:
  - `title_embedding`: Embedding cho job title
  - `skills_embedding`: Embedding cho job skills
  - `requirement_embedding`: Embedding cho job requirements
- **Preprocessing**: 
  - Vietnamese text được translate sang English (nếu cần)
  - Text được normalize và truncate (title: 200 chars, skills: 1000 chars, requirement: 2000 chars)
  - Sử dụng Vietnamese tokenization (pyvi) nếu model yêu cầu

#### 2.1.2. Candidate Tower Encoder
- **File**: `src/embeddings/candidate_tower_encoder.py`
- **Chức năng**: Encode candidates thành 3 embeddings riêng biệt:
  - `title_embedding`: Embedding cho candidate title (desired job)
  - `skills_embedding`: Embedding cho candidate skills
  - `experience_embedding`: Embedding cho candidate experience
- **Preprocessing**: Tương tự Job Tower Encoder

#### 2.1.3. Multi-field Embeddings
**Theo codebase hiện tại**, mỗi record (job hoặc candidate) có 3 embeddings riêng biệt thay vì 1 embedding duy nhất. Điều này cho phép:
- Matching theo từng field riêng biệt
- Weighted scoring với weights khác nhau cho từng field
- Fine-grained similarity analysis

### 2.2. Cấu trúc phân lớp (Layered Architecture)

**Theo codebase hiện tại**, hệ thống được tổ chức thành các layers:

#### 2.2.1. API Layer (FastAPI)
- **File**: `src/api/main.py`, `src/api/two_tower_routes.py`
- **Chức năng**: 
  - RESTful API endpoints
  - Request/Response handling
  - Error handling
- **Endpoints chính**:
  - `POST /api/v2/search/jobs`: Tìm jobs cho candidate
  - `POST /api/v2/search/candidates`: Tìm candidates cho job
  - `POST /api/v2/index/job`: Index một job mới
  - `POST /api/v2/index/candidate`: Index một candidate mới
  - `POST /api/v2/reindex`: Trigger reindex operation
  - `GET /api/v2/health`: Health check

#### 2.2.2. Service Layer (Business Logic)
- **Files**: 
  - `src/services/two_tower_matching_service.py`: Matching logic
  - `src/services/embedding_service.py`: Embedding generation và caching
  - `src/services/embedding_cache_manager.py`: Cache management
  - `src/services/embedding_scheduler.py`: Scheduled embedding updates
- **Chức năng**: Business logic, matching algorithms, caching

#### 2.2.3. Embedding Layer (Sentence Transformers)
- **Files**: 
  - `src/embeddings/job_tower_encoder.py`
  - `src/embeddings/candidate_tower_encoder.py`
  - `src/embeddings/model_selector.py`
- **Chức năng**: Generate embeddings từ text

#### 2.2.4. Database Layer (PostgreSQL)
- **Files**: 
  - `src/database/connection.py`: Database connection
  - `src/database/models.py`: SQLAlchemy models
  - `src/database/two_tower_repository.py`: Repository pattern
  - `src/database/multi_field_repository.py`: Multi-field repository
- **Chức năng**: Data persistence, CRUD operations

#### 2.2.5. Vector Search Layer (FAISS)
- **File**: `src/vector_search/two_tower_faiss_manager.py`
- **Chức năng**: 
  - Quản lý 6 FAISS indices riêng biệt (3 cho jobs, 3 cho candidates)
  - Similarity search operations
  - Index building và management

### 2.3. Entry Point và Application Bootstrap

**Theo codebase hiện tại**:

#### 2.3.1. Main Entry Point
- **File**: `main.py`
```python
uvicorn.run(
    "src.api.main:app",
    host=settings.API_HOST,
    port=settings.API_PORT,
    reload=True
)
```

#### 2.3.2. FastAPI Application Initialization
- **File**: `src/api/main.py`
- **Version**: 2.0.0
- **Features**:
  - CORS middleware enabled
  - Static files serving tại `/static`
  - Two-Tower routes với prefix `/api/v2`
  - Startup/shutdown event handlers

#### 2.3.3. Configuration Management
- **File**: `config/settings.py`
- **Settings**:
  - Database: PostgreSQL connection (user, password, host, port, database)
  - Embedding Model: Vietnamese SimCSE model
  - API: Host và port
  - Logging: Log level

---

## 3. MÔ HÌNH VÀ EMBEDDINGS

### 3.1. Two-Tower Model Architecture

**Theo codebase hiện tại**, Two-Tower Model được định nghĩa trong `src/models/two_tower_model.py`:

#### 3.1.1. CandidateTower Neural Network
- **Input**: 3 embeddings (title, skills, experience) - mỗi embedding 768 dimensions
- **Architecture**:
  - Concatenate 3 embeddings → input_dim = 768 * 3 = 2304
  - Hidden layers: [512, 256] (configurable)
  - Output layer: 256 dimensions (configurable)
  - Batch normalization và dropout (0.1)
  - L2 normalization ở output
- **Activation**: ReLU
- **Weight initialization**: Xavier uniform

#### 3.1.2. JobTower Neural Network
- **Input**: 3 embeddings (title, skills, requirement) - mỗi embedding 768 dimensions
- **Architecture**: Tương tự CandidateTower
  - Concatenate 3 embeddings → input_dim = 768 * 3 = 2304
  - Hidden layers: [512, 256]
  - Output layer: 256 dimensions
  - Batch normalization và dropout
  - L2 normalization ở output

#### 3.1.3. TwoTowerModel Class
- **Forward pass**: 
  - Encode candidate qua CandidateTower → candidate_repr [batch_size, 256]
  - Encode job qua JobTower → job_repr [batch_size, 256]
  - Return (candidate_repr, job_repr)
- **Similarity computation**: 
  - Dot product (vì cả 2 đều L2 normalized → cosine similarity)
  - `similarity = torch.sum(candidate_repr * job_repr, dim=1)`

#### 3.1.4. Model Loading
**Theo codebase hiện tại**, trong `src/services/two_tower_matching_service.py`:
- Default model path: `outputs_improved/best_model_improved.pt`
- Model được load với `torch.load()` và `load_state_dict()`
- Model được set to eval mode: `model.eval()`
- Device: CPU (có thể config thành GPU)

### 3.2. Embedding Encoders

#### 3.2.1. JobTowerEncoder
**Theo codebase hiện tại** (`src/embeddings/job_tower_encoder.py`):

- **Model**: Sử dụng `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base` (default) hoặc fallback model
- **Dimension**: 768 (từ PhoBERT base)
- **Preprocessing functions**:
  - `preprocess_job_title()`: Translate Vietnamese → English, normalize, truncate 200 chars
  - `preprocess_job_skills()`: Translate, normalize, truncate 1000 chars
  - `preprocess_job_requirements()`: Translate, normalize, truncate 2000 chars
- **Tokenization**: Sử dụng pyvi nếu model yêu cầu Vietnamese tokenization
- **Encoding**: 
  - `encode_job()`: Generate 3 embeddings (title, skills, requirement)
  - Normalize embeddings (L2 normalization)
  - Return dict với keys: `title_embedding`, `skills_embedding`, `requirement_embedding`

#### 3.2.2. CandidateTowerEncoder
**Theo codebase hiện tại** (`src/embeddings/candidate_tower_encoder.py`):

- **Tương tự JobTowerEncoder** nhưng cho candidates:
  - `title_embedding`: Candidate title (desired job)
  - `skills_embedding`: Candidate skills
  - `experience_embedding`: Candidate experience

### 3.3. Embedding Model Selection

**Theo codebase hiện tại** (`src/embeddings/model_selector.py` và `src/utils/embedding_loader.py`):

- **Preferred model**: `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base`
- **Fallback model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Model loading**: Sử dụng `SentenceTransformer.from_pretrained()`
- **Dimension detection**: `model.get_sentence_embedding_dimension()`

### 3.4. Embedding Service và Caching

#### 3.4.1. OptimizedEmbeddingService
**Theo codebase hiện tại** (`src/services/embedding_service.py`):

- **Features**:
  - Smart caching với 12-hour TTL
  - Batch processing
  - Content hash tracking để detect changes
  - Database integration
- **Methods**:
  - `get_candidate_embedding()`: Get candidate embedding với caching
  - `get_job_embedding()`: Get job embedding với caching
  - `_save_candidate_embedding()`: Save to database
  - `_save_job_embedding()`: Save to database

#### 3.4.2. EmbeddingCacheManager
**Theo codebase hiện tại** (`src/services/embedding_cache_manager.py`):

- **Cache TTL**: 12 hours (configurable)
- **Features**:
  - In-memory cache với thread-safe operations (Lock)
  - Content hash tracking
  - Automatic expiration
  - Cache statistics
- **Methods**:
  - `get()`: Get cached embedding (check expiration và content hash)
  - `set()`: Cache embedding với timestamp và content hash
  - `invalidate()`: Invalidate cache cho specific entity
  - `clear_expired()`: Clear expired entries
  - `needs_refresh()`: Check if entity needs refresh

---

## 4. HỆ THỐNG MATCHING VÀ TÌM KIẾM

### 4.1. Two-Tower Matching Service

**Theo codebase hiện tại** (`src/services/two_tower_matching_service.py`):

#### 4.1.1. TwoTowerMatchingService Class
- **Initialization**:
  - Load TwoTowerModel từ checkpoint
  - Initialize TwoTowerRepository
  - Set device (CPU/GPU)

#### 4.1.2. find_jobs_for_candidate()
**Theo codebase hiện tại**, quy trình:
1. Get candidate từ database
2. Build candidate text: `"Title: {title} | Skills: {skills} | Experience: {experience}"`
3. Get all jobs từ database
4. Build job texts tương tự
5. Encode candidate: `model.encode_candidates([candidate_text])[0]`
6. Encode all jobs in batches (batch_size=32)
7. Compute cosine similarity: `np.dot(job_embs, candidate_emb_np)`
8. Get top-k: `np.argsort(similarities)[::-1][:top_k]`
9. Return results với scores

**Lưu ý**: Hiện tại service này sử dụng **single embedding** (concatenated text) thay vì 3 embeddings riêng biệt. Đây là "Simple Two-Tower Matching Service" như comment trong code.

#### 4.1.3. find_candidates_for_job()
- Tương tự `find_jobs_for_candidate()` nhưng ngược lại

### 4.2. Vector Search với FAISS

**Theo codebase hiện tại** (`src/vector_search/two_tower_faiss_manager.py`):

#### 4.2.1. TwoTowerFAISSManager Class
- **6 FAISS indices riêng biệt**:
  - Job indices: `job_title_index`, `job_skills_index`, `job_requirement_index`
  - Candidate indices: `candidate_title_index`, `candidate_skills_index`, `candidate_experience_index`
- **Index type**: HNSW (Hierarchical Navigable Small World) hoặc Flat
- **HNSW parameters**:
  - `M = 32` (default): Number of connections per node
  - `ef_construction = 200` (default): Size of candidate list during construction
  - `ef_search = 128` (default): Size of candidate list during search
- **Dimension**: 768 (embedding dimension)
- **Normalization**: Vectors được normalize cho cosine similarity

#### 4.2.2. Index Building
**Theo codebase hiện tại**, `build_indices_from_db()`:
1. Query all jobs từ `JobDescriptionTwoTower` table
2. Extract 3 embeddings cho mỗi job
3. Normalize vectors
4. Add vectors vào 3 job indices
5. Store ID mappings (FAISS index position → job_id)
6. Tương tự cho candidates

#### 4.2.3. Search Operations
- `search_job_by_field()`: Search jobs by field type (title/skills/requirement)
- `search_candidate_by_field()`: Search candidates by field type (title/skills/experience)
- **Search process**:
  1. Select appropriate index và ID map
  2. Normalize query vector
  3. Set `ef_search` cho HNSW
  4. Search: `index.search(query_vec, k)`
  5. Convert L2 distance → cosine similarity: `1.0 - (dist / 2.0)`
  6. Return list of (entity_id, similarity_score)

#### 4.2.4. Index Persistence
- `save_indices()`: Save indices và ID maps to disk
- `load_indices()`: Load indices và ID maps from disk
- **Storage**: 
  - Indices: `.faiss` files
  - ID maps: `.pkl` files

### 4.3. Rule-based Matching

**Theo codebase hiện tại** (`src/utils/rule_matcher.py`):

#### 4.3.1. RuleMatcher Class
- **Chức năng**: Enhanced rule-based matching cho CV-Job validation
- **Features**:
  - Title similarity matching
  - Skills matching (exact, fuzzy, category-level)
  - Experience matching
  - Vietnamese text handling
  - Skill synonyms và translations

#### 4.3.2. Title Similarity Matching
**Theo codebase hiện tại**:
- Tokenize và normalize titles
- Compute similarity scores:
  - Exact match
  - Token-level matching
  - Sequence similarity (difflib.SequenceMatcher)
- Threshold: 0.60 (60%) để PASS

#### 4.3.3. Skills Matching
**Theo codebase hiện tại**, có nhiều levels:
- **Exact matching**: Exact string match (case-insensitive)
- **Fuzzy matching**: SequenceMatcher ratio > 0.85
- **Category-level matching**: Match skills trong cùng category (frontend, backend, devops, etc.)
- **Synonym matching**: Sử dụng `SKILL_SYNONYMS` dictionary
- **Vietnamese-English translation**: Sử dụng `VIETNAMESE_ENGLISH_SKILLS` mapping

#### 4.3.4. Skill Categories
**Theo codebase hiện tại**, có các categories:
- `frontend`: React, Vue, Angular, JavaScript, TypeScript, HTML, CSS, etc.
- `backend`: Python, Java, Node.js, Go, Rust, PHP, Ruby, etc.
- `devops`: AWS, Azure, Docker, Kubernetes, Jenkins, Terraform, etc.
- `database`: PostgreSQL, MySQL, MongoDB, Redis, etc.
- `mobile`: React Native, Flutter, iOS, Android, etc.
- `data`: Python, Pandas, Spark, Hadoop, TensorFlow, PyTorch, etc.

#### 4.3.5. Rule Scoring
**Theo codebase hiện tại**, rule matching trả về:
- `rule1`: Title similarity score và status (PASS/FAIL)
- `skill_score`: Skills matching score
- `final_title_score`: Final title score
- `final_status`: Overall status (OK/NG)

---

## 5. XỬ LÝ DỮ LIỆU

### 5.1. Data Processing Layer

**Theo codebase hiện tại**:

#### 5.1.1. JDProcessor
- **File**: `src/data_processing/jd_processor.py`
- **Chức năng**: Process job description datasets
- **Features**: Data validation, field extraction

#### 5.1.2. CandidateProcessor
- **File**: `src/data_processing/candidate_processor.py`
- **Chức năng**: Process candidate datasets
- **Features**: Data validation, field extraction

### 5.2. Data Preprocessing Utilities

**Theo codebase hiện tại**:

#### 5.2.1. DataValidator
- **File**: `src/utils/data_validator.py`
- **Chức năng**: Validate data quality

#### 5.2.2. DataPreprocessor
- **File**: `src/utils/data_preprocessor.py`
- **Chức năng**: Clean và normalize data

#### 5.2.3. CleanData
- **File**: `src/utils/clean_data.py`
- **Chức năng**: Data cleaning operations

#### 5.2.4. ThreeFieldExtractor
- **File**: `src/utils/three_field_extractor.py`
- **Chức năng**: Extract 3 fields (title, skills, requirement/experience) từ raw data

### 5.3. Data Quality và Validation

**Theo codebase hiện tại**, có các scripts:
- `scripts/check_raw_data.py`: Comprehensive raw data quality check
- `scripts/validate_data.py`: Data validation script
- `scripts/preprocess_data.py`: Data preprocessing script
- `scripts/data_pipeline.py`: Complete pipeline (validate + preprocess)

---

## 6. LỚP DATABASE

### 6.1. Database Connection và Configuration

**Theo codebase hiện tại** (`src/database/connection.py`):

- **Engine**: SQLAlchemy engine với PostgreSQL
- **Connection pooling**:
  - `pool_size=10`
  - `max_overflow=20`
  - `pool_pre_ping=True` (check connection before use)
- **Session**: `SessionLocal` factory
- **Base**: SQLAlchemy declarative base

### 6.2. Database Models

**Theo codebase hiện tại** (`src/database/models.py`):

#### 6.2.1. Legacy Models
- **JobDescriptionEmbedding**: Single embedding per job
- **CandidateEmbedding**: Single embedding per candidate

#### 6.2.2. Multi-field Embedding Models
- **JobDescriptionMultiEmbedding**:
  - 3 embeddings: `title_embedding`, `skills_embedding`, `requirement_embedding`
  - Fields: `job_id`, `title`, `skills`, `requirement`, `company`, `location`
  - Timestamps: `created_at`, `updated_at`, `embedding_timestamp`
  - Content hash: `content_hash` (MD5) để detect changes
  - Indexes: GIN indexes cho embeddings

- **CandidateMultiEmbedding**:
  - 3 embeddings: `title_embedding`, `skills_embedding`, `experience_embedding`
  - Fields: `candidate_id`, `name`, `email`, `title`, `skills`, `experience`
  - Tương tự JobDescriptionMultiEmbedding

#### 6.2.3. Two-Tower Models
- **JobDescriptionTwoTower**:
  - 3 embeddings: `title_embedding`, `skills_embedding`, `requirement_embedding`
  - Fields: `job_id`, `title`, `skills`, `requirement`, `company`, `location`
  - Indexes: GIN indexes cho embeddings

- **CandidateTwoTower**:
  - 3 embeddings: `title_embedding`, `skills_embedding`, `experience_embedding`
  - Fields: `candidate_id`, `name`, `email`, `title`, `skills`, `experience`
  - Indexes: GIN indexes cho embeddings

#### 6.2.4. Recommendation Models
- **ProcessedCandidateRecommendation**:
  - Store top 10 jobs per candidate
  - Fields: `candidate_id`, `job_id`, `similarity_score`
  - Field scores: `skills_similarity`, `experience_similarity`, `desired_job_similarity`
  - Explainability fields:
    - `rule_scores`: JSON string (rule matching results)
    - `embedding_scores`: JSON string (embedding similarity scores)
    - `explanation_text`: Human-readable explanation
    - `comprehensive_explanation`: JSON string (full explanation với all levels)
    - `confidence_score`: Final confidence score (0-1)

#### 6.2.5. Reindex Tracking
- **ReindexTracking**:
  - Track reindex operations
  - Fields: `reindex_type`, `status`, `total_records`, `processed_records`, `started_at`, `completed_at`, `error_message`

### 6.3. Database Repositories

**Theo codebase hiện tại**:

#### 6.3.1. TwoTowerRepository
- **File**: `src/database/two_tower_repository.py`
- **Methods**:
  - `create_job()`: Create/update job
  - `get_job()`: Get job by job_id
  - `get_all_jobs()`: Get all jobs
  - `create_candidate()`: Create/update candidate
  - `get_candidate()`: Get candidate by candidate_id
  - `get_all_candidates()`: Get all candidates
  - `create_reindex_tracking()`: Create reindex tracking record
  - `update_reindex_tracking()`: Update reindex tracking

#### 6.3.2. MultiFieldEmbeddingRepository
- **File**: `src/database/multi_field_repository.py`
- **Chức năng**: CRUD operations cho multi-field embeddings

### 6.4. Database Migrations

**Theo codebase hiện tại**:
- **Alembic**: Configuration file `alembic.ini`
- **Migration scripts**:
  - `scripts/migrate_to_two_tower_schema.py`: Migrate to Two-Tower schema
  - `scripts/add_explanation_fields_migration.py`: Add explainability fields
  - `scripts/create_embedding_timestamp_migration.py`: Add embedding timestamp

---

## 7. API LAYER

### 7.1. FastAPI Application

**Theo codebase hiện tại** (`src/api/main.py`):

- **Version**: 2.0.0
- **Title**: "AI Job Recommendation Service - Two-Tower Architecture"
- **CORS**: Enabled với `allow_origins=["*"]`
- **Static files**: Mount tại `/static`
- **Routes**: Include `two_tower_router` với prefix `/api/v2`

### 7.2. API Routes

**Theo codebase hiện tại** (`src/api/two_tower_routes.py`):

#### 7.2.1. POST /api/v2/search/jobs
- **Request**: `JobSearchRequest` (candidate_id, top_k)
- **Response**: `JobSearchResponse` (total_matches, matches[])
- **Logic**: 
  - Create `TwoTowerMatchingService`
  - Call `find_jobs_for_candidate()`
  - Return job matches với scores

#### 7.2.2. POST /api/v2/search/candidates
- **Request**: `CandidateSearchRequest` (job_id, top_k)
- **Response**: `CandidateSearchResponse` (total_matches, matches[])
- **Logic**: Tương tự search jobs

#### 7.2.3. POST /api/v2/index/job
- **Request**: `IndexJobRequest` (job_id, title, skills, requirement, company, location)
- **Response**: `IndexResponse` (status, job_id, message)
- **Logic**:
  - Create `JobTowerEncoder`
  - Encode job → 3 embeddings
  - Save to database via `TwoTowerRepository`

#### 7.2.4. POST /api/v2/index/candidate
- **Request**: `IndexCandidateRequest` (candidate_id, title, skills, experience, name, email)
- **Response**: `IndexResponse` (status, candidate_id, message)
- **Logic**: Tương tự index job

#### 7.2.5. POST /api/v2/reindex
- **Request**: `ReindexRequest` (reindex_type)
- **Response**: `ReindexResponse` (status, reindex_id, message, estimated_time_minutes)
- **Logic**: 
  - Create reindex tracking record
  - TODO: Run reindex in background task (chưa implement)

#### 7.2.6. GET /api/v2/health
- **Response**: `HealthResponse` (status, version, indices, database, total_jobs, total_candidates)
- **Logic**:
  - Check database connection
  - Count jobs và candidates
  - Check FAISS indices status

### 7.3. API Schemas

**Theo codebase hiện tại** (`src/api/two_tower_schemas.py`):

- **Request models**: Pydantic models cho validation
- **Response models**: Pydantic models cho response
- **Models**:
  - `JobSearchRequest`, `JobSearchResponse`, `JobMatch`
  - `CandidateSearchRequest`, `CandidateSearchResponse`, `CandidateMatch`
  - `IndexJobRequest`, `IndexCandidateRequest`, `IndexResponse`
  - `ReindexRequest`, `ReindexResponse`
  - `HealthResponse`
  - `FieldScores`, `RuleMatchingResult`, `RuleDetails` (cho explainability)

---

## 8. SERVICES VÀ BUSINESS LOGIC

### 8.1. Matching Services

#### 8.1.1. TwoTowerMatchingService
**Theo codebase hiện tại** (`src/services/two_tower_matching_service.py`):
- **Matching pipeline**: 
  1. Get candidate/job từ database
  2. Build text representation
  3. Encode với TwoTowerModel
  4. Compute cosine similarity
  5. Return top-k results
- **Lưu ý**: Hiện tại sử dụng single embedding (concatenated text) thay vì 3 embeddings riêng biệt

### 8.2. Embedding Services

#### 8.2.1. OptimizedEmbeddingService
**Theo codebase hiện tại** (`src/services/embedding_service.py`):
- **Features**:
  - Smart caching (12-hour TTL)
  - Batch processing
  - Content hash tracking
  - Database integration
- **Lazy loading**: Encoders chỉ được load khi cần

#### 8.2.2. EmbeddingCacheManager
**Theo codebase hiện tại** (`src/services/embedding_cache_manager.py`):
- **Cache TTL**: 12 hours
- **Thread-safe**: Sử dụng Lock
- **Content hash**: Detect content changes
- **Automatic expiration**: Clear expired entries

### 8.3. Scheduler Services

#### 8.3.1. EmbeddingScheduler
**Theo codebase hiện tại** (`src/services/embedding_scheduler.py`):
- **Chức năng**: Scheduled embedding updates (12-hour cycle)
- **Script**: `scripts/run_embedding_scheduler.py`

---

## 9. UTILITIES VÀ HELPERS

### 9.1. Text Processing Utilities

**Theo codebase hiện tại**:

#### 9.1.1. VietnameseTranslator
- **File**: `src/utils/vietnamese_translator.py`
- **Chức năng**: Translate Vietnamese text to English
- **Backends**: Multiple translation backends (Google Translate, etc.)

#### 9.1.2. TextEnhancer
- **File**: `src/utils/text_enhancer.py`
- **Chức năng**: Text enhancement operations

#### 9.1.3. Vietnamese Tokenization
- **Library**: pyvi
- **Usage**: Tokenize Vietnamese text nếu model yêu cầu

### 9.2. Embedding Utilities

#### 9.2.1. EmbeddingLoader
- **File**: `src/utils/embedding_loader.py`
- **Chức năng**: Load embedding models với fallback support
- **Features**: Model loading, dimension detection

### 9.3. Explanation Utilities

#### 9.3.1. ExplanationGenerator
**Theo codebase hiện tại** (`src/utils/explanation_generator.py`):

**5 levels of explainability**:

1. **Level 1: Rule Matching Explanation (Deterministic)**
   - Rule scores và status
   - Matched tokens
   - Title similarity analysis

2. **Level 2: Embedding Similarity Explanation (Semantic Features)**
   - Embedding similarity scores cho từng field
   - Combined similarity
   - Interpretation

3. **Level 3: Humanized Explanation (Natural Language)**
   - Human-readable explanation text (Vietnamese)
   - English translation
   - Components breakdown

4. **Level 4: Counterfactual Explanation (What-if scenarios)**
   - Suggest missing skills
   - Estimated score improvement
   - Recommendations

5. **Level 5: Confidence Score Calculation**
   - Final confidence score (0-1)
   - Weighted combination của rule scores và embedding scores

#### 9.3.2. ExplanationStorage
- **File**: `src/utils/explanation_storage.py`
- **Chức năng**: Store explanations to database

### 9.4. Data Utilities

#### 9.4.1. ColumnMapper
- **File**: `src/utils/column_mapper.py`
- **Chức năng**: Map columns từ different data sources

#### 9.4.2. ReportGenerator
- **File**: `src/utils/report_generator.py`
- **Chức năng**: Generate quality và preprocessing reports

### 9.5. Logging Utilities

#### 9.5.1. LoggingUTF8
- **File**: `src/utils/logging_utf8.py`
- **Chức năng**: UTF-8 logging support

---

## 10. TRAINING VÀ EVALUATION

### 10.1. Training Pipeline

**Theo codebase hiện tại** (`src/models/training_pipeline.py`):

#### 10.1.1. TrainingPipeline
- **Dataset**: `GroundTruthDataset` (ground truth pairs)
- **DataLoader**: Custom collate function
- **Training loop**: Standard PyTorch training loop
- **Checkpointing**: Save best model

#### 10.1.2. GroundTruthDataset
- **Features**: 
  - Candidate embeddings (title, skills, experience)
  - Job embeddings (title, skills, requirement)
  - Labels (0/1 for match/no-match)

### 10.2. Evaluation Metrics

**Theo codebase hiện tại** (`src/models/evaluation_metrics.py`):

#### 10.2.1. TwoTowerEvaluator
- **Metrics**:
  - Accuracy
  - Recall@K
  - Precision metrics

### 10.3. Ground Truth Building

**Theo codebase hiện tại** (`src/models/ground_truth_builder.py`):

#### 10.3.1. GroundTruthBuilder
- **Chức năng**: Generate ground truth dataset từ labeled data
- **Script**: `scripts/generate_ground_truth_500_pairs.py`

---

## 11. KIẾN THỨC BỔ SUNG VÀ ĐỀ XUẤT CẢI TIẾN

### 11.1. Lý thuyết về Two-Tower Architecture

**Theo tổng hợp từ Web Search**, Two-Tower Architecture là một kiến trúc neural network phổ biến trong recommendation systems:

#### 11.1.1. Nguyên lý hoạt động
- **Two separate towers**: Một tower encode user/candidate features, một tower encode item/job features
- **Shared embedding space**: Cả 2 towers output embeddings trong cùng một space
- **Similarity computation**: Cosine similarity hoặc dot product giữa 2 embeddings
- **Training**: Contrastive learning với positive và negative pairs

#### 11.1.2. Ưu điểm
- **Efficiency**: Có thể pre-compute item embeddings offline
- **Scalability**: Fast inference cho large-scale systems
- **Flexibility**: Có thể update user tower mà không cần retrain item tower

#### 11.1.3. Best Practices
- **Normalization**: L2 normalize embeddings để cosine similarity = dot product
- **Negative sampling**: Sử dụng hard negatives để improve training
- **Multi-field embeddings**: Separate embeddings cho different fields (như hệ thống hiện tại)
- **Fine-tuning**: Fine-tune pre-trained models cho domain-specific tasks

### 11.2. Lý thuyết về FAISS và Vector Search

**Theo tổng hợp từ Web Search**, FAISS (Facebook AI Similarity Search) là một library cho efficient similarity search:

#### 11.2.1. Index Types
- **Flat Index**: Exact search, O(n) complexity
- **IVF (Inverted File Index)**: Approximate search với clustering
- **HNSW (Hierarchical Navigable Small World)**: Graph-based approximate search
  - **M**: Number of connections per node (higher = more accurate, slower)
  - **ef_construction**: Candidate list size during construction (higher = better quality, slower)
  - **ef_search**: Candidate list size during search (higher = more accurate, slower)

#### 11.2.2. Performance Optimization
- **Normalization**: Normalize vectors cho cosine similarity
- **Batch operations**: Process multiple queries cùng lúc
- **Index parameters tuning**: Balance giữa accuracy và speed
- **GPU acceleration**: Sử dụng GPU cho large-scale search

#### 11.2.3. Best Practices
- **Index building**: Build index offline, load at runtime
- **Incremental updates**: Support incremental index updates
- **Memory management**: Monitor memory usage với large indices
- **Query optimization**: Batch queries để improve throughput

### 11.3. Lý thuyết về Hybrid Retrieval Systems

**Theo tổng hợp từ Web Search**, Hybrid Retrieval Systems kết hợp multiple retrieval methods:

#### 11.3.1. Components
- **Semantic search**: Embedding-based similarity search
- **Keyword search**: Traditional keyword matching
- **Rule-based filtering**: Deterministic rules
- **Reranking**: Cross-encoder hoặc learning-to-rank

#### 11.3.2. Best Practices
- **Weighted combination**: Combine scores với learned weights
- **Multi-stage retrieval**: Coarse-to-fine retrieval pipeline
- **Explainability**: Provide explanations cho recommendations
- **A/B testing**: Test different retrieval strategies

### 11.4. Đề xuất cải tiến

**Dựa trên codebase hiện tại và best practices**:

#### 11.4.1. Cải tiến Two-Tower Matching Service
**Vấn đề hiện tại**: `TwoTowerMatchingService` sử dụng single embedding (concatenated text) thay vì 3 embeddings riêng biệt.

**Đề xuất**:
- Sử dụng 3 embeddings riêng biệt từ `TwoTowerFAISSManager`
- Weighted combination: `final_score = w1*title_score + w2*skills_score + w3*experience_score`
- Tận dụng FAISS indices thay vì encode all jobs/candidates mỗi lần

#### 11.4.2. Cải tiến FAISS Index Management
**Vấn đề hiện tại**: FAISS indices chưa được tích hợp vào matching service.

**Đề xuất**:
- Integrate `TwoTowerFAISSManager` vào `TwoTowerMatchingService`
- Sử dụng FAISS search thay vì brute-force similarity computation
- Support incremental index updates
- Add index health monitoring

#### 11.4.3. Cải tiến Caching Strategy
**Vấn đề hiện tại**: Cache TTL cố định 12 hours.

**Đề xuất**:
- Adaptive TTL dựa trên access patterns
- Redis cache cho distributed systems
- Cache warming strategies
- Cache invalidation policies

#### 11.4.4. Cải tiến Explainability
**Vấn đề hiện tại**: ExplanationGenerator có 5 levels nhưng chưa được tích hợp đầy đủ.

**Đề xuất**:
- Integrate explanation generation vào matching pipeline
- Store explanations trong database
- API endpoint để get explanations
- Visualization cho explanations

#### 11.4.5. Cải tiến Training Pipeline
**Vấn đề hiện tại**: Training pipeline có thể được optimize.

**Đề xuất**:
- Hard negative mining
- Curriculum learning
- Multi-task learning
- Hyperparameter optimization

#### 11.4.6. Cải tiến API Performance
**Vấn đề hiện tại**: API có thể được optimize cho better performance.

**Đề xuất**:
- Async operations cho long-running tasks
- Background tasks cho reindexing
- Response caching
- Rate limiting
- API versioning

#### 11.4.7. Cải tiến Monitoring và Observability
**Vấn đề hiện tại**: Monitoring chưa đầy đủ.

**Đề xuất**:
- Metrics collection (Prometheus)
- Distributed tracing (OpenTelemetry)
- Log aggregation (ELK stack)
- Alerting system
- Performance dashboards

---

## 12. KẾT LUẬN

### 12.1. Tóm tắt hệ thống

**Theo codebase hiện tại**, hệ thống AI Service là một job recommendation system với:

- **Kiến trúc**: Two-Tower Architecture với multi-field embeddings
- **Matching**: Semantic similarity search với FAISS
- **Rule-based**: Enhanced rule matching cho validation
- **Explainability**: 5-level explanation system
- **Caching**: Smart caching với 12-hour TTL
- **API**: RESTful API với FastAPI

### 12.2. Điểm mạnh

1. **Multi-field embeddings**: 3 embeddings riêng biệt cho fine-grained matching
2. **FAISS integration**: Efficient vector search với HNSW indices
3. **Rule-based matching**: Deterministic validation với explainability
4. **Caching strategy**: Smart caching để improve performance
5. **Vietnamese support**: Vietnamese text processing và translation

### 12.3. Điểm cần cải thiện

1. **Matching service**: Cần integrate FAISS indices thay vì brute-force
2. **Weighted scoring**: Cần support weighted combination của 3 embeddings
3. **Background tasks**: Cần implement async reindexing
4. **Monitoring**: Cần thêm metrics và observability
5. **Testing**: Cần thêm unit tests và integration tests

### 12.4. Hướng phát triển

1. **Hybrid retrieval**: Kết hợp semantic search, keyword search, và rule-based filtering
2. **Reranking**: Cross-encoder reranking cho better accuracy
3. **Learning-to-rank**: Train ranking model với user feedback
4. **Real-time updates**: Support real-time index updates
5. **Distributed system**: Scale to distributed architecture

---

**Tài liệu tham khảo**:
- Codebase: `E:\4. CODE\AI_SERVICE`
- Two-Tower Architecture: Industry best practices
- FAISS Documentation: Facebook AI Research
- Hybrid Retrieval Systems: Research papers và industry implementations




