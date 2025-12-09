# 🏗️ Two-Tower Architecture Design & Implementation Guide

## 📋 Tóm Tắt

Tài liệu này mô tả thiết kế và kế hoạch triển khai để chuyển đổi hệ thống AI Job Recommendation Service từ kiến trúc hiện tại (multi-field embeddings với single encoder) sang **kiến trúc Two-Tower** với hai tower độc lập (Job Tower và Candidate Tower), mỗi tower encode 3 fields riêng biệt (Title/Experience, Skills, Requirements/Description).

**Mục tiêu chính:**
- Tách biệt encoding logic cho Jobs và Candidates
- Hỗ trợ training/fine-tuning độc lập cho mỗi tower
- Matching theo multi-stage: per-field ANN → merge/rerank → final scoring
- Dễ dàng mở rộng (thêm field mới)
- Giữ tương thích API hiện tại

---

## 📑 Mục Lục

- [A. Thiết kế kiến trúc](#a-thiết-kế-kiến-trúc-chi-tiết)
- [B. Thiết kế dữ liệu & lưu trữ](#b-thiết-kế-dữ-liệu--lưu-trữ)
- [C. Pipeline Embedding](#c-pipeline-embedding-mỗi-tower)
- [D. Indexing & Update Strategy](#d-indexing--update-strategy)
- [E. Matching Pipeline (3-stage)](#e-matching-pipeline-3-stage)
- [F. API Contract](#f-api-contract-endpoints)
- [G. Performance, Scalability & Ops](#g-performance-scalability--ops)
- [H. Migration Plan](#h-migration-plan-step-by-step)
- [I. Testing & Evaluation](#i-testing--evaluation)
- [J. Security & Privacy](#j-security--privacy)
- [K. Deliverables & Acceptance Criteria](#k-deliverables--acceptance-criteria)

---

## A. Thiết Kế Kiến Trúc (Chi Tiết)

### A.1. Sơ Đồ Kiến Trúc Two-Tower (End-to-End)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API LAYER (FastAPI)                                │
│  ┌──────────────────────┐          ┌──────────────────────┐                 │
│  │ POST /search/jobs    │          │ POST /search/candidates│              │
│  │ POST /index/job      │          │ POST /index/candidate │               │
│  │ POST /reindex        │          │ GET /health           │               │
│  └──────────────────────┘          └──────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SERVICE LAYER (Matching Coordinator)                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  TwoTowerMatchingService                                           │   │
│  │  - Stage 1: Per-field ANN search (3 parallel searches)             │   │
│  │  - Stage 2: Candidate union & coarse scoring                      │   │
│  │  - Stage 3: Rerank (optional cross-encoder)                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EMBEDDING LAYER (Two Towers)                            │
│                                                                             │
│  ┌──────────────────────────────────┐  ┌─────────────────────────────────┐│
│  │      JOB TOWER ENCODER            │  │  CANDIDATE TOWER ENCODER        ││
│  │                                    │  │                                  ││
│  │  Input Fields:                    │  │  Input Fields:                   ││
│  │  - Title (text)                   │  │  - Title/Experience (text)       ││
│  │  - Skills (text)                  │  │  - Skills (text)                ││
│  │  - Requirements (text)            │  │  - Experience (text)              ││
│  │                                    │  │                                  ││
│  │  Processing:                       │  │  Processing:                    ││
│  │  1. Preprocess (normalize,         │  │  1. Preprocess (normalize,      ││
│  │     tokenize, truncate)            │  │     tokenize, truncate)          ││
│  │  2. Encode per field              │  │  2. Encode per field            ││
│  │     (SentenceTransformer)          │  │     (SentenceTransformer)        ││
│  │  3. Output: 3 embeddings           │  │  3. Output: 3 embeddings        ││
│  │     [title_emb, skills_emb,        │  │     [title_emb, skills_emb,     ││
│  │      req_emb]                      │  │      exp_emb]                   ││
│  └──────────────────────────────────┘  └─────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VECTOR STORE (FAISS + PostgreSQL)                        │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  FAISS INDICES (Per-Field, Per-Tower)                              │   │
│  │  - faiss_job_title_idx          (Job Tower - Title)                │   │
│  │  - faiss_job_skills_idx         (Job Tower - Skills)               │   │
│  │  - faiss_job_requirement_idx    (Job Tower - Requirements)        │   │
│  │  - faiss_candidate_title_idx    (Candidate Tower - Title)          │   │
│  │  - faiss_candidate_skills_idx   (Candidate Tower - Skills)         │   │
│  │  - faiss_candidate_experience_idx (Candidate Tower - Experience)   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  PostgreSQL (Metadata + Embeddings)                                 │   │
│  │  - job_description_two_tower (job_id, title, skills, requirement,  │   │
│  │    title_emb, skills_emb, req_emb, ...)                            │   │
│  │  - candidate_two_tower (candidate_id, title, skills, experience,   │   │
│  │    title_emb, skills_emb, exp_emb, ...)                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### A.2. Trách Nhiệm Các Component

#### **Job Tower Encoder Service**
- **File**: `src/embeddings/job_tower_encoder.py`
- **Trách nhiệm**:
  - Nhận input: `title`, `skills`, `requirements` (text)
  - Preprocess từng field (normalize, tokenize, truncate)
  - Encode mỗi field thành embedding vector (768-dim, giả định)
  - Trả về dict: `{'title_embedding': [...], 'skills_embedding': [...], 'requirement_embedding': [...]}`
- **Model**: SentenceTransformer (có thể fine-tune riêng cho Job Tower)
- **Output**: 3 embeddings độc lập, mỗi embedding 768 dimensions

#### **Candidate Tower Encoder Service**
- **File**: `src/embeddings/candidate_tower_encoder.py`
- **Trách nhiệm**:
  - Nhận input: `title/experience`, `skills`, `experience` (text)
  - Preprocess từng field
  - Encode mỗi field thành embedding vector
  - Trả về dict: `{'title_embedding': [...], 'skills_embedding': [...], 'experience_embedding': [...]}`
- **Model**: SentenceTransformer (có thể fine-tune riêng cho Candidate Tower)
- **Output**: 3 embeddings độc lập, mỗi embedding 768 dimensions

#### **Vector Store (FAISS)**
- **File**: `src/vector_search/two_tower_faiss_manager.py`
- **Tổ chức**: 6 FAISS indices riêng biệt (3 cho Jobs, 3 cho Candidates)
- **Index naming convention**:
  - `faiss_job_title_idx`
  - `faiss_job_skills_idx`
  - `faiss_job_requirement_idx`
  - `faiss_candidate_title_idx`
  - `faiss_candidate_skills_idx`
  - `faiss_candidate_experience_idx`
- **Index type**: HNSW (Hierarchical Navigable Small World) với params:
  - `M=32` (connections per node)
  - `ef_construction=200` (build time)
  - `ef_search=128` (search time)
- **Normalization**: L2 normalization cho cosine similarity

#### **Service Layer (Matching Coordinator)**
- **File**: `src/services/two_tower_matching_service.py`
- **Trách nhiệm**:
  - Stage 1: Thực hiện 3 ANN searches song song (title, skills, requirement)
  - Stage 2: Merge kết quả, tính weighted score
  - Stage 3: Rerank top M candidates (optional cross-encoder)
  - Trả về final ranked list với explainability (per-field scores)

#### **API Layer**
- **File**: `src/api/two_tower_routes.py` (hoặc extend `src/api/routes.py`)
- **Endpoints**:
  - `POST /api/v2/search/jobs` - Tìm jobs cho candidate
  - `POST /api/v2/search/candidates` - Tìm candidates cho job
  - `POST /api/v2/index/job` - Index 1 job
  - `POST /api/v2/index/candidate` - Index 1 candidate
  - `POST /api/v2/reindex` - Reindex toàn bộ hoặc incremental
  - `GET /api/v2/health` - Health check

### A.3. Microservice vs Monolith

**Đề xuất: Monolith với module separation**

**Ưu điểm Monolith:**
- Dễ deploy và maintain (1 service)
- Latency thấp (no network overhead giữa services)
- Shared database connection pool
- Dễ debug và test
- Phù hợp với scale hiện tại (< 1M jobs/candidates)

**Nhược điểm:**
- Khó scale riêng từng component (nhưng có thể dùng async workers cho encoding)
- Tight coupling (nhưng có thể giảm bằng dependency injection)

**Khi nào nên chuyển sang Microservice:**
- Khi cần scale encoding riêng (nhiều GPU workers)
- Khi cần deploy riêng cho từng tower
- Khi có team riêng maintain từng service

**Kiến trúc đề xuất:**
```
FastAPI App (Monolith)
├── API Routes
├── Matching Service
├── Job Tower Encoder (có thể async worker)
├── Candidate Tower Encoder (có thể async worker)
├── FAISS Manager
└── Database Repository
```

---

## B. Thiết Kế Dữ Liệu & Lưu Trữ

### B.1. PostgreSQL Schema

```sql
-- ============================================================================
-- Two-Tower Architecture Tables
-- ============================================================================

-- Job Description Two-Tower Table
CREATE TABLE IF NOT EXISTS job_description_two_tower (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(100) UNIQUE NOT NULL,
    title VARCHAR(500) NOT NULL,
    skills TEXT,
    requirement TEXT,
    company VARCHAR(200),
    location VARCHAR(200),
    
    -- 3 separate embeddings (Job Tower)
    title_embedding REAL[] NOT NULL,
    skills_embedding REAL[] NOT NULL,
    requirement_embedding REAL[] NOT NULL,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes
    CONSTRAINT job_id_unique UNIQUE (job_id)
);

-- Indexes for embeddings (GIN index for array similarity search)
CREATE INDEX IF NOT EXISTS idx_job_tt_title_emb 
    ON job_description_two_tower USING GIN (title_embedding);
CREATE INDEX IF NOT EXISTS idx_job_tt_skills_emb 
    ON job_description_two_tower USING GIN (skills_embedding);
CREATE INDEX IF NOT EXISTS idx_job_tt_req_emb 
    ON job_description_two_tower USING GIN (requirement_embedding);

-- Index for job_id lookup
CREATE INDEX IF NOT EXISTS idx_job_tt_job_id 
    ON job_description_two_tower (job_id);


-- Candidate Two-Tower Table
CREATE TABLE IF NOT EXISTS candidate_two_tower (
    id SERIAL PRIMARY KEY,
    candidate_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(200),
    email VARCHAR(200),
    title VARCHAR(500),  -- desired job title or current job title
    skills TEXT,
    experience TEXT,
    
    -- 3 separate embeddings (Candidate Tower)
    title_embedding REAL[] NOT NULL,
    skills_embedding REAL[] NOT NULL,
    experience_embedding REAL[] NOT NULL,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes
    CONSTRAINT candidate_id_unique UNIQUE (candidate_id)
);

-- Indexes for embeddings
CREATE INDEX IF NOT EXISTS idx_candidate_tt_title_emb 
    ON candidate_two_tower USING GIN (title_embedding);
CREATE INDEX IF NOT EXISTS idx_candidate_tt_skills_emb 
    ON candidate_two_tower USING GIN (skills_embedding);
CREATE INDEX IF NOT EXISTS idx_candidate_tt_exp_emb 
    ON candidate_two_tower USING GIN (experience_embedding);

-- Index for candidate_id lookup
CREATE INDEX IF NOT EXISTS idx_candidate_tt_candidate_id 
    ON candidate_two_tower (candidate_id);


-- Reindex Tracking Table (để track reindex progress)
CREATE TABLE IF NOT EXISTS reindex_tracking (
    id SERIAL PRIMARY KEY,
    reindex_type VARCHAR(50) NOT NULL,  -- 'full', 'incremental', 'job', 'candidate'
    status VARCHAR(20) NOT NULL,  -- 'pending', 'running', 'completed', 'failed'
    total_records INTEGER,
    processed_records INTEGER DEFAULT 0,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_reindex_status 
    ON reindex_tracking (status, started_at DESC);
```

### B.2. Lưu Trữ Vector: PostgreSQL vs FAISS

**Đề xuất: Hybrid (PostgreSQL + FAISS)**

#### **PostgreSQL (pgvector extension - nếu có)**
- **Pros**:
  - ACID transactions
  - Dễ query metadata kèm vectors
  - Backup/restore đơn giản
- **Cons**:
  - Chậm hơn FAISS cho ANN search
  - Tốn disk space (vectors lưu trong DB)

#### **FAISS (File-based)**
- **Pros**:
  - Rất nhanh cho ANN search (HNSW index)
  - Memory-mapped files (efficient)
  - Hỗ trợ nhiều index types (Flat, IVF, HNSW)
- **Cons**:
  - Không có ACID (cần sync với PostgreSQL)
  - Cần rebuild khi update

#### **Giải pháp Hybrid:**
1. **PostgreSQL**: Lưu metadata + embeddings (source of truth)
2. **FAISS**: Lưu indices cho fast search (6 indices riêng biệt)
3. **Sync strategy**: 
   - Build FAISS từ PostgreSQL khi start service
   - Update FAISS khi có new/updated records (incremental)
   - Full rebuild định kỳ (nightly)

**Cấu trúc thư mục FAISS:**
```
indices/two_tower/
├── job_title_index.faiss
├── job_title_index.pkl          # ID mapping
├── job_skills_index.faiss
├── job_skills_index.pkl
├── job_requirement_index.faiss
├── job_requirement_index.pkl
├── candidate_title_index.faiss
├── candidate_title_index.pkl
├── candidate_skills_index.faiss
├── candidate_skills_index.pkl
├── candidate_experience_index.faiss
└── candidate_experience_index.pkl
```

---

## C. Pipeline Embedding (Mỗi Tower)

### C.1. Quy Trình Tiền Xử Lý

#### **Job Tower - Field Preprocessing**

```python
def preprocess_job_title(title: str) -> str:
    """
    Preprocess job title.
    - Normalize: lowercase, remove extra spaces
    - Remove special chars (optional)
    - Truncate to max 200 chars
    """
    if not title:
        return ""
    # Normalize
    text = title.strip().lower()
    text = " ".join(text.split())  # Remove extra spaces
    # Truncate
    if len(text) > 200:
        text = text[:200]
    return text

def preprocess_job_skills(skills: str) -> str:
    """
    Preprocess job skills.
    - Tokenize (nếu cần cho Vietnamese)
    - Normalize: lowercase
    - Remove duplicates (nếu skills là list)
    - Truncate to max 1000 chars
    """
    if not skills:
        return ""
    # Normalize
    text = skills.strip().lower()
    # Tokenize Vietnamese nếu cần
    if PYVI_AVAILABLE and requires_vietnamese_tokenization:
        text = vietnamese_tokenize(text)
    # Truncate
    if len(text) > 1000:
        text = text[:1000]
    return text

def preprocess_job_requirements(requirements: str) -> str:
    """
    Preprocess job requirements.
    - Sentence splitting (lấy first 3 sentences nếu quá dài)
    - Normalize
    - Truncate to max 2000 chars
    """
    if not requirements:
        return ""
    # Normalize
    text = requirements.strip().lower()
    # Sentence splitting nếu quá dài
    sentences = text.split('.')
    if len(sentences) > 3 and len(text) > 2000:
        text = '. '.join(sentences[:3]) + '.'
    # Truncate
    if len(text) > 2000:
        text = text[:2000]
    return text
```

#### **Candidate Tower - Field Preprocessing**

```python
def preprocess_candidate_title(title: str) -> str:
    """Similar to job title preprocessing."""
    return preprocess_job_title(title)  # Reuse logic

def preprocess_candidate_skills(skills: str) -> str:
    """Similar to job skills preprocessing."""
    return preprocess_job_skills(skills)  # Reuse logic

def preprocess_candidate_experience(experience: str) -> str:
    """
    Preprocess candidate experience.
    - Similar to requirements preprocessing
    - Extract key points (years, companies, roles)
    - Truncate to max 2000 chars
    """
    if not experience:
        return ""
    text = experience.strip().lower()
    # Truncate
    if len(text) > 2000:
        text = text[:2000]
    return text
```

### C.2. Model & Chiến Lược Embedding

**Giả định hợp lý dựa trên codebase hiện tại:**

- **Model**: `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base` (SentenceTransformer)
- **Dimension**: 768 (PhoBERT base)
- **Normalization**: L2 normalization (cho cosine similarity)
- **Encoding**: Per-field encoding (mỗi field encode riêng)

**Lý do:**
- Model hiện tại đã được optimize cho Vietnamese text
- 768-dim đủ để capture semantic meaning
- L2 normalization cho phép dùng cosine similarity (dot product)

**Có thể fine-tune riêng:**
- Job Tower có thể fine-tune với job-specific data
- Candidate Tower có thể fine-tune với candidate-specific data
- Fine-tuning offline, deploy model mới khi ready

### C.3. Pseudocode: Encode Functions

```python
class JobTowerEncoder:
    def __init__(self, model_name: str = None):
        self.model = SentenceTransformer(model_name or settings.EMBEDDING_MODEL)
        self.dimension = self.model.get_sentence_embedding_dimension()  # 768
    
    def encode_job(self, job: Dict[str, str]) -> Dict[str, List[float]]:
        """
        Encode job into 3 embeddings.
        
        Args:
            job: Dict with keys: 'title', 'skills', 'requirement'
        
        Returns:
            Dict with keys: 'title_embedding', 'skills_embedding', 'requirement_embedding'
        """
        # Preprocess
        title_text = preprocess_job_title(job.get('title', ''))
        skills_text = preprocess_job_skills(job.get('skills', ''))
        req_text = preprocess_job_requirements(job.get('requirement', ''))
        
        # Encode each field
        title_emb = self.model.encode(
            title_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        skills_emb = self.model.encode(
            skills_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        req_emb = self.model.encode(
            req_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Ensure L2 normalization
        title_emb = title_emb / np.linalg.norm(title_emb) if np.linalg.norm(title_emb) > 0 else np.zeros(self.dimension)
        skills_emb = skills_emb / np.linalg.norm(skills_emb) if np.linalg.norm(skills_emb) > 0 else np.zeros(self.dimension)
        req_emb = req_emb / np.linalg.norm(req_emb) if np.linalg.norm(req_emb) > 0 else np.zeros(self.dimension)
        
        return {
            'title_embedding': title_emb.tolist(),
            'skills_embedding': skills_emb.tolist(),
            'requirement_embedding': req_emb.tolist()
        }

class CandidateTowerEncoder:
    def __init__(self, model_name: str = None):
        self.model = SentenceTransformer(model_name or settings.EMBEDDING_MODEL)
        self.dimension = self.model.get_sentence_embedding_dimension()  # 768
    
    def encode_candidate(self, candidate: Dict[str, str]) -> Dict[str, List[float]]:
        """
        Encode candidate into 3 embeddings.
        
        Args:
            candidate: Dict with keys: 'title', 'skills', 'experience'
        
        Returns:
            Dict with keys: 'title_embedding', 'skills_embedding', 'experience_embedding'
        """
        # Preprocess
        title_text = preprocess_candidate_title(candidate.get('title', ''))
        skills_text = preprocess_candidate_skills(candidate.get('skills', ''))
        exp_text = preprocess_candidate_experience(candidate.get('experience', ''))
        
        # Encode each field (similar to JobTowerEncoder)
        title_emb = self.model.encode(title_text, ...)
        skills_emb = self.model.encode(skills_text, ...)
        exp_emb = self.model.encode(exp_text, ...)
        
        # Normalize
        title_emb = normalize(title_emb)
        skills_emb = normalize(skills_emb)
        exp_emb = normalize(exp_emb)
        
        return {
            'title_embedding': title_emb.tolist(),
            'skills_embedding': skills_emb.tolist(),
            'experience_embedding': exp_emb.tolist()
        }
```

---

## D. Indexing & Update Strategy

### D.1. Per-Field Index vs Concat Vectors

**Đề xuất: Per-Field Index (6 indices riêng biệt)**

**Per-Field Index (Recommended):**
- **Pros**:
  - Flexible: có thể search riêng từng field
  - Dễ tune weights cho từng field
  - Có thể skip field nếu embedding invalid
  - Dễ debug (xem kết quả từng field)
- **Cons**:
  - Nhiều indices cần maintain (6 indices)
  - Tốn memory hơn (nhưng có thể memory-map)

**Concat Vectors:**
- **Pros**:
  - Chỉ 2 indices (1 cho Jobs, 1 cho Candidates)
  - Đơn giản hơn
- **Cons**:
  - Không flexible (không thể search riêng field)
  - Khó tune weights
  - Vector dimension lớn (768 * 3 = 2304)

**Quyết định: Per-Field Index** vì lợi ích về flexibility và explainability.

### D.2. Reindex Strategy

#### **Full Reindex**
- **Khi nào**: 
  - Lần đầu build indices
  - Sau khi update model embeddings
  - Định kỳ (nightly/weekly) để optimize
- **Process**:
  1. Load tất cả records từ PostgreSQL
  2. Build 6 FAISS indices từ đầu
  3. Save indices to disk
  4. Update metadata (reindex_tracking table)

#### **Incremental Update**
- **Khi nào**: 
  - Khi có new/updated records
  - Real-time hoặc batch (mỗi 5-10 phút)
- **Process**:
  1. Track updated records (dùng `updated_at` timestamp)
  2. Load only new/updated records
  3. Update FAISS indices (remove old, add new)
  4. Save indices

#### **Soft-Delete Handling**
- **Strategy**: 
  - Mark deleted records trong PostgreSQL (soft delete)
  - Remove từ FAISS index khi reindex
  - Hoặc maintain deleted_ids list và filter khi search

### D.3. Batching & Parallelization

```python
def build_indices_parallel(db: Session, batch_size: int = 1000):
    """
    Build FAISS indices with batching and parallelization.
    """
    from concurrent.futures import ThreadPoolExecutor
    
    # Load all records in batches
    total_jobs = db.query(JobDescriptionTwoTower).count()
    total_candidates = db.query(CandidateTwoTower).count()
    
    # Process Jobs
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Build 3 job indices in parallel
        futures = [
            executor.submit(build_job_index, db, 'title', batch_size),
            executor.submit(build_job_index, db, 'skills', batch_size),
            executor.submit(build_job_index, db, 'requirement', batch_size)
        ]
        job_results = [f.result() for f in futures]
    
    # Process Candidates
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Build 3 candidate indices in parallel
        futures = [
            executor.submit(build_candidate_index, db, 'title', batch_size),
            executor.submit(build_candidate_index, db, 'skills', batch_size),
            executor.submit(build_candidate_index, db, 'experience', batch_size)
        ]
        candidate_results = [f.result() for f in futures]
    
    return job_results, candidate_results
```

---

## E. Matching Pipeline (3-Stage)

### E.1. Stage 1: Per-Field ANN Search

```python
def stage1_per_field_search(
    candidate_embeddings: Dict[str, List[float]],
    faiss_manager: TwoTowerFAISSManager,
    top_n_per_field: int = 1000
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Stage 1: Search per-field using ANN.
    
    Args:
        candidate_embeddings: Dict with 'title_embedding', 'skills_embedding', 'experience_embedding'
        faiss_manager: FAISS manager with 6 indices
        top_n_per_field: Top N results per field
    
    Returns:
        Dict with keys: 'title_results', 'skills_results', 'experience_results'
        Each value is List of (job_id, similarity_score) tuples
    """
    from concurrent.futures import ThreadPoolExecutor
    
    results = {}
    
    # Parallel search across 3 fields
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            'title': executor.submit(
                faiss_manager.search_job_by_field,
                candidate_embeddings['title_embedding'],
                'title',
                top_n_per_field
            ),
            'skills': executor.submit(
                faiss_manager.search_job_by_field,
                candidate_embeddings['skills_embedding'],
                'skills',
                top_n_per_field
            ),
            'experience': executor.submit(
                faiss_manager.search_job_by_field,
                candidate_embeddings['experience_embedding'],
                'requirement',  # Match candidate experience với job requirement
                top_n_per_field
            )
        }
        
        results['title_results'] = futures['title'].result()
        results['skills_results'] = futures['skills'].result()
        results['experience_results'] = futures['experience'].result()
    
    return results
```

### E.2. Stage 2: Candidate Union & Coarse Scoring

```python
def stage2_merge_and_score(
    stage1_results: Dict[str, List[Tuple[str, float]]],
    weights: Dict[str, float] = None
) -> Dict[str, float]:
    """
    Stage 2: Merge results and compute weighted score.
    
    Args:
        stage1_results: Results from Stage 1
        weights: Field weights (default: {'title': 0.2, 'skills': 0.4, 'experience': 0.4})
    
    Returns:
        Dict mapping job_id -> weighted_score
    """
    if weights is None:
        weights = {'title': 0.2, 'skills': 0.4, 'experience': 0.4}
    
    # Collect all unique job_ids
    all_job_ids = set()
    for field_results in stage1_results.values():
        all_job_ids.update([job_id for job_id, _ in field_results])
    
    # Compute weighted score for each job
    job_scores = {}
    job_field_scores = {}  # For explainability
    
    for job_id in all_job_ids:
        # Get scores from each field
        title_score = next(
            (score for jid, score in stage1_results['title_results'] if jid == job_id),
            0.0
        )
        skills_score = next(
            (score for jid, score in stage1_results['skills_results'] if jid == job_id),
            0.0
        )
        experience_score = next(
            (score for jid, score in stage1_results['experience_results'] if jid == job_id),
            0.0
        )
        
        # Weighted sum
        weighted_score = (
            title_score * weights['title'] +
            skills_score * weights['skills'] +
            experience_score * weights['experience']
        )
        
        job_scores[job_id] = weighted_score
        job_field_scores[job_id] = {
            'title': title_score,
            'skills': skills_score,
            'experience': experience_score
        }
    
    return job_scores, job_field_scores
```

### E.3. Stage 3: Rerank (Optional)

```python
def stage3_rerank(
    top_jobs: List[Tuple[str, float]],
    candidate_text: Dict[str, str],
    job_texts: Dict[str, Dict[str, str]],
    reranker: CrossEncoderReranker = None,
    top_m: int = 100
) -> List[Tuple[str, float]]:
    """
    Stage 3: Rerank top M jobs using cross-encoder.
    
    Args:
        top_jobs: List of (job_id, score) from Stage 2
        candidate_text: Dict with 'title', 'skills', 'experience'
        job_texts: Dict mapping job_id -> {'title', 'skills', 'requirement'}
        reranker: Cross-encoder reranker (optional)
        top_m: Rerank top M jobs
    
    Returns:
        Reranked list of (job_id, final_score)
    """
    if reranker is None:
        # No reranking, return as-is
        return top_jobs[:top_m]
    
    # Prepare pairs for reranking
    pairs = []
    for job_id, score in top_jobs[:top_m]:
        candidate_str = f"{candidate_text['title']} | {candidate_text['skills']} | {candidate_text['experience']}"
        job_str = f"{job_texts[job_id]['title']} | {job_texts[job_id]['skills']} | {job_texts[job_id]['requirement']}"
        pairs.append((candidate_str, job_str))
    
    # Rerank
    rerank_scores = reranker.predict(pairs)
    
    # Combine with original scores (weighted)
    final_scores = []
    for i, (job_id, original_score) in enumerate(top_jobs[:top_m]):
        rerank_score = rerank_scores[i]
        # Combine: 70% rerank, 30% original (có thể tune)
        final_score = 0.7 * rerank_score + 0.3 * original_score
        final_scores.append((job_id, final_score))
    
    # Sort by final score
    final_scores.sort(key=lambda x: x[1], reverse=True)
    return final_scores
```

### E.4. Formula Tổng Điểm & Weights

**Default Weights:**
```python
DEFAULT_WEIGHTS = {
    'title': 0.2,      # 20% - Title matching
    'skills': 0.4,     # 40% - Skills matching (quan trọng nhất)
    'experience': 0.4  # 40% - Experience/Requirement matching
}
```

**Scoring Formula:**
```python
final_score = (
    title_similarity * w_title +
    skills_similarity * w_skills +
    experience_similarity * w_experience
)
```

**Cách Tune Weights:**
1. **A/B Testing**: Thử các weight combinations, đo recall@10, MRR
2. **Grid Search**: Thử weights từ 0.1 đến 0.5, step 0.1
3. **Learning to Rank**: Train model để học weights tự động
4. **Domain-specific**: Tune theo domain (ví dụ: IT jobs → skills weight cao hơn)

**Pseudocode Service Layer:**

```python
class TwoTowerMatchingService:
    def find_jobs_for_candidate(
        self,
        candidate_id: str,
        top_k: int = 10,
        weights: Dict[str, float] = None,
        use_reranking: bool = True
    ) -> List[Dict]:
        """
        Main matching function.
        """
        # Get candidate embeddings
        candidate = self.repository.get_candidate_two_tower(candidate_id)
        candidate_embeddings = {
            'title_embedding': candidate.title_embedding,
            'skills_embedding': candidate.skills_embedding,
            'experience_embedding': candidate.experience_embedding
        }
        
        # Stage 1: Per-field ANN search
        stage1_results = stage1_per_field_search(
            candidate_embeddings,
            self.faiss_manager,
            top_n_per_field=1000
        )
        
        # Stage 2: Merge and score
        job_scores, job_field_scores = stage2_merge_and_score(
            stage1_results,
            weights=weights or DEFAULT_WEIGHTS
        )
        
        # Sort by score
        top_jobs = sorted(job_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Stage 3: Rerank (optional)
        if use_reranking and len(top_jobs) > 100:
            top_jobs = stage3_rerank(
                top_jobs,
                candidate_text={'title': candidate.title, ...},
                job_texts={...},  # Load from DB
                reranker=self.reranker,
                top_m=100
            )
        
        # Format results
        results = []
        for job_id, final_score in top_jobs[:top_k]:
            results.append({
                'job_id': job_id,
                'score': final_score,
                'explain': job_field_scores[job_id],  # Per-field scores
                'metadata': {...}  # Job metadata
            })
        
        return results
```

---

## F. API Contract (Endpoints)

### F.1. POST /api/v2/search/jobs

**Request:**
```json
{
  "candidate_id": "CAND001",
  "top_k": 10,
  "weights": {
    "title": 0.2,
    "skills": 0.4,
    "experience": 0.4
  },
  "use_reranking": true
}
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
      "score": 0.85,
      "explain": {
        "title": 0.75,
        "skills": 0.90,
        "experience": 0.88
      },
      "metadata": {
        "description": "...",
        "requirements": "..."
      }
    }
  ]
}
```

### F.2. POST /api/v2/search/candidates

**Request:**
```json
{
  "job_id": "JD001",
  "top_k": 10,
  "weights": {
    "title": 0.2,
    "skills": 0.4,
    "experience": 0.4
  }
}
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
      "score": 0.85,
      "explain": {
        "title": 0.75,
        "skills": 0.90,
        "experience": 0.88
      }
    }
  ]
}
```

### F.3. POST /api/v2/index/job

**Request:**
```json
{
  "job_id": "JD001",
  "title": "Senior Software Engineer",
  "skills": "Python, FastAPI, PostgreSQL",
  "requirement": "5+ years experience...",
  "company": "Tech Corp",
  "location": "Ho Chi Minh City"
}
```

**Response:**
```json
{
  "status": "success",
  "job_id": "JD001",
  "message": "Job indexed successfully"
}
```

### F.4. POST /api/v2/index/candidate

**Request:**
```json
{
  "candidate_id": "CAND001",
  "title": "Software Engineer",
  "skills": "Python, FastAPI, PostgreSQL",
  "experience": "5 years at Company X...",
  "name": "Nguyen Van A",
  "email": "a@example.com"
}
```

**Response:**
```json
{
  "status": "success",
  "candidate_id": "CAND001",
  "message": "Candidate indexed successfully"
}
```

### F.5. POST /api/v2/reindex

**Request:**
```json
{
  "reindex_type": "full",  // or "incremental", "job", "candidate"
  "force": false
}
```

**Response:**
```json
{
  "status": "accepted",
  "reindex_id": 123,
  "message": "Reindex job started",
  "estimated_time_minutes": 30
}
```

### F.6. GET /api/v2/health

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "indices": {
    "job_title": "loaded",
    "job_skills": "loaded",
    "job_requirement": "loaded",
    "candidate_title": "loaded",
    "candidate_skills": "loaded",
    "candidate_experience": "loaded"
  },
  "database": "connected",
  "total_jobs": 10000,
  "total_candidates": 5000
}
```

---

## G. Performance, Scalability & Ops

### G.1. Latency Expectations

**Per-Stage Latency (p95):**
- **Stage 1 (ANN Search)**: 50-100ms (3 parallel searches)
- **Stage 2 (Merge & Score)**: 10-20ms
- **Stage 3 (Rerank)**: 200-500ms (nếu dùng cross-encoder)
- **Total (no rerank)**: 60-120ms
- **Total (with rerank)**: 260-620ms

**Bottlenecks:**
1. **Embedding Compute**: Nếu encode real-time → 100-200ms per field
2. **FAISS IO**: Load indices từ disk → 1-2s (chỉ lần đầu)
3. **Cross-encoder Rerank**: 200-500ms cho 100 candidates

### G.2. Caching Suggestions

**1. Embedding Cache (Redis):**
- Cache candidate embeddings (TTL: 1 hour)
- Key: `candidate_emb:{candidate_id}`
- Value: JSON với 3 embeddings

**2. Search Result Cache (Redis):**
- Cache top 10 results cho frequent candidates
- Key: `search_results:{candidate_id}:{weights_hash}`
- TTL: 30 minutes

**3. FAISS Index Cache (Memory):**
- Load indices vào memory khi start service
- Memory-map files để share giữa processes

### G.3. Horizontal Scaling

**FAISS Sharding:**
- Shard by prefix: `job_id` prefix → shard index
- Ví dụ: `JD001` → shard 0, `JD1001` → shard 1
- Mỗi shard có 6 indices riêng

**IVF + HNSW Settings:**
```python
# For large datasets (> 1M records)
index_params = {
    "index_type": "IVF_HNSW",  # IVF for clustering, HNSW for search
    "nlist": 4096,  # Number of clusters
    "M": 32,  # HNSW connections
    "ef_construction": 200,
    "ef_search": 128
}
```

**Recommended FAISS Parameters:**
- **Small dataset (< 100K)**: HNSW với M=32, ef_search=64
- **Medium dataset (100K-1M)**: HNSW với M=32, ef_search=128
- **Large dataset (> 1M)**: IVF_HNSW với nlist=4096

### G.4. Observability

**Metrics to Collect:**
1. **QPS**: Queries per second
2. **Latency**: p50, p95, p99
3. **Recall@K**: Recall@1, Recall@5, Recall@10
4. **MRR**: Mean Reciprocal Rank
5. **Reindex Time**: Time to rebuild indices
6. **Cache Hit Rate**: Embedding cache, search cache
7. **Error Rate**: 4xx, 5xx errors

**Tools:**
- **Prometheus + Grafana**: Metrics dashboard
- **ELK Stack**: Log aggregation
- **Sentry**: Error tracking

### G.5. Infrastructure Recommendations

**CPU/GPU:**
- **Encoding**: CPU (4-8 cores) hoặc GPU (nếu batch encoding)
- **FAISS Search**: CPU (single-threaded, nhưng có thể parallelize queries)
- **Reranking**: GPU (nếu dùng cross-encoder)

**Memory:**
- **FAISS Indices**: ~500MB per 100K records (6 indices)
- **Model Loading**: ~500MB per SentenceTransformer model
- **Total**: 2-4GB cho 1M records

**Storage:**
- **FAISS Files**: ~500MB per 100K records
- **PostgreSQL**: ~1GB per 100K records (với embeddings)

**Redis:**
- **Cache**: 1-2GB (tùy cache size)

---

## H. Migration Plan (Step-by-Step)

### H.1. Phase 1: Schema & Infrastructure (Week 1)

**Tasks:**
1. ✅ Create new tables: `job_description_two_tower`, `candidate_two_tower`, `reindex_tracking`
2. ✅ Create migration script: `scripts/migrate_to_two_tower_schema.py`
3. ✅ Test schema với sample data

**Rollback:** Drop new tables nếu có issue

### H.2. Phase 2: Encoder Services (Week 2)

**Tasks:**
1. ✅ Implement `JobTowerEncoder` class
2. ✅ Implement `CandidateTowerEncoder` class
3. ✅ Unit tests cho encoders
4. ✅ Integration test: encode sample jobs/candidates

**Files to Create:**
- `src/embeddings/job_tower_encoder.py`
- `src/embeddings/candidate_tower_encoder.py`
- `tests/test_job_tower_encoder.py`
- `tests/test_candidate_tower_encoder.py`

### H.3. Phase 3: FAISS Manager (Week 2-3)

**Tasks:**
1. ✅ Implement `TwoTowerFAISSManager` với 6 indices
2. ✅ Implement build/load/save indices
3. ✅ Implement incremental update
4. ✅ Test với sample data

**Files to Create:**
- `src/vector_search/two_tower_faiss_manager.py`
- `scripts/build_two_tower_faiss.py`

### H.4. Phase 4: Matching Service (Week 3-4)

**Tasks:**
1. ✅ Implement `TwoTowerMatchingService` với 3-stage pipeline
2. ✅ Implement per-field search
3. ✅ Implement merge & scoring
4. ✅ Implement reranking (optional)
5. ✅ Unit tests

**Files to Create:**
- `src/services/two_tower_matching_service.py`
- `tests/test_two_tower_matching.py`

### H.5. Phase 5: API Endpoints (Week 4)

**Tasks:**
1. ✅ Implement API routes (`/api/v2/*`)
2. ✅ Request/response schemas
3. ✅ Integration tests
4. ✅ API documentation

**Files to Create:**
- `src/api/two_tower_routes.py`
- `src/api/two_tower_schemas.py`

### H.6. Phase 6: Indexing Pipeline (Week 5)

**Tasks:**
1. ✅ Implement batch indexing script
2. ✅ Implement incremental update script
3. ✅ Test với full dataset
4. ✅ Monitor reindex progress

**Files to Create:**
- `scripts/batch_reindex_two_tower.py`
- `scripts/incremental_upsert_two_tower.py`

### H.7. Phase 7: A/B Testing (Week 6)

**Tasks:**
1. ✅ Route X% traffic (10%) to new Two-Tower API
2. ✅ Collect metrics: recall@10, MRR, latency
3. ✅ Compare với legacy system
4. ✅ Tune weights nếu cần

**Rollback:** Route 0% traffic nếu metrics worse

### H.8. Phase 8: Full Migration (Week 7)

**Tasks:**
1. ✅ Route 100% traffic to Two-Tower
2. ✅ Monitor for 1 week
3. ✅ Retire legacy pipeline (nếu metrics OK)
4. ✅ Cleanup old code/tables (optional)

**Rollback Plan:**
- Keep legacy code for 1 month
- Can route back to legacy nếu có issue
- Database: Keep old tables, migrate data back nếu cần

---

## I. Testing & Evaluation

### I.1. Offline Evaluation Dataset

**Ground Truth Dataset:**
- **Source**: Manual annotations hoặc historical click data
- **Format**: JSON với positive/negative pairs
- **Size**: 1000-5000 pairs (tùy dataset size)

```json
[
  {
    "candidate_id": "CAND001",
    "job_id": "JD001",
    "label": 1,  // 1 = match, 0 = no match
    "ground_truth_scores": {
      "title": 0.75,
      "skills": 0.90,
      "experience": 0.88
    }
  }
]
```

### I.2. Evaluation Metrics

**Ranking Metrics:**
- **Recall@K**: Recall@1, Recall@5, Recall@10
- **MRR**: Mean Reciprocal Rank
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Precision@K**: Precision@1, Precision@5, Precision@10

**Similarity Metrics:**
- **Label Correlation**: Correlation giữa predicted scores và labels
- **Field Correlations**: Correlation với từng field similarity

**Code Example:**
```python
def evaluate_two_tower(
    test_pairs: List[Dict],
    matching_service: TwoTowerMatchingService
) -> Dict[str, float]:
    """
    Evaluate Two-Tower matching service.
    """
    recalls = {1: [], 5: [], 10: []}
    mrr_scores = []
    
    for pair in test_pairs:
        candidate_id = pair['candidate_id']
        true_job_id = pair['job_id']
        label = pair['label']
        
        # Get recommendations
        recommendations = matching_service.find_jobs_for_candidate(
            candidate_id,
            top_k=10
        )
        
        # Check if true_job_id in top K
        job_ids = [r['job_id'] for r in recommendations]
        for k in [1, 5, 10]:
            if true_job_id in job_ids[:k]:
                recalls[k].append(1.0)
            else:
                recalls[k].append(0.0)
        
        # MRR
        if true_job_id in job_ids:
            rank = job_ids.index(true_job_id) + 1
            mrr_scores.append(1.0 / rank)
        else:
            mrr_scores.append(0.0)
    
    return {
        'recall@1': np.mean(recalls[1]),
        'recall@5': np.mean(recalls[5]),
        'recall@10': np.mean(recalls[10]),
        'mrr': np.mean(mrr_scores)
    }
```

### I.3. Integration Tests

**Test Scenarios:**
1. **Index → Search**: Index job/candidate, search và verify results
2. **Batch Indexing**: Index 100 jobs, verify all indexed
3. **Incremental Update**: Update 1 job, verify FAISS updated
4. **Reindex**: Full reindex, verify indices rebuilt

**Example:**
```python
def test_index_and_search():
    """Test indexing and searching."""
    # Index job
    job_data = {
        "job_id": "TEST_JD001",
        "title": "Software Engineer",
        "skills": "Python, FastAPI",
        "requirement": "3+ years experience"
    }
    api_client.post("/api/v2/index/job", json=job_data)
    
    # Index candidate
    candidate_data = {
        "candidate_id": "TEST_CAND001",
        "title": "Software Engineer",
        "skills": "Python, FastAPI, PostgreSQL",
        "experience": "5 years at Company X"
    }
    api_client.post("/api/v2/index/candidate", json=candidate_data)
    
    # Search
    response = api_client.post("/api/v2/search/jobs", json={
        "candidate_id": "TEST_CAND001",
        "top_k": 10
    })
    
    assert response.status_code == 200
    assert len(response.json()["matches"]) > 0
    assert response.json()["matches"][0]["job_id"] == "TEST_JD001"
```

### I.4. End-to-End Smoke Tests

**Scenarios:**
1. Health check → all indices loaded
2. Search với invalid candidate_id → return 404
3. Search với empty results → return empty list
4. Reindex → verify status updated

### I.5. Load Tests

**k6 Script Example:**
```javascript
import http from 'k6/http';
import { check } from 'k6';

export let options = {
  stages: [
    { duration: '1m', target: 10 },   // Ramp up to 10 users
    { duration: '3m', target: 10 },   // Stay at 10 users
    { duration: '1m', target: 50 },   // Ramp up to 50 users
    { duration: '3m', target: 50 },   // Stay at 50 users
  ],
};

export default function () {
  let response = http.post('http://localhost:8000/api/v2/search/jobs', JSON.stringify({
    candidate_id: 'CAND001',
    top_k: 10
  }), {
    headers: { 'Content-Type': 'application/json' },
  });
  
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });
}
```

**Expected Results:**
- **QPS**: 100-200 requests/second
- **p95 Latency**: < 500ms (no rerank), < 1000ms (with rerank)
- **Error Rate**: < 1%

---

## J. Security & Privacy

### J.1. Data Privacy

- **PII Handling**: 
  - Candidate email/name: Chỉ lưu nếu cần, có thể hash
  - Resume text: Encrypt nếu lưu trong DB
- **Access Control**: 
  - API authentication (JWT tokens)
  - Role-based access (admin, user)

### J.2. Input Validation

- **SQL Injection**: Dùng parameterized queries (SQLAlchemy)
- **XSS**: Sanitize text input
- **Rate Limiting**: Limit requests per IP/user

### J.3. Model Security

- **Model Files**: Verify checksums khi load
- **Embedding Validation**: Check vector dimensions, norms

---

## K. Deliverables & Acceptance Criteria

### K.1. Code Changes

**New Files:**
- `src/embeddings/job_tower_encoder.py`
- `src/embeddings/candidate_tower_encoder.py`
- `src/vector_search/two_tower_faiss_manager.py`
- `src/services/two_tower_matching_service.py`
- `src/api/two_tower_routes.py`
- `src/api/two_tower_schemas.py`
- `src/database/two_tower_repository.py`

**Modified Files:**
- `src/database/models.py` (add Two-Tower models)
- `main.py` (register new routes)

### K.2. Scripts

- `scripts/batch_reindex_two_tower.py` - Full reindex
- `scripts/incremental_upsert_two_tower.py` - Incremental update
- `scripts/migrate_to_two_tower_schema.py` - Schema migration
- `scripts/evaluate_two_tower.py` - Evaluation script

### K.3. Tests

- `tests/test_job_tower_encoder.py`
- `tests/test_candidate_tower_encoder.py`
- `tests/test_two_tower_matching.py`
- `tests/test_two_tower_api.py`
- `tests/test_two_tower_faiss.py`

### K.4. Documentation

- `docs/two_tower_design.md` (this document)
- `docs/two_tower_api_spec.md` - API specification
- `docs/two_tower_runbook.md` - Operations runbook

### K.5. Acceptance Criteria

**Functional:**
- ✅ All API endpoints work as specified
- ✅ Matching returns results với explainability
- ✅ Reindex completes successfully
- ✅ Incremental update works

**Performance:**
- ✅ p95 latency < 500ms (no rerank), < 1000ms (with rerank)
- ✅ Recall@10 >= baseline (legacy system)
- ✅ Error rate < 1%

**Quality:**
- ✅ Unit test coverage > 80%
- ✅ Integration tests pass
- ✅ Load tests pass (100 QPS)

---

## 📅 Timeline Estimate

**Total: 7 weeks**

- **Week 1**: Schema & Infrastructure
- **Week 2-3**: Encoders & FAISS Manager
- **Week 3-4**: Matching Service
- **Week 4**: API Endpoints
- **Week 5**: Indexing Pipeline
- **Week 6**: A/B Testing
- **Week 7**: Full Migration

**Phân chia tasks:**
- **Wireframe/Design**: Week 1 (done)
- **Implement**: Week 2-5
- **Test**: Week 5-6
- **Deploy**: Week 6-7

---

## 📝 Notes

- **Giả định hợp lý**: Model dimension = 768 (PhoBERT base), có thể thay đổi
- **Model name**: Sử dụng model hiện tại, có thể fine-tune sau
- **Weights**: Default weights có thể tune dựa trên evaluation
- **FAISS params**: Có thể tune dựa trên dataset size

---

**Version**: 1.0  
**Last Updated**: 2024-12-04  
**Author**: AI System Architect


