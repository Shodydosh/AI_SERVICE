# BÁO CÁO KIẾN TRÚC HỆ THỐNG

## TỔNG QUAN

Hệ thống AI Job Recommendation Service sử dụng kiến trúc **Two-Tower Architecture** để match Job Descriptions (JD) với Candidates.

---

## KIẾN TRÚC TỔNG QUAN

### 1. Entry Point

**File:** `main.py`

```python
if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )
```

- Khởi động FastAPI server
- Entry point: `src.api.main:app`

---

### 2. API Layer

**File:** `src/api/main.py`

- FastAPI application với version 2.0.0
- CORS middleware được bật
- Routes được include từ `src.api.two_tower_routes` với prefix `/api/v2`
- Static files được mount tại `/static`

**File:** `src/api/two_tower_routes.py`

Các endpoints chính:
- `POST /api/v2/search/jobs` - Tìm jobs cho candidate
- `POST /api/v2/search/candidates` - Tìm candidates cho job
- `POST /api/v2/index/job` - Index một job mới
- `POST /api/v2/index/candidate` - Index một candidate mới
- `POST /api/v2/reindex` - Trigger reindex operation
- `GET /api/v2/health` - Health check

---

### 3. Service Layer

#### 3.1. TwoTowerMatchingService

**File:** `src/services/two_tower_matching_service.py`

**Class:** `TwoTowerMatchingService`

**Khởi tạo:**
```python
def __init__(
    self,
    db: Session,
    model_path: Optional[str] = None,
    device: str = 'cpu'
):
```

- Load model từ `outputs_improved/best_model_improved.pt` (default)
- Model: `TwoTowerModel` với:
  - `candidate_model_name`: "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
  - `job_model_name`: "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
  - `output_dim`: 768

**Phương thức chính:**

1. `find_jobs_for_candidate(candidate_id: str, top_k: int = 10)`
   - Lấy candidate từ database
   - Build candidate text: `"Title: {title} | Skills: {skills} | Experience: {experience}"`
   - Lấy tất cả jobs từ database
   - Build job text: `"Title: {title} | Skills: {skills} | Requirements: {requirement}"`
   - Encode candidate: `model.encode_candidates([candidate_text])[0]`
   - Encode jobs theo batch (batch_size=32)
   - Tính cosine similarity: `np.dot(job_embs, candidate_emb_np)`
   - Lấy top K: `np.argsort(similarities)[::-1][:top_k]`
   - Trả về list jobs với score

2. `find_candidates_for_job(job_id: str, top_k: int = 10)`
   - Tương tự như trên nhưng ngược lại (tìm candidates cho job)

#### 3.2. OptimizedEmbeddingService

**File:** `src/services/embedding_service.py`

**Class:** `OptimizedEmbeddingService`

- Cache TTL: 12 giờ (default)
- Batch size: 100 (default)
- Sử dụng `CandidateTowerEncoder` và `JobTowerEncoder`
- Lưu embeddings vào PostgreSQL qua `MultiFieldEmbeddingRepository`

**Phương thức:**
- `get_candidate_embedding()` - Lấy embedding cho candidate (có cache)
- `get_job_embedding()` - Lấy embedding cho job (có cache)

---

### 4. Embedding Layer

#### 4.1. TwoTowerModel

**File:** `two_tower/model.py`

**Class:** `TwoTowerModel`

**Cấu trúc:**
- `candidate_tower`: Tower encoder cho candidates
- `job_tower`: Tower encoder cho jobs
- Mỗi Tower có:
  - Backbone: SentenceTransformer model
  - Projection: `nn.Sequential(Linear -> ReLU -> Dropout -> Linear)`
  - Output được normalize (L2)

**Phương thức:**
- `encode_candidates(texts: List[str])` - Encode candidate texts
- `encode_jobs(texts: List[str])` - Encode job texts
- `forward(candidate_texts, job_texts)` - Tính similarity matrix

#### 4.2. CandidateTowerEncoder

**File:** `src/embeddings/candidate_tower_encoder.py`

**Class:** `CandidateTowerEncoder`

**Chức năng:**
- Encode candidate thành 3 embeddings riêng biệt:
  - `title_embedding`: Từ title
  - `skills_embedding`: Từ skills
  - `experience_embedding`: Từ experience

**Preprocessing:**
- Title: Translate Vietnamese → English, lowercase, max 200 chars
- Skills: Translate Vietnamese → English, lowercase, max 1000 chars
- Experience: Translate Vietnamese → English, lowercase, max 2000 chars

**Model:**
- Sử dụng model từ `settings.EMBEDDING_MODEL`
- Fallback: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
- Nếu model là SimCSE-Vietnamese → tokenize bằng pyvi

#### 4.3. JobTowerEncoder

**File:** `src/embeddings/job_tower_encoder.py`

**Class:** `JobTowerEncoder`

**Chức năng:**
- Encode job thành 3 embeddings riêng biệt:
  - `title_embedding`: Từ title
  - `skills_embedding`: Từ skills
  - `requirement_embedding`: Từ requirements

**Preprocessing:**
- Title: Translate Vietnamese → English, lowercase, max 200 chars
- Skills: Translate Vietnamese → English, lowercase, max 1000 chars
- Requirements: Translate Vietnamese → English, lowercase, max 2000 chars, chỉ lấy 3 câu đầu nếu quá dài

---

### 5. Database Layer

#### 5.1. TwoTowerRepository

**File:** `src/database/two_tower_repository.py`

**Class:** `TwoTowerRepository`

**Models:**
- `JobDescriptionTwoTower`: Lưu job với 3 embeddings (title, skills, requirement)
- `CandidateTwoTower`: Lưu candidate với 3 embeddings (title, skills, experience)
- `ReindexTracking`: Tracking reindex operations

**Phương thức:**
- `create_job()` - Tạo/update job
- `get_job()` - Lấy job theo job_id
- `get_all_jobs()` - Lấy tất cả jobs
- `create_candidate()` - Tạo/update candidate
- `get_candidate()` - Lấy candidate theo candidate_id
- `get_all_candidates()` - Lấy tất cả candidates

#### 5.2. Models

**File:** `src/database/models.py`

**Các models:**
- `JobDescriptionEmbedding`: Model cũ (single embedding)
- `CandidateEmbedding`: Model cũ (single embedding)
- `JobDescriptionTwoTower`: Model mới (3 embeddings)
- `CandidateTwoTower`: Model mới (3 embeddings)

---

## LUỒNG XỬ LÝ (FLOW)

### Flow 1: Index Job/Candidate

```
POST /api/v2/index/job
  ↓
JobTowerEncoder.encode_job(title, skills, requirement)
  ↓
Tạo 3 embeddings: title_embedding, skills_embedding, requirement_embedding
  ↓
TwoTowerRepository.create_job()
  ↓
Lưu vào PostgreSQL (JobDescriptionTwoTower table)
```

### Flow 2: Search Jobs for Candidate

```
POST /api/v2/search/jobs
  ↓
TwoTowerMatchingService.find_jobs_for_candidate(candidate_id, top_k)
  ↓
1. Lấy candidate từ database
2. Build candidate text: "Title: {title} | Skills: {skills} | Experience: {experience}"
3. Encode candidate: model.encode_candidates([candidate_text])[0]
4. Lấy tất cả jobs từ database
5. Build job texts cho tất cả jobs
6. Encode jobs theo batch (batch_size=32)
7. Tính cosine similarity: np.dot(job_embeddings, candidate_embedding)
8. Sort và lấy top K: np.argsort(similarities)[::-1][:top_k]
9. Trả về results với score
```

### Flow 3: Recommendation Script

**File:** `scripts/recommend_jobs_for_candidates.py`

```
1. Load TwoTowerModel từ checkpoint
2. Initialize RuleMatcher
3. Load candidates và jobs từ database
4. Pre-compute job embeddings (tất cả jobs)
5. Với mỗi candidate:
   a. Encode candidate
   b. Tính similarity với tất cả jobs
   c. Lấy top K jobs
   d. Apply rule matching (RuleMatcher.evaluate_match)
   e. In kết quả
```

---

## CÁCH HỆ THỐNG LOAD/LƯU/DÙNG PRECOMPUTED EMBEDDINGS

### 1. Lưu Embeddings

**Trong TwoTowerRepository:**
- Embeddings được lưu trực tiếp vào PostgreSQL dưới dạng `ARRAY(Float)`
- Mỗi record có 3 embeddings:
  - Job: `title_embedding`, `skills_embedding`, `requirement_embedding`
  - Candidate: `title_embedding`, `skills_embedding`, `experience_embedding`

**Trong OptimizedEmbeddingService:**
- Có cache layer (12 giờ TTL)
- Check cache trước → check database → compute mới nếu cần
- Lưu vào database sau khi compute

### 2. Load Embeddings

**Trong TwoTowerMatchingService:**
- Không load precomputed embeddings từ database
- **Tính toán real-time**: Build text từ database records → encode bằng model
- Code hiện tại không sử dụng precomputed embeddings từ database

**Trong recommend_jobs_for_candidates.py:**
- Pre-compute job embeddings một lần cho tất cả jobs
- Lưu trong memory: `job_embeddings = model.encode_jobs(job_texts)`
- Reuse cho tất cả candidates

### 3. FAISS Index

**File:** `two_tower/inference.py`

**Class:** `JobRecommender`

- Load job embeddings từ pickle file: `load_embeddings(Path(job_embeddings_path))`
- Build FAISS index: `faiss.IndexHNSWFlat(dim, 32)` hoặc `faiss.IndexFlatIP(dim)`
- Add embeddings vào index: `index.add(job_embeddings)`
- Search: `index.search(candidate_emb, top_k)`

**Lưu ý:** FAISS index không được sử dụng trong `TwoTowerMatchingService` hiện tại. Chỉ có trong `JobRecommender` class.

---

## CẤU TRÚC MODULES

```
src/
├── api/
│   ├── main.py                    # FastAPI app
│   └── two_tower_routes.py        # API routes
├── services/
│   ├── two_tower_matching_service.py  # Matching service
│   └── embedding_service.py           # Embedding service với cache
├── embeddings/
│   ├── candidate_tower_encoder.py     # Candidate encoder
│   └── job_tower_encoder.py            # Job encoder
├── database/
│   ├── models.py                       # Database models
│   ├── two_tower_repository.py        # Repository cho Two-Tower
│   └── connection.py                   # Database connection
└── utils/
    ├── rule_matcher.py                 # Rule-based matching
    └── embedding_loader.py             # Model loader với fallback

two_tower/
├── model.py                            # TwoTowerModel architecture
└── inference.py                        # JobRecommender với FAISS
```

---

## GHI CHÚ

- Hệ thống hiện tại **không sử dụng precomputed embeddings từ database** trong matching service
- Embeddings được tính toán real-time từ text
- FAISS index chỉ có trong `JobRecommender` class, không được tích hợp vào main service
- Model được load từ checkpoint file, không từ database

