# 🔄 Luồng Hoạt Động Của Hệ Thống Two-Tower

## 📋 Tổng Quan

Hệ thống Two-Tower sử dụng kiến trúc neural network với 2 tower độc lập để match **Candidate (ứng viên)** với **Job (công việc)**. Mỗi tower encode thông tin thành embedding vector, sau đó tính similarity để tìm matches tốt nhất.

---

## 🏗️ Kiến Trúc Tổng Quan

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT DATA                                    │
│  Candidate: [Title, Skills, Experience]                         │
│  Job:       [Title, Skills, Requirement]                         │
└─────────────────────────────────────────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │   BASE ENCODER (SimCSE)        │
        │   Encode từng field riêng biệt │
        │   Output: 768-dim embeddings   │
        └───────────────────────────────┘
                        ↓
┌───────────────────────────────┐  ┌───────────────────────────────┐
│   CANDIDATE TOWER              │  │   JOB TOWER                    │
│   Input: [768, 768, 768]       │  │   Input: [768, 768, 768]       │
│   → Concatenate: [2304]        │  │   → Concatenate: [2304]        │
│   → Dense: [512]               │  │   → Dense: [512]               │
│   → Dense: [256]               │  │   → Dense: [256]               │
│   → L2 Normalize               │  │   → L2 Normalize               │
│   Output: [256] (normalized)    │  │   Output: [256] (normalized)   │
└───────────────────────────────┘  └───────────────────────────────┘
                        ↓                    ↓
                    ┌───────────────────────────┐
                    │   SIMILARITY COMPUTATION   │
                    │   Dot Product = Cosine     │
                    │   Score = cand · job       │
                    └───────────────────────────┘
                                ↓
                    ┌───────────────────────────┐
                    │   RANKING & TOP-K          │
                    │   Sort by score            │
                    │   Return top K matches     │
                    └───────────────────────────┘
```

---

## 🔄 Luồng Hoạt Động Chi Tiết

### 1️⃣ **LUỒNG INDEX JOB/CANDIDATE** (Lưu dữ liệu vào hệ thống)

#### A. Index Job

```
POST /api/v2/index/job
  ↓
[API Layer] two_tower_routes.py::index_job()
  ↓
[Encoder] JobTowerEncoder.encode_job()
  ├─ Encode title → title_embedding [768]
  ├─ Encode skills → skills_embedding [768]
  └─ Encode requirement → requirement_embedding [768]
  ↓
[Repository] TwoTowerRepository.create_job()
  ↓
[Database] PostgreSQL - JobDescriptionTwoTower table
  ├─ Lưu job_id, title, skills, requirement, company, location
  └─ Lưu 3 embeddings: title_embedding, skills_embedding, requirement_embedding
  ↓
Response: {status: "success", job_id: "...", message: "Job indexed successfully"}
```

**Code Flow:**
```python
# 1. API nhận request
@router.post("/index/job")
async def index_job(request: IndexJobRequest, db: Session):
    # 2. Encode job thành 3 embeddings
    job_encoder = JobTowerEncoder()
    embeddings = job_encoder.encode_job(
        title=request.title,
        skills=request.skills,
        requirements=request.requirement
    )
    
    # 3. Lưu vào database
    repository = TwoTowerRepository(db)
    repository.create_job(
        job_id=request.job_id,
        title_embedding=embeddings['title_embedding'],
        skills_embedding=embeddings['skills_embedding'],
        requirement_embedding=embeddings['requirement_embedding'],
        ...
    )
```

#### B. Index Candidate

```
POST /api/v2/index/candidate
  ↓
[API Layer] two_tower_routes.py::index_candidate()
  ↓
[Encoder] CandidateTowerEncoder.encode_candidate()
  ├─ Encode title → title_embedding [768]
  ├─ Encode skills → skills_embedding [768]
  └─ Encode experience → experience_embedding [768]
  ↓
[Repository] TwoTowerRepository.create_candidate()
  ↓
[Database] PostgreSQL - CandidateTwoTower table
  ├─ Lưu candidate_id, name, email, title, skills, experience
  └─ Lưu 3 embeddings: title_embedding, skills_embedding, experience_embedding
  ↓
Response: {status: "success", candidate_id: "...", message: "Candidate indexed successfully"}
```

---

### 2️⃣ **LUỒNG SEARCH JOBS FOR CANDIDATE** (Tìm jobs cho ứng viên)

```
POST /api/v2/search/jobs
  {
    "candidate_id": "CAND001",
    "top_k": 10
  }
  ↓
[API Layer] two_tower_routes.py::search_jobs()
  ↓
[Service] TwoTowerMatchingService.find_jobs_for_candidate()
  │
  ├─ Bước 1: Lấy candidate từ database
  │   candidate = repository.get_candidate(candidate_id)
  │
  ├─ Bước 2: Build candidate text
  │   candidate_text = "Title: {title} | Skills: {skills} | Experience: {experience}"
  │
  ├─ Bước 3: Encode candidate
  │   candidate_emb = model.encode_candidates([candidate_text])[0]
  │   Shape: [768] hoặc [256] (tùy model)
  │
  ├─ Bước 4: Lấy tất cả jobs từ database
  │   all_jobs = repository.get_all_jobs()
  │
  ├─ Bước 5: Build job texts cho tất cả jobs
  │   for job in all_jobs:
  │       job_text = "Title: {title} | Skills: {skills} | Requirements: {requirement}"
  │       job_texts.append(job_text)
  │
  ├─ Bước 6: Encode jobs theo batch (batch_size=32)
  │   for i in range(0, len(job_texts), 32):
  │       batch = job_texts[i:i+32]
  │       batch_embs = model.encode_jobs(batch)
  │       all_job_embs.append(batch_embs)
  │   job_embs = np.vstack(all_job_embs)  # [num_jobs, embedding_dim]
  │
  ├─ Bước 7: Tính cosine similarity
  │   similarities = np.dot(job_embs, candidate_emb)  # [num_jobs]
  │
  ├─ Bước 8: Sort và lấy top K
  │   top_indices = np.argsort(similarities)[::-1][:top_k]
  │
  └─ Bước 9: Format và trả về results
      results = []
      for idx in top_indices:
          results.append({
              'job_id': job.job_id,
              'title': job.title,
              'company': job.company,
              'score': similarities[idx]
          })
  ↓
Response: {
  "total_matches": 10,
  "matches": [
    {"job_id": "...", "title": "...", "score": 0.95},
    ...
  ]
}
```

**Code Flow Chi Tiết:**

```python
def find_jobs_for_candidate(self, candidate_id: str, top_k: int = 10):
    # 1. Lấy candidate
    candidate = self.repository.get_candidate(candidate_id)
    
    # 2. Build text
    candidate_text = f"Title: {candidate.title} | Skills: {candidate.skills} | Experience: {candidate.experience}"
    
    # 3. Encode candidate (real-time)
    with torch.no_grad():
        candidate_emb = self.model.encode_candidates([candidate_text])[0]
        candidate_emb_np = candidate_emb.cpu().numpy()  # [768] hoặc [256]
    
    # 4. Lấy tất cả jobs
    all_jobs = self.repository.get_all_jobs()
    
    # 5. Encode jobs theo batch
    job_texts = [self._build_job_text(job) for job in all_jobs]
    batch_size = 32
    all_job_embs = []
    with torch.no_grad():
        for i in range(0, len(job_texts), batch_size):
            batch = job_texts[i:i+batch_size]
            batch_embs = self.model.encode_jobs(batch)
            all_job_embs.append(batch_embs.cpu().numpy())
    job_embs = np.vstack(all_job_embs)  # [num_jobs, embedding_dim]
    
    # 6. Tính similarity (cosine = dot product vì đã normalize)
    similarities = np.dot(job_embs, candidate_emb_np)  # [num_jobs]
    
    # 7. Top-K
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # 8. Format results
    return [{
        'job_id': all_jobs[idx].job_id,
        'title': all_jobs[idx].title,
        'score': float(similarities[idx])
    } for idx in top_indices]
```

---

### 3️⃣ **LUỒNG SEARCH CANDIDATES FOR JOB** (Tìm ứng viên cho job)

Luồng này tương tự luồng search jobs, nhưng đảo ngược:

```
POST /api/v2/search/candidates
  {
    "job_id": "JD001",
    "top_k": 10
  }
  ↓
[Service] TwoTowerMatchingService.find_candidates_for_job()
  │
  ├─ Bước 1: Lấy job từ database
  ├─ Bước 2: Build job text
  ├─ Bước 3: Encode job
  ├─ Bước 4: Lấy tất cả candidates
  ├─ Bước 5: Encode candidates theo batch
  ├─ Bước 6: Tính similarity
  ├─ Bước 7: Sort và lấy top K
  └─ Bước 8: Format results
  ↓
Response: {
  "total_matches": 10,
  "matches": [
    {"candidate_id": "...", "name": "...", "score": 0.92},
    ...
  ]
}
```

---

### 4️⃣ **LUỒNG VỚI PRECOMPUTED EMBEDDINGS** (Sử dụng embeddings đã lưu sẵn)

Khi embeddings đã được lưu sẵn trong database, luồng sẽ nhanh hơn:

```
Script: test_two_tower_precomputed.py
  ↓
1. Load embeddings từ database
   ├─ candidate.title_embedding [768]
   ├─ candidate.skills_embedding [768]
   └─ candidate.experience_embedding [768]
   
   ├─ job.title_embedding [768]
   ├─ job.skills_embedding [768]
   └─ job.requirement_embedding [768]
  ↓
2. Combine embeddings (simple average)
   candidate_combined = mean([title_emb, skills_emb, experience_emb])
   job_combined = mean([title_emb, skills_emb, requirement_emb])
  ↓
3. Pre-compute tất cả job embeddings
   job_embeddings = [combined_emb for job in all_jobs]  # [num_jobs, 768]
  ↓
4. Với mỗi candidate:
   ├─ Load candidate embeddings từ DB
   ├─ Combine: candidate_combined = mean([...])
   ├─ Tính similarity: similarities = np.dot(job_embeddings, candidate_combined)
   ├─ Top-K: top_indices = np.argsort(similarities)[::-1][:top_k]
   └─ Apply rule matching (optional)
  ↓
5. Output results với scores và rule matching results
```

**Ưu điểm:**
- ✅ Không cần encode lại (tiết kiệm thời gian)
- ✅ Có thể cache embeddings trong memory
- ✅ Phù hợp cho batch processing

**Nhược điểm:**
- ❌ Không sử dụng Two-Tower model (chỉ dùng simple average)
- ❌ Chất lượng có thể thấp hơn model đã train

---

## 🔍 Chi Tiết Các Component

### 1. **JobTowerEncoder / CandidateTowerEncoder**

**Nhiệm vụ:** Encode text thành embedding vectors

```python
class JobTowerEncoder:
    def encode_job(self, title, skills, requirements):
        # Sử dụng SimCSE model
        title_emb = self.model.encode(title)      # [768]
        skills_emb = self.model.encode(skills)     # [768]
        req_emb = self.model.encode(requirements) # [768]
        
        return {
            'title_embedding': title_emb,
            'skills_embedding': skills_emb,
            'requirement_embedding': req_emb
        }
```

### 2. **TwoTowerModel**

**Nhiệm vụ:** Kết hợp 3 embeddings thành representation tối ưu

```python
class TwoTowerModel:
    def encode_candidate(self, title_emb, skills_emb, experience_emb):
        # Concatenate
        combined = concat([title_emb, skills_emb, experience_emb])  # [2304]
        
        # Pass qua neural network
        x = self.candidate_tower.fc1(combined)  # [512]
        x = self.candidate_tower.fc2(x)          # [256]
        
        # L2 normalize
        output = F.normalize(x, p=2, dim=1)      # [256]
        return output
    
    def encode_job(self, title_emb, skills_emb, requirement_emb):
        # Tương tự candidate tower
        ...
    
    def compute_similarity(self, candidate_repr, job_repr):
        # Dot product = cosine similarity (vì đã normalize)
        return torch.sum(candidate_repr * job_repr, dim=1)
```

### 3. **TwoTowerMatchingService**

**Nhiệm vụ:** Orchestrate toàn bộ matching process

- Load model
- Encode candidate/job
- Tính similarity
- Ranking và top-K
- Format results

### 4. **TwoTowerRepository**

**Nhiệm vụ:** Database operations

- `create_job()`: Lưu job và embeddings
- `create_candidate()`: Lưu candidate và embeddings
- `get_candidate()`: Lấy candidate từ DB
- `get_all_jobs()`: Lấy tất cả jobs
- `get_all_candidates()`: Lấy tất cả candidates

---

## 📊 So Sánh 2 Phương Pháp

### Phương Pháp 1: Real-time Encoding (TwoTowerMatchingService)

```
Request → Encode candidate → Encode all jobs → Similarity → Top-K
```

**Ưu điểm:**
- ✅ Sử dụng Two-Tower model đã train (chất lượng cao)
- ✅ Tự động học cách kết hợp fields
- ✅ Tối ưu cho matching task

**Nhược điểm:**
- ❌ Chậm hơn (phải encode real-time)
- ❌ Tốn tài nguyên (CPU/GPU)

### Phương Pháp 2: Precomputed Embeddings (test_two_tower_precomputed.py)

```
Load embeddings từ DB → Combine (average) → Similarity → Top-K
```

**Ưu điểm:**
- ✅ Nhanh (không cần encode)
- ✅ Tiết kiệm tài nguyên
- ✅ Phù hợp batch processing

**Nhược điểm:**
- ❌ Không dùng Two-Tower model (chỉ average)
- ❌ Chất lượng có thể thấp hơn

---

## 🚀 Tối Ưu Hóa Performance

### 1. **Batch Processing**
- Encode jobs theo batch (batch_size=32)
- Giảm số lần forward pass

### 2. **Pre-compute Job Embeddings**
- Encode tất cả jobs một lần
- Cache trong memory hoặc database
- Chỉ encode candidate mới khi search

### 3. **FAISS Index** (Future)
- Build FAISS index cho job embeddings
- Approximate Nearest Neighbor search
- Tăng tốc độ search lên 10-100x

### 4. **Caching**
- Cache candidate embeddings nếu search nhiều lần
- Cache job embeddings trong memory

---

## 📝 Tóm Tắt Luồng

### **Index Flow:**
```
Input → Encoder → 3 Embeddings → Database
```

### **Search Flow:**
```
Request → Load Data → Encode → Similarity → Ranking → Top-K → Response
```

### **Precomputed Flow:**
```
Load Embeddings → Combine → Similarity → Ranking → Top-K → Response
```

---

## 🎯 Kết Luận

Hệ thống Two-Tower hoạt động theo 3 luồng chính:

1. **Index**: Lưu job/candidate và embeddings vào database
2. **Search Real-time**: Encode real-time và tính similarity
3. **Search Precomputed**: Sử dụng embeddings đã lưu sẵn

Mỗi luồng có ưu/nhược điểm riêng, tùy vào use case mà chọn phương pháp phù hợp:
- **Real-time**: Khi cần chất lượng cao, có GPU
- **Precomputed**: Khi cần tốc độ, batch processing











