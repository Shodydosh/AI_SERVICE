# BÁO CÁO RECOMMENDATION SYSTEM

## TỔNG QUAN

Hệ thống recommendation sử dụng **Two-Tower Architecture** để tính similarity giữa candidates và jobs, sau đó rank và trả về top K jobs.

---

## CÁCH TÍNH SIMILARITY / SCORE

### 1. TwoTowerMatchingService

**File:** `src/services/two_tower_matching_service.py`

**Phương thức:** `find_jobs_for_candidate(candidate_id: str, top_k: int = 10)`

**Cách tính:**

1. **Build candidate text:**
   ```python
   candidate_text = "Title: {title} | Skills: {skills} | Experience: {experience}"
   ```

2. **Encode candidate:**
   ```python
   candidate_emb = model.encode_candidates([candidate_text])[0]  # [output_dim]
   candidate_emb_np = candidate_emb.cpu().numpy()  # [output_dim]
   ```

3. **Build job texts cho tất cả jobs:**
   ```python
   job_text = "Title: {title} | Skills: {skills} | Requirements: {requirement}"
   ```

4. **Encode jobs theo batch:**
   ```python
   batch_size = 32
   for i in range(0, len(job_texts), batch_size):
       batch_texts = job_texts[i:i+batch_size]
       batch_embs = model.encode_jobs(batch_texts)  # [batch_size, output_dim]
       all_job_embs.append(batch_embs.cpu().numpy())
   
   job_embs = np.vstack(all_job_embs)  # [num_jobs, output_dim]
   ```

5. **Tính cosine similarity:**
   ```python
   # Embeddings đã được normalize (L2), nên dot product = cosine similarity
   similarities = np.dot(job_embs, candidate_emb_np)  # [num_jobs]
   ```

6. **Lấy top K:**
   ```python
   top_indices = np.argsort(similarities)[::-1][:top_k]
   # [::-1] để sort descending (từ cao xuống thấp)
   ```

7. **Format results:**
   ```python
   results = []
   for idx in top_indices:
       job = job_records[idx]
       score = float(similarities[idx])
       results.append({
           'job_id': job.job_id,
           'title': job.title,
           'company': job.company,
           'location': job.location,
           'score': score
       })
   ```

**Lưu ý:**
- Score là cosine similarity (range: -1 đến 1, nhưng thường là 0 đến 1 vì embeddings được normalize)
- Score càng cao = càng giống nhau

---

### 2. recommend_jobs_for_candidates Script

**File:** `scripts/recommend_jobs_for_candidates.py`

**Phương thức:** `recommend_jobs_for_candidates()`

**Cách tính (tương tự như trên):**

1. **Pre-compute job embeddings:**
   ```python
   # Tính một lần cho tất cả jobs
   job_embeddings = model.encode_jobs(job_texts)
   job_embeddings = job_embeddings.cpu().numpy()  # [num_jobs, output_dim]
   ```

2. **Với mỗi candidate:**
   ```python
   # Encode candidate
   candidate_emb = model.encode_candidates([candidate_text])[0]
   candidate_emb_np = candidate_emb.cpu().numpy()
   
   # Compute similarities với tất cả jobs
   similarities = np.dot(job_embeddings, candidate_emb_np)
   
   # Get top K jobs
   top_indices = np.argsort(similarities)[::-1][:top_k]
   ```

3. **Apply rule matching:**
   ```python
   # Sau khi có top K, apply rule matching để validate
   rule_result = rule_matcher.evaluate_match(
       candidate_title=candidate.title or "",
       candidate_skills=candidate_skills,
       job_title=job.title or "",
       job_requirements=job.requirement,
       job_description=getattr(job, 'description', None)
   )
   ```

**Output format:**
```python
{
    'job_id': job.job_id,
    'similarity': similarity,  # Cosine similarity score
    'rule_result': rule_result  # Rule matching result (OK/NG)
}
```

---

### 3. JobRecommender (FAISS)

**File:** `two_tower/inference.py`

**Class:** `JobRecommender`

**Cách tính:**

1. **Load precomputed job embeddings:**
   ```python
   job_embeddings, job_ids = load_embeddings(Path(job_embeddings_path))
   job_embeddings = normalize_embeddings(job_embeddings.astype(np.float32))
   ```

2. **Build FAISS index:**
   ```python
   if index_type == "HNSW":
       index = faiss.IndexHNSWFlat(dim, 32)
       index.hnsw.efConstruction = 200
       index.hnsw.efSearch = 128
   elif index_type == "Flat":
       index = faiss.IndexFlatIP(dim)  # Inner Product = Dot Product
   
   index.add(job_embeddings)
   ```

3. **Search:**
   ```python
   candidate_emb = model.encode_candidates([candidate_text])
   candidate_emb = normalize_embeddings(candidate_emb.cpu().numpy())
   
   distances, indices = index.search(candidate_emb, top_k)
   
   # distances là similarity scores (inner product)
   for dist, idx in zip(distances[0], indices[0]):
       score = float(dist)
       results.append({
           'job_id': job_ids[idx],
           'score': score
       })
   ```

**Lưu ý:** FAISS index không được sử dụng trong `TwoTowerMatchingService` hiện tại. Chỉ có trong `JobRecommender` class.

---

## CÁCH LẤY TOP K

### Trong TwoTowerMatchingService:

```python
# Tính similarities với tất cả jobs
similarities = np.dot(job_embs, candidate_emb_np)  # [num_jobs]

# Sort descending và lấy top K
top_indices = np.argsort(similarities)[::-1][:top_k]
# argsort trả về indices sắp xếp theo giá trị tăng dần
# [::-1] để đảo ngược thành giảm dần
# [:top_k] để lấy K phần tử đầu

# Format results
results = []
for idx in top_indices:
    score = float(similarities[idx])
    results.append({
        'job_id': job.job_id,
        'score': score,
        ...
    })
```

### Trong recommend_jobs_for_candidates:

```python
# Tương tự
similarities = np.dot(job_embeddings, candidate_emb_np)
top_indices = np.argsort(similarities)[::-1][:top_k]

# Process top K jobs
for rank, job_idx in enumerate(top_indices, 1):
    job = all_jobs[job_idx]
    similarity = float(similarities[job_idx])
    # Apply rule matching
    rule_result = rule_matcher.evaluate_match(...)
```

---

## MODEL ĐƯỢC SỬ DỤNG

**File:** `two_tower/model.py`

**Class:** `TwoTowerModel`

**Cấu hình:**
- `candidate_model_name`: "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
- `job_model_name`: "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
- `output_dim`: 768

**Checkpoint:**
- Default: `outputs_improved/best_model_improved.pt`

**Cách encode:**
- Candidate text → Candidate Tower → [768] embedding (normalized)
- Job text → Job Tower → [768] embedding (normalized)
- Similarity = dot product (vì đã normalize)

---

## RECOMMENDATION CHO 5 CANDIDATES

### Kết quả thực tế:

Đã chạy script `scripts/test_two_tower_precomputed.py` với parameters:
- `--max-candidates 5`
- `--top-k 5`
- `--output two_tower_recommendations_5x5_final.txt`

**Dữ liệu:**
- Loaded 44,963 candidates với embeddings từ database
- Loaded 14,634 jobs với embeddings từ database
- Sampling 5 candidates ngẫu nhiên (seed=42)

**Kết quả:**

### Candidate 10126 (Lưu Công Quang)

**Title:** Nhân Viên Kỹ Thuật Điện - Điện Lạnh

1. **Job 7852** – score: 0.8319
   - Title: Nhân Viên Kỹ Thuật Điện - Điện Lạnh (Lắp Đặt - Bảo Dưỡng - Sửa Chữa)
   - Company: Công ty Cổ phần đầu tư phát triển hạ tầng Thành Công
   - Location: Hà Nội
   - Rule 1 (Title): 0.8338 (Status: PASS)
   - Rule 2 (Skill): 1.0000 (Status: PASS)
   - Final Decision: OK

2. **Job 5950** – score: 0.8316
   - Title: Nhân Viên Kỹ Thuật Điện, Điện Tử
   - Company: CÔNG TY TNHH LIÊN DOANH DƯỢC PHẨM NUTRAMED
   - Location: Bà Rịa Vũng Tàu
   - Rule 1 (Title): 0.9062 (Status: PASS)
   - Rule 2 (Skill): 2.5000 (Status: PASS)
   - Final Decision: OK

3. **Job 3971** – score: 0.8281
   - Title: Nhân Viên Kỹ Thuật (Điện, Điện Tử)
   - Company: CÔNG TY CỔ PHẦN NUTRAMED
   - Location: Bà Rịa Vũng Tàu
   - Rule 1 (Title): 0.9062 (Status: PASS)
   - Rule 2 (Skill): 2.5000 (Status: PASS)
   - Final Decision: OK

4. **Job 6854** – score: 0.8251
   - Title: Nhân Viên Kỹ Thuật Điện Lạnh
   - Company: Công ty TNHH thương mại K&K Toàn Cầu
   - Location: Hà Nội
   - Rule 1 (Title): 1.0000 (Status: PASS)
   - Rule 2 (Skill): 2.0000 (Status: PASS)
   - Final Decision: OK

5. **Job 3599** – score: 0.8213
   - Title: Nhân Viên Kỹ Thuật Điện - Điện Điều Khiển
   - Company: CÔNG TY TNHH THIẾT BỊ CƠ KHÍ TOÀN CẦU
   - Location: Hà Nội
   - Rule 1 (Title): 0.8887 (Status: PASS)
   - Rule 2 (Skill): 2.5000 (Status: PASS)
   - Final Decision: OK

---

### Candidate 6590 (Võ Thị Kim Lan)

**Title:** Nhan Vien Phu Bep

1. **Job 14054** – score: 0.6815
   - Title: Cloudeats Tuyển dụng Bếp Chính làm việc tại Quận 2, Tân Phú, Tân Bình, Gò Vấp
   - Company: Tnhh Rbox
   - Location: Hồ Chí Minh
   - Rule 1 (Title): 0.4040 (Status: FAIL)
   - Rule 2 (Skill): 1.5000 (Status: PASS)
   - Final Decision: OK

2. **Job 2936** – score: 0.6570
   - Title: Giám Sát Bếp
   - Company: Khách sạn Mường Thanh Hà Nội Centre
   - Location: Hà Nội
   - Rule 1 (Title): 0.4138 (Status: FAIL)
   - Rule 2 (Skill): 2.0000 (Status: PASS)
   - Final Decision: OK

3. **Job 2338** – score: 0.6464
   - Title: Nhân Viên Bếp (Chuyên Món Á - Âu)
   - Company: CÔNG TY TNHH DỊCH VỤ GIẢI TRÍ TNT GROUP
   - Location: Hồ Chí Minh
   - Rule 1 (Title): 0.5909 (Status: FAIL)
   - Rule 2 (Skill): 1.5000 (Status: PASS)
   - Final Decision: OK

4. **Job 13062** – score: 0.6442
   - Title: Đầu Bếp
   - Company: Lien Minh Group
   - Location: Lâm Đồng
   - Rule 1 (Title): 0.5000 (Status: FAIL)
   - Rule 2 (Skill): 2.5000 (Status: PASS)
   - Final Decision: OK

5. **Job 13196** – score: 0.6442
   - Title: Đầu Bếp
   - Company: Vịt Quay Da Giòn Trần Ký
   - Location: Hồ Chí Minh
   - Rule 1 (Title): 0.5000 (Status: FAIL)
   - Rule 2 (Skill): 1.5000 (Status: PASS)
   - Final Decision: OK

---

### Candidate 44229 (Võ Thị Kim Ngân)

**Title:** Nhân Viên Kinh Doanh

1. **Job 1712** – score: 0.7673
   - Title: Nhân Viên Kinh Doanh Phần Mềm
   - Company: CÔNG TY TNHH MEDIASTEP SOFTWARE VIỆT NAM
   - Location: Hồ Chí Minh
   - Rule 1 (Title): 0.9339 (Status: PASS)
   - Rule 2 (Skill): 4.5000 (Status: PASS)
   - Final Decision: OK

2. **Job 13149** – score: 0.7657
   - Title: Nhân Viên Kinh Doanh - Chuyên Viên Tư Vấn
   - Company: CÔNG TY CỔ PHẦN EXIM PREMIUM
   - Location: Hồ Chí Minh
   - Rule 1 (Title): 0.8654 (Status: PASS)
   - Rule 2 (Skill): 3.5000 (Status: PASS)
   - Final Decision: OK

3. **Job 3769** – score: 0.7645
   - Title: Nhân Viên Tư Vấn Kinh Doanh - Lễ Tân
   - Company: Nhất Tín Logistics
   - Location: Hồ Chí Minh
   - Rule 1 (Title): 0.8250 (Status: PASS)
   - Rule 2 (Skill): 2.5000 (Status: PASS)
   - Final Decision: OK

4. **Job 8139** – score: 0.7528
   - Title: Thực Tập Sinh Kinh Doanh Phần Mềm
   - Company: Công ty Công nghệ AISOLUTIONS
   - Location: Hà Nội
   - Rule 1 (Title): 0.8307 (Status: PASS)
   - Rule 2 (Skill): 3.5000 (Status: PASS)
   - Final Decision: OK

5. **Job 884** – score: 0.7523
   - Title: Quản Lý Kinh Doanh
   - Company: CÔNG TY TNHH SẢN XUẤT - THƯƠNG MẠI - XUẤT NHẬP KHẨU MÓC ÁO DUY PHÁT
   - Location: Hồ Chí Minh
   - Rule 1 (Title): 0.8717 (Status: PASS)
   - Rule 2 (Skill): 2.0000 (Status: PASS)
   - Final Decision: OK

---

### Candidate 15638 (Hoàng Ngọc Huyền)

**Title:** Nhân Viên Cskh

1. **Job 7540** – score: 0.6741
   - Title: Nhân Viên QC Học Việc
   - Company: Công ty cổ phần trang
   - Location: Hồ Chí Minh
   - Rule 1 (Title): 0.6976 (Status: PASS)
   - Rule 2 (Skill): 0.0000 (Status: FAIL)
   - Final Decision: OK

2. **Job 4477** – score: 0.6700
   - Title: Nhân Viên Lập Trình .Net
   - Company: Công ty TNHH đầu tư công nghệ và dịch vụ Sconnect Việt Nam
   - Location: Hà Nội
   - Rule 1 (Title): 0.6123 (Status: PASS)
   - Rule 2 (Skill): 1.5000 (Status: PASS)
   - Final Decision: OK

3. **Job 11016** – score: 0.6676
   - Title: Thực Tập Sinh Nhân Sự - Làm việc ngay
   - Company: Công ty TNHH Thương Mại Nhơn Mỹ
   - Location: Hồ Chí Minh
   - Rule 1 (Title): 0.4995 (Status: FAIL)
   - Rule 2 (Skill): 1.0000 (Status: PASS)
   - Final Decision: OK

4. **Job 8939** – score: 0.6586
   - Title: Nhân Viên QA/Cs
   - Company: CÔNG TY TNHH THƯƠNG MẠI VÀ SẢN XUẤT BAO BÌ SÔNG LAM
   - Location: Hà Nội
   - Rule 1 (Title): 0.8276 (Status: PASS)
   - Rule 2 (Skill): 0.5000 (Status: FAIL)
   - Final Decision: OK

5. **Job 9263** – score: 0.6516
   - Title: Thực Tập Sinh Lập Trình Viên .Net
   - Company: Công ty TNHH Digital Innovation
   - Location: Hà Nội
   - Rule 1 (Title): 0.6190 (Status: PASS)
   - Rule 2 (Skill): 0.5000 (Status: FAIL)
   - Final Decision: OK

---

### Candidate 37173 (Hà Thế Vinh)

**Title:** Nhân Viên Kĩ Thuật

1. **Job 128** – score: 0.7680
   - Title: Nhân Viên Kĩ Thuật
   - Company: Công ty CP đầu tư thương mại và phát triển nông nghiệp ADI
   - Location: Hà Nội
   - Rule 1 (Title): 1.0000 (Status: PASS)
   - Rule 2 (Skill): 1.0000 (Status: PASS)
   - Final Decision: OK

2. **Job 7591** – score: 0.7527
   - Title: Quản Lỹ Kỹ Thuật
   - Company: CÔNG TY TNHH THƯƠNG MẠI VÀ DỊCH VỤ ANP VIỆT NAM
   - Location: Bắc Ninh
   - Rule 1 (Title): 0.8241 (Status: PASS)
   - Rule 2 (Skill): 2.0000 (Status: PASS)
   - Final Decision: OK

3. **Job 3808** – score: 0.7516
   - Title: Kỹ Thuật Vận Hành Máy Chấn
   - Company: Công ty TNHH MTV Luxer
   - Location: Ninh Bình
   - Rule 1 (Title): 0.7559 (Status: PASS)
   - Rule 2 (Skill): 2.0000 (Status: PASS)
   - Final Decision: OK

4. **Job 8629** – score: 0.7508
   - Title: Nhân Viên Kỹ Thuật Bảo Hành
   - Company: Công ty cổ phần phương tiện điện thông minh Selex
   - Location: Hà Nội
   - Rule 1 (Title): 0.8434 (Status: PASS)
   - Rule 2 (Skill): 3.0000 (Status: PASS)
   - Final Decision: OK

5. **Job 7564** – score: 0.7504
   - Title: Nhân Viên Bảo Trì - Kỹ Thuật
   - Company: Circle K Viet Nam Ltd. Co.
   - Location: Hà Nội
   - Rule 1 (Title): 0.8888 (Status: PASS)
   - Rule 2 (Skill): 1.5000 (Status: PASS)
   - Final Decision: OK

---

### Thống kê:

- **Total similarity computations:** 73,170
- **Average similarity:** 0.4544
- **Max similarity:** 0.8319
- **Min similarity:** 0.0230
- **Median similarity:** 0.4522

**Lưu ý:** Tất cả embeddings được load từ database (precomputed), không tính toán real-time.

---

## FORMAT OUTPUT (Nếu có dữ liệu)

Nếu có dữ liệu, format output sẽ như sau:

### Candidate {candidate_id}

1. Job {job_id} – score: {similarity_score}
   - Title: {job_title}
   - Company: {company}
   - Location: {location}
   - Rule Match: {OK/NG}
   - Rule 1 (Title): {score} (Status: {PASS/FAIL})
   - Rule 2 (Skill): {score} (Status: {PASS/FAIL})

2. Job {job_id} – score: {similarity_score}
   ...

3. Job {job_id} – score: {similarity_score}
   ...

4. Job {job_id} – score: {similarity_score}
   ...

5. Job {job_id} – score: {similarity_score}
   ...

---

## SCRIPT CHẠY RECOMMENDATION

**File:** `scripts/recommend_jobs_for_candidates.py`

**Cách chạy:**
```bash
python scripts/recommend_jobs_for_candidates.py \
    --max-candidates 5 \
    --top-k 10 \
    --output job_recommendations.txt \
    --model-path outputs_improved/best_model_improved.pt
```

**Yêu cầu:**
- Database phải có dữ liệu candidates và jobs
- Model checkpoint phải tồn tại
- Embeddings sẽ được tính toán real-time (không dùng precomputed từ database)

---

## API ENDPOINT

**Endpoint:** `POST /api/v2/search/jobs`

**Request:**
```json
{
    "candidate_id": "candidate_001",
    "top_k": 10
}
```

**Response:**
```json
{
    "total_matches": 10,
    "matches": [
        {
            "job_id": "job_001",
            "title": "Senior Software Engineer",
            "company": "Tech Corp",
            "location": "San Francisco, CA",
            "score": 0.8523
        },
        ...
    ]
}
```

**Flow:**
1. Gọi `TwoTowerMatchingService.find_jobs_for_candidate()`
2. Tính similarity với tất cả jobs
3. Lấy top K
4. Trả về JSON response

---

## GHI CHÚ

- Hệ thống **không sử dụng precomputed embeddings từ database** trong matching service
- Embeddings được tính toán **real-time** từ text
- Score là **cosine similarity** (range: -1 đến 1, thường là 0 đến 1)
- Top K được lấy bằng `np.argsort(similarities)[::-1][:top_k]`
- Rule matching được apply sau khi có top K để validate kết quả

---

## KẾT LUẬN

**Đã chạy thành công recommendation cho 5 candidates sử dụng precomputed embeddings từ database.**

**Script sử dụng:**
- `scripts/test_two_tower_precomputed.py`
- Load embeddings từ database (MultiFieldEmbeddingRepository)
- Sử dụng combined embedding (average của title, skills, experience/requirement embeddings)
- Tính cosine similarity và lấy top K

**Kết quả:**
- Processed 5 candidates
- Recommended 5 jobs per candidate
- Tất cả 25 recommendations đều có Final Decision: OK
- Average similarity: 0.4544
- Max similarity: 0.8319

**File output:** `two_tower_recommendations_5x5_final.txt`

