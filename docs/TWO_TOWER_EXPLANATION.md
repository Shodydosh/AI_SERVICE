# Two-Tower Model - Giải Thích Chi Tiết

## 📋 Tổng Quan

**Two-Tower Model** là kiến trúc neural network được thiết kế đặc biệt cho bài toán **recommendation và matching**. Trong hệ thống này, nó được sử dụng để match **Candidate (ứng viên)** với **Job (công việc)**.

---

## 🏗️ Kiến Trúc Two-Tower

### Cấu Trúc Tổng Quan

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT LAYER                          │
│  Candidate: [Title(768), Skills(768), Experience(768)]  │
│  Job:      [Title(768), Skills(768), Requirement(768)]  │
└─────────────────────────────────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │   CANDIDATE TOWER              │
        │   Input: 768×3 = 2304          │
        │   Hidden: [512, 256]           │
        │   Output: 256 (normalized)     │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │   JOB TOWER                    │
        │   Input: 768×3 = 2304          │
        │   Hidden: [512, 256]           │
        │   Output: 256 (normalized)     │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │   SIMILARITY COMPUTATION       │
        │   Dot Product (Cosine Similarity)│
        │   Score = candidate_repr · job_repr│
        └───────────────────────────────┘
```

---

## 🔍 Chi Tiết Kiến Trúc

### 1. Candidate Tower

**Input:**
- `title_embedding`: [768 dim] - Vị trí mong muốn của ứng viên
- `skills_embedding`: [768 dim] - Kỹ năng của ứng viên  
- `experience_embedding`: [768 dim] - Kinh nghiệm làm việc

**Quy Trình:**

```python
# Bước 1: Concatenate 3 embeddings
combined = concat([title_emb, skills_emb, experience_emb])
# Shape: [batch_size, 768×3] = [batch_size, 2304]

# Bước 2: Pass qua neural network
# Layer 1: 2304 → 512 (Linear + BatchNorm + ReLU + Dropout)
# Layer 2: 512 → 256 (Linear + BatchNorm + ReLU + Dropout)
# Output: 256 (Linear)

# Bước 3: L2 Normalize
candidate_repr = normalize(output, p=2, dim=1)
# Shape: [batch_size, 256], ||candidate_repr|| = 1
```

**Code:**

```12:101:src/models/two_tower_model.py
class CandidateTower(nn.Module):
    """
    Candidate Tower: Encodes candidate features (title, skills, experience)
    into a unified representation.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dims: List[int] = [512, 256],
        output_dim: int = 256,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        # Input: 768×3 = 2304
        # Hidden layers: [512, 256]
        # Output: 256
    
    def forward(
        self,
        title_emb: torch.Tensor,
        skills_emb: torch.Tensor,
        experience_emb: torch.Tensor
    ) -> torch.Tensor:
        # Concatenate three embeddings
        combined = torch.cat([title_emb, skills_emb, experience_emb], dim=1)
        
        # Pass through network
        output = self.network(combined)
        
        # L2 normalize output
        output = F.normalize(output, p=2, dim=1)
        
        return output
```

### 2. Job Tower

**Input:**
- `title_embedding`: [768 dim] - Tiêu đề công việc
- `skills_embedding`: [768 dim] - Kỹ năng yêu cầu
- `requirement_embedding`: [768 dim] - Yêu cầu công việc

**Quy Trình:** Tương tự Candidate Tower

```104:193:src/models/two_tower_model.py
class JobTower(nn.Module):
    """
    Job Tower: Encodes job features (title, skills, requirements)
    into a unified representation.
    """
    
    def forward(
        self,
        title_emb: torch.Tensor,
        skills_emb: torch.Tensor,
        requirement_emb: torch.Tensor
    ) -> torch.Tensor:
        # Concatenate three embeddings
        combined = torch.cat([title_emb, skills_emb, requirement_emb], dim=1)
        
        # Pass through network
        output = self.network(combined)
        
        # L2 normalize output
        output = F.normalize(output, p=2, dim=1)
        
        return output
```

### 3. Similarity Computation

**Công Thức:**

```279:296:src/models/two_tower_model.py
def compute_similarity(
    self,
    candidate_repr: torch.Tensor,
    job_repr: torch.Tensor
) -> torch.Tensor:
    """
    Compute similarity score between candidate and job representations.
    
    Args:
        candidate_repr: Candidate representation [batch_size, output_dim]
        job_repr: Job representation [batch_size, output_dim]
    
    Returns:
        Similarity scores [batch_size]
    """
    # Dot product (since both are L2 normalized, this is cosine similarity)
    similarity = torch.sum(candidate_repr * job_repr, dim=1)
    return similarity
```

**Giải Thích:**
- Vì cả `candidate_repr` và `job_repr` đều được **L2 normalize** (||vector|| = 1)
- Dot product = **Cosine Similarity**
- Range: [-1, 1] (thực tế thường [0, 1] vì embeddings dương)

---

## 🎯 Tại Sao Dùng Two-Tower?

### Ưu Điểm:

1. **Học Representation Tối Ưu:**
   - Model tự động học cách kết hợp 3 embeddings (title, skills, experience)
   - Không cần manual weights như baseline

2. **Scalable:**
   - Có thể pre-compute job embeddings một lần
   - Chỉ cần encode candidate mới khi inference
   - Tốc độ nhanh với FAISS index

3. **Chất Lượng Cao:**
   - Học từ ground truth data
   - Tối ưu cho task cụ thể (job matching)
   - Tốt hơn baseline weighted average

4. **Flexible:**
   - Có thể fine-tune với domain-specific data
   - Dễ dàng thêm features mới

---

## 📊 Training Process

### 1. Ground Truth Data

**Format:**
```json
{
  "candidate_id": "123",
  "job_id": "456",
  "label": 1.0  // 1.0 = positive match, 0.0 = negative
}
```

**Tạo Ground Truth:**
- Positive pairs: Candidate-Job có similarity cao (title + skills + experience)
- Negative pairs: Candidate-Job không match

### 2. Loss Function

**BCE With Logits Loss:**

```206:207:src/models/training_pipeline.py
# Compute loss (using BCE with logits)
loss = self.criterion(similarity, labels)
```

**Giải Thích:**
- `similarity`: Score từ model (dot product)
- `labels`: 1.0 (match) hoặc 0.0 (no match)
- Model học để maximize similarity cho positive pairs
- Model học để minimize similarity cho negative pairs

### 3. Training Loop

```164:221:src/models/training_pipeline.py
def train_epoch(
    self,
    dataloader: DataLoader,
    epoch: int
) -> Dict[str, float]:
    self.model.train()
    
    for batch in dataloader:
        # Forward pass
        candidate_repr, job_repr = self.model(
            candidate_title, candidate_skills, candidate_experience,
            job_title, job_skills, job_requirement
        )
        
        # Compute similarity
        similarity = self.model.compute_similarity(candidate_repr, job_repr)
        
        # Compute loss
        loss = self.criterion(similarity, labels)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
```

---

## 📈 Evaluation Metrics

### Metrics Được Sử Dụng:

1. **AUC-ROC:**
   - Đo khả năng phân biệt positive vs negative pairs
   - Range: [0, 1], càng cao càng tốt

2. **NDCG@10:**
   - Normalized Discounted Cumulative Gain
   - Đo chất lượng ranking top 10 recommendations
   - Range: [0, 1], càng cao càng tốt

3. **Precision@K:**
   - Tỷ lệ recommendations đúng trong top K

4. **Recall@K:**
   - Tỷ lệ positive pairs được tìm thấy trong top K

### Kết Quả Mẫu:

Từ `two_tower/RUN_RESULTS.md`:
- **Train Loss:** Giảm từ 0.4520 → 0.4866 (có thể overfitting)
- **Val Recall@10:** 1.0000 (100% positive pairs trong top 10)
- **Model saved:** `outputs/best_model.pt`

---

## 🚀 Inference Process

### 1. Encode Candidate

```python
# Input: Candidate với 3 embeddings
candidate_repr = model.encode_candidate(
    title_emb=title_embedding,      # [768]
    skills_emb=skills_embedding,    # [768]
    experience_emb=experience_emb   # [768]
)
# Output: [256] - normalized vector
```

### 2. Encode Job (Pre-computed)

```python
# Pre-compute tất cả job embeddings
job_repr = model.encode_job(
    title_emb=job_title_emb,        # [768]
    skills_emb=job_skills_emb,      # [768]
    requirement_emb=job_req_emb     # [768]
)
# Output: [256] - normalized vector
```

### 3. Compute Similarity & Rank

```python
# Tính similarity với tất cả jobs
similarities = []
for job_repr in all_job_reprs:
    score = model.compute_similarity(candidate_repr, job_repr)
    similarities.append((job_id, score))

# Sort và lấy top K
top_jobs = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]
```

### 4. Với FAISS Index (Nhanh Hơn)

```python
# Build FAISS index với tất cả job_reprs
index = faiss.IndexFlatIP(256)  # Inner Product = Dot Product
index.add(job_reprs_matrix)     # [n_jobs, 256]

# Search
scores, indices = index.search(candidate_repr, k=10)
# Trả về top 10 jobs nhanh chóng
```

---

## 🔄 So Sánh với Baseline

### Baseline (Weighted Average):

```python
# Manual weights
similarity = (
    title_sim * 0.2 +
    skills_sim * 0.4 +
    exp_sim * 0.4
)
```

**Vấn Đề:**
- Weights cố định, không học được
- Không tối ưu cho task cụ thể
- Không capture interactions giữa fields

### Two-Tower:

```python
# Learned representation
candidate_repr = CandidateTower(title, skills, experience)
job_repr = JobTower(title, skills, requirement)
similarity = dot_product(candidate_repr, job_repr)
```

**Ưu Điểm:**
- Tự động học cách kết hợp fields
- Tối ưu cho task matching
- Capture complex interactions

---

## 📝 Tóm Tắt

### Input:
- **Candidate:** 3 embeddings (title, skills, experience) × 768 dim
- **Job:** 3 embeddings (title, skills, requirement) × 768 dim

### Process:
1. **Candidate Tower:** 2304 → 512 → 256 (normalized)
2. **Job Tower:** 2304 → 512 → 256 (normalized)
3. **Similarity:** Dot product = Cosine similarity

### Output:
- **Similarity Score:** [0, 1] (càng cao càng match)
- **Top K Jobs:** Ranked by similarity

### Training:
- **Loss:** BCE With Logits
- **Optimizer:** Adam
- **Metrics:** AUC-ROC, NDCG@10

### Inference:
- **Pre-compute:** Job embeddings (một lần)
- **Real-time:** Encode candidate mới
- **Search:** FAISS index cho tốc độ cao

---

## 🎓 Kết Luận

Two-Tower Model là phương pháp **state-of-the-art** cho recommendation systems:

✅ **Học được** representation tối ưu từ data  
✅ **Scalable** với FAISS index  
✅ **Chất lượng cao** hơn baseline  
✅ **Flexible** dễ fine-tune và mở rộng  

Đây là phương pháp được khuyến nghị cho production systems!














