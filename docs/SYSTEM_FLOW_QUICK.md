# 🔄 TÓM TẮT NHANH LUỒNG HỆ THỐNG

## 🎯 TỔNG QUAN

**AI Job Recommendation Service** - Hệ thống match JD với Candidates bằng semantic embeddings theo pipeline 3 bước.

---

## 📊 LUỒNG CHÍNH

### 1️⃣ DATA → EMBEDDING → MATCHING

```
Raw CSV 
  ↓ 
[Lọc records có skills]
  ↓
[Generate 3 embeddings: Title, Skills, Requirements]
  ↓
[Lưu PostgreSQL]
  ↓
[FAISS Index cho tìm kiếm nhanh]
```

### 2️⃣ MATCHING PIPELINE (3 BƯỚC)

```
Candidate Request
  ↓
BƯỚC 1: Experience Match → 1000 jobs (40% weight)
  ↓
BƯỚC 2: Skills Match → 100 jobs (40% weight)
  ↓
BƯỚC 3: Title Match → 10 jobs (20% weight)
  ↓
Tính Combined Score → Top 10 Recommendations
```

### 3️⃣ API WORKFLOW

```
HTTP Request 
  → FastAPI Routes 
  → MultiFilterMatchingService 
  → PostgreSQL/FAISS Search 
  → JSON Response
```

---

## ⚙️ COMPONENTS CHÍNH

| Component | Chức năng |
|-----------|-----------|
| **Data Processing** | Validate, clean, filter data |
| **Embedding Generator** | Tạo 3 embeddings (Title, Skills, Requirements) |
| **MultiFilterMatchingService** | Pipeline 3 bước matching |
| **FAISS Manager** | Tìm kiếm vector nhanh |
| **Scheduler** | Tự động regenerate embeddings mỗi 12h |

---

## 🚀 QUY TRÌNH SETUP

```bash
# 1. Khởi tạo DB
python scripts/init_multi_field_tables.py

# 2. Lọc data có skills
python scripts/filter_data_with_skills.py

# 3. Tạo embeddings
python scripts/process_multi_field_embeddings.py \
    --jd-file data/filtered/jds_with_skills.csv \
    --candidate-file data/filtered/candidates_with_skills.csv

# 4. Chạy API
python main.py
```

---

## 🔑 ĐIỂM QUAN TRỌNG

1. **3 embeddings riêng biệt**: Title, Skills, Requirements
2. **Pipeline 3 bước**: 1000 → 100 → 10 jobs
3. **Weighted scoring**: Experience 40% + Skills 40% + Title 20%
4. **Scheduler tự động**: Regenerate mỗi 12 giờ
5. **FAISS**: Tăng tốc tìm kiếm

---

## 📡 API ENDPOINTS CHÍNH

- `POST /api/v1/multi-filter/match/candidate` - Tìm jobs cho candidate text
- `POST /api/v1/multi-filter/recommend/jobs` - Recommend jobs cho candidate ID
- `GET /api/v1/scheduler/status` - Kiểm tra scheduler

---

**Version**: Quick Summary v1.0


