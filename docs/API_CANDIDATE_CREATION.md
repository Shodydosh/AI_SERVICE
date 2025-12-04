# 📝 API Documentation: Candidate Creation

## Tổng Quan

Hệ thống đã được cải tiến để tự động xử lý CV mới:
- ✅ Tự động generate embeddings khi thêm CV
- ✅ Background task update FAISS indices
- ✅ Background task pre-compute recommendations
- ✅ Response nhanh, xử lý async

---

## Endpoints

### 1. Tạo CV Mới (Single)

**Endpoint:** `POST /api/v1/candidates`

**Request Body:**
```json
{
  "candidate_id": "CANDIDATE_001",
  "title": "Nhân Viên Kế Toán",
  "skills": "Excel, Kế toán, Báo cáo tài chính, SAP",
  "experience": "5 năm kinh nghiệm làm kế toán tại công ty lớn, xử lý báo cáo tài chính, quản lý sổ sách",
  "name": "Nguyễn Văn A",
  "email": "nguyenvana@example.com"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "candidate_id": "CANDIDATE_001",
    "status": "created",
    "embeddings_generated": true,
    "recommendations_pending": true
  },
  "message": "Candidate created successfully. Embeddings generated. Recommendations are being processed in background.",
  "timestamp": "2024-01-15T10:30:00"
}
```

**Flow:**
1. Validate input (experience required)
2. Generate 3 embeddings: Title, Skills, Experience
3. Lưu vào database (`candidate_multi_embeddings`)
4. Return response ngay
5. Background task:
   - Update FAISS indices
   - Pre-compute top 10 recommendations
   - Lưu vào `processed_candidate_recommendations`

---

### 2. Tạo CV Batch (Multiple)

**Endpoint:** `POST /api/v1/candidates/batch`

**Request Body:**
```json
{
  "candidates": [
    {
      "candidate_id": "CANDIDATE_001",
      "title": "Nhân Viên Kế Toán",
      "skills": "Excel, Kế toán",
      "experience": "5 năm kinh nghiệm",
      "name": "Nguyễn Văn A",
      "email": "a@example.com"
    },
    {
      "candidate_id": "CANDIDATE_002",
      "title": "Lập Trình Viên",
      "skills": "Python, Java, SQL",
      "experience": "3 năm kinh nghiệm lập trình",
      "name": "Trần Thị B",
      "email": "b@example.com"
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "total_requested": 2,
    "created": 2,
    "failed": 0,
    "candidate_ids": ["CANDIDATE_001", "CANDIDATE_002"],
    "failed_candidates": []
  },
  "message": "Batch processing completed: 2 created, 0 failed. Recommendations are being processed in background.",
  "timestamp": "2024-01-15T10:30:00"
}
```

**Giới hạn:** Tối đa 100 candidates mỗi batch

---

## Schema

### CandidateCreateRequest

```python
{
  "candidate_id": str,          # Required: Unique ID
  "title": str,                 # Optional: Desired job title
  "skills": str,                # Optional: Skills
  "experience": str,             # Required: Work experience
  "name": str,                  # Optional: Name
  "email": str                  # Optional: Email
}
```

### Validation Rules

- `candidate_id`: Required, unique
- `experience`: Required, không được rỗng
- `title`, `skills`: Optional nhưng nên có để matching tốt hơn

---

## Background Processing

Sau khi tạo candidate, hệ thống tự động chạy background tasks:

### 1. Update FAISS Indices
- Load existing indices (nếu có)
- Rebuild với candidate mới
- Save indices

### 2. Pre-compute Recommendations
- Tìm top 10 jobs phù hợp
- Lưu vào `processed_candidate_recommendations`
- Có thể query nhanh sau này

**Thời gian xử lý:** 
- Single candidate: ~5-10 giây
- Batch 10 candidates: ~30-60 giây

---

## Error Handling

### 400 Bad Request
```json
{
  "detail": "Experience is required"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Error creating candidate: [error message]"
}
```

---

## Examples

### cURL - Single Candidate

```bash
curl -X POST "http://localhost:8000/api/v1/candidates" \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_id": "CANDIDATE_001",
    "title": "Nhân Viên Kế Toán",
    "skills": "Excel, Kế toán, Báo cáo tài chính",
    "experience": "5 năm kinh nghiệm làm kế toán",
    "name": "Nguyễn Văn A",
    "email": "a@example.com"
  }'
```

### Python - Batch Create

```python
import requests

url = "http://localhost:8000/api/v1/candidates/batch"
data = {
    "candidates": [
        {
            "candidate_id": "CANDIDATE_001",
            "title": "Nhân Viên Kế Toán",
            "skills": "Excel, Kế toán",
            "experience": "5 năm kinh nghiệm",
            "name": "Nguyễn Văn A"
        },
        {
            "candidate_id": "CANDIDATE_002",
            "title": "Lập Trình Viên",
            "skills": "Python, Java",
            "experience": "3 năm kinh nghiệm",
            "name": "Trần Thị B"
        }
    ]
}

response = requests.post(url, json=data)
print(response.json())
```

---

## Notes

1. **Response Time:** API trả về ngay sau khi lưu DB, không đợi background tasks
2. **Recommendations:** Có thể query sau 5-10 giây (sau khi background task hoàn thành)
3. **FAISS Update:** Indices được update tự động, không cần rebuild thủ công
4. **Idempotency:** Nếu candidate_id đã tồn tại, sẽ update thay vì tạo mới

---

## Integration với Existing Endpoints

Sau khi tạo candidate, có thể sử dụng:

- `GET /api/v1/candidates/{candidate_id}` - Xem thông tin
- `POST /api/v1/multi-filter/recommend/jobs` - Lấy recommendations
- `POST /api/v1/multi-filter/match/candidate` - Match jobs

---

**Version:** 1.0.0  
**Last Updated:** 2024-01-15

