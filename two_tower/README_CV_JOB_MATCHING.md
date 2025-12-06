# Hệ Thống Two-Tower Matching CV-Job

## 📋 Tổng Quan

Hệ thống matching CV-Job sử dụng kiến trúc **Two-Tower** kết hợp với **Rule-based Filtering** để đánh giá mức độ phù hợp giữa CV và Job.

## 🏗️ Kiến Trúc

### Two-Tower Architecture
- **Tower A (CV Tower)**: Embedding của CV (title, skills, experience, description)
- **Tower B (Job Tower)**: Embedding của Job (title, requirements, description)
- **Similarity**: Cosine similarity giữa 2 embeddings

### Rule-Based Filtering

#### Rule 1: Title Match
- Chuẩn hóa title (lowercase, bỏ dấu, bỏ stopwords)
- Tính overlap giữa CV.title và Job.title
- **PASS nếu >= 75%** ký tự hoặc token trùng nhau

#### Rule 2: Skill-Requirement Match
- Extract skills từ CV
- Kiểm tra semantic matching với job requirements/description
- **PASS nếu >= 1 skill** quan trọng khớp (semantic match)

#### Final Decision
- **OK**: Nếu Rule 1 HOẶC Rule 2 PASS
- **NG**: Nếu cả 2 rules đều FAIL

## 🚀 Cách Sử Dụng

### 1. Chạy với Example Data

```bash
python -m two_tower.run_cv_job_matching
```

### 2. Chạy với Input File

Tạo file JSON với format:

```json
{
  "cv": {
    "title": "Senior Python Developer",
    "skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"],
    "experience": "5 years experience...",
    "description": "Experienced Python developer..."
  },
  "jobs": [
    {
      "job_id": "job_1",
      "title": "Senior Python Developer",
      "requirements": "Python, FastAPI, PostgreSQL, 5+ years",
      "description": "We are looking for..."
    },
    ...
  ]
}
```

Sau đó chạy:

```bash
python -m two_tower.run_cv_job_matching input.json
```

## 📊 Output Format

```json
{
  "cv": {
    "title": "...",
    "skills": [...],
    "description": "..."
  },
  "results": [
    {
      "job_id": "job_1",
      "title": "...",
      "cosine_similarity": 0.82,
      "rule1_title_match": "PASS - Title overlap: 100.00% >= 75%",
      "rule2_skill_match": "PASS - Found 5 matching skills: ...",
      "final": "OK"
    }
  ],
  "metrics": {
    "ok_ratio": 0.6,
    "avg_similarity_ok": 0.71,
    "similarity_distribution": {
      "min": 0.32,
      "max": 0.89,
      "mean": 0.58
    },
    "top_3_jobs": ["job_3", "job_7", "job_1"],
    "failed_jobs": ["job_2", "job_5"]
  }
}
```

## 📈 Metrics

- **OK Ratio**: Tỷ lệ jobs được match (OK / Total)
- **Average Similarity (OK)**: Cosine similarity trung bình của các jobs OK
- **Similarity Distribution**: Min, Max, Mean của tất cả similarities
- **Top 3 Jobs**: 3 jobs có similarity cao nhất (chỉ tính jobs OK)
- **Failed Jobs**: Danh sách jobs bị loại (NG)

## 🔧 Cấu Hình

### Model
- **Default**: `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base`
- **Output Dimension**: 768
- **Device**: CPU (hoặc GPU nếu có)

### Rule Thresholds
- **Title Overlap**: >= 75%
- **Skill Match**: >= 1 skill

### Skill Synonyms
Hệ thống hỗ trợ semantic matching với skill synonyms:
- `TensorFlow` ≈ `Deep Learning Framework`
- `React` ≈ `ReactJS` ≈ `Frontend Framework`
- `FastAPI` ≈ `REST API`
- Và nhiều hơn nữa...

## 📝 Ví Dụ

### Example 1: Perfect Match
```
CV: "Senior Python Developer"
Job: "Senior Python Developer"
Result: OK (Rule1: PASS 100%, Rule2: PASS)
```

### Example 2: Skill Match
```
CV: "Python Developer" với skills: ["Python", "FastAPI"]
Job: "Backend Engineer" với requirements: "Python, REST APIs"
Result: OK (Rule1: FAIL, Rule2: PASS - Python matched)
```

### Example 3: No Match
```
CV: "Frontend Developer" với skills: ["React", "TypeScript"]
Job: "Data Engineer" với requirements: "Spark, Hadoop"
Result: NG (Rule1: FAIL, Rule2: FAIL)
```

## 🎯 Best Practices

1. **CV Input**:
   - Cung cấp đầy đủ title, skills, experience, description
   - Skills nên là list hoặc comma-separated string

2. **Job Input**:
   - Mỗi job cần có: job_id, title, requirements, description
   - Requirements nên liệt kê rõ các skills/technologies cần thiết

3. **Interpretation**:
   - Cosine similarity cao không đảm bảo match (cần check rules)
   - Rule-based filtering đảm bảo match có cơ sở thực tế

## 🔬 Technical Details

### Text Normalization
- Lowercase conversion
- Vietnamese accent removal
- Stopword removal
- Special character removal

### Semantic Matching
- Skill synonyms mapping
- Partial word matching
- Related concept matching

### Embedding
- Sentence-level embedding
- L2 normalization
- Cosine similarity computation

## 📚 References

- Two-Tower Architecture for Recommendation Systems
- Semantic Similarity in NLP
- Job-CV Matching Research

## 🐛 Troubleshooting

### Low Cosine Similarity
- Model có thể chưa được fine-tune tốt
- Có thể cần điều chỉnh temperature hoặc retrain

### Rule Matching Issues
- Kiểm tra skill synonyms mapping
- Điều chỉnh threshold nếu cần

## 📞 Support

Nếu có vấn đề, kiểm tra:
1. Input format đúng chưa
2. Model đã load thành công chưa
3. Logs trong `logs/cv_job_matching_result.json`

