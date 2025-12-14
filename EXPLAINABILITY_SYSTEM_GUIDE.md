# HỆ THỐNG EXPLAINABILITY CHO TWO-TOWER MATCHING

## TỔNG QUAN

Hệ thống explainability cung cấp 5 levels giải thích cho việc matching CV-Job, giúp người dùng hiểu tại sao hệ thống đề xuất một công việc cụ thể.

---

## CẤU TRÚC HỆ THỐNG

### 1. Core Components

#### `src/utils/explanation_generator.py`
- **ExplanationGenerator**: Class chính để generate explanations
- **AuditLogger**: Class để log explainability data cho audit trail

#### `src/utils/explanation_storage.py`
- Utility functions để lưu/đọc explanations từ database

#### `src/database/models.py`
- **ProcessedCandidateRecommendation**: Model đã được mở rộng với explanation fields

---

## 5 LEVELS EXPLAINABILITY

### Level 1: Rule Matching (Deterministic)

**Mục đích**: Giải thích dựa trên rules rõ ràng, dễ hiểu.

**Output**:
```json
{
  "level": 1,
  "type": "rule_matching",
  "rules_triggered": [
    {
      "rule": "title_similarity",
      "score": 0.75,
      "status": "PASS",
      "details": "Python Developer ↔ Senior Python Developer",
      "matched_tokens": ["python", "developer"],
      "percent": 75.0
    },
    {
      "rule": "skill_overlap",
      "score": 2.6,
      "status": "PASS",
      "overlap": ["python", "fastapi", "postgresql"],
      "matched_count": 3,
      "total_count": 5,
      "percent": 60.0
    }
  ],
  "final_status": "OK",
  "summary": "Triggered 2 rule(s)"
}
```

**Lợi ích**:
- Dễ hiểu, không cần kiến thức về ML
- Có thể logging vào PostgreSQL
- Transparent và auditable

---

### Level 2: Embedding Similarity (Semantic Features)

**Mục đích**: Giải thích dựa trên cosine similarity của embeddings.

**Output**:
```json
{
  "level": 2,
  "type": "embedding_similarity",
  "embedding_scores": {
    "title_similarity": 0.83,
    "title_similarity_percent": 83.0,
    "skills_similarity": 0.76,
    "skills_similarity_percent": 76.0,
    "experience_requirement_similarity": 0.69,
    "experience_requirement_similarity_percent": 69.0,
    "combined_similarity": 0.76,
    "combined_similarity_percent": 76.0
  },
  "interpretation": {
    "title_match": "83.0%",
    "skills_match": "76.0%",
    "experience_match": "69.0%"
  }
}
```

**Lợi ích**:
- Hiển thị rõ ràng từng component (title, skills, experience)
- Giúp hiểu tại sao hệ thống chọn job A thay vì job B
- Tránh cảm giác "AI tự quyết"

---

### Level 3: Humanized Explanation (Natural Language)

**Mục đích**: Tạo explanation bằng ngôn ngữ tự nhiên, dễ đọc.

**Output**:
```json
{
  "level": 3,
  "type": "humanized_explanation",
  "explanation_text": "Tiêu đề 'Python Developer' của bạn tương đồng 75.0% với tiêu đề công việc 'Senior Python Developer'. Hồ sơ của bạn có 3/5 kỹ năng (60.0%) phù hợp với yêu cầu công việc, bao gồm: python, fastapi, postgresql. Về mặt ngữ nghĩa, tiêu đề của bạn có độ tương đồng cao (83.0%) với mô tả công việc. Dựa trên các tiêu chí trên, hệ thống đánh giá bạn phù hợp với công việc này.",
  "explanation_text_en": "...",
  "components": [
    "Tiêu đề 'Python Developer' của bạn tương đồng 75.0%...",
    "Hồ sơ của bạn có 3/5 kỹ năng...",
    ...
  ]
}
```

**Lợi ích**:
- Dễ đọc, thân thiện với người dùng
- Có thể hiển thị trực tiếp trong UI
- Hỗ trợ cả tiếng Việt và tiếng Anh

---

### Level 4: Counterfactual Explanation (What-if)

**Mục đích**: Gợi ý cách cải thiện match score.

**Output**:
```json
{
  "level": 4,
  "type": "counterfactual",
  "current_score": 2.6,
  "suggestions": [
    {
      "skill": "react",
      "action": "add",
      "estimated_score_improvement": 0.1,
      "message": "Nếu bạn thêm kỹ năng 'react', điểm phù hợp có thể tăng thêm khoảng 10.0%."
    },
    {
      "skill": "aws",
      "action": "add",
      "estimated_score_improvement": 0.15,
      "message": "Nếu bạn thêm kỹ năng 'aws', điểm phù hợp có thể tăng thêm khoảng 15.0%."
    }
  ],
  "missing_skills_count": 5,
  "message": "Có 5 kỹ năng trong yêu cầu công việc mà bạn chưa có."
}
```

**Lợi ích**:
- Giúp candidate hiểu cách cải thiện profile
- Tương tự LinkedIn's "Skills to add" feature
- Tăng engagement và user experience

---

### Level 5: Confidence Score

**Mục đích**: Tính toán confidence score tổng hợp từ nhiều nguồn.

**Công thức**:
```python
confidence = 0.4 * embedding_score + 
             0.5 * normalized_rule_score + 
             0.05 * recency_score + 
             0.05 * popularity_score
```

**Output**:
```json
{
  "level": 5,
  "type": "confidence_score",
  "final_confidence": 0.87,
  "final_confidence_percent": 87.0,
  "components": {
    "embedding_score": 0.76,
    "rule_score": 2.6,
    "normalized_rule_score": 0.87,
    "recency_score": null,
    "popularity_score": null
  },
  "weights": {
    "embedding": 0.4,
    "rule": 0.5,
    "recency": 0.05,
    "popularity": 0.05
  },
  "interpretation": "Cao - Phù hợp tốt"
}
```

**Interpretation Levels**:
- >= 90%: "Rất cao - Phù hợp xuất sắc"
- >= 75%: "Cao - Phù hợp tốt"
- >= 60%: "Trung bình - Phù hợp khá"
- >= 40%: "Thấp - Phù hợp yếu"
- < 40%: "Rất thấp - Không phù hợp"

---

## COMPREHENSIVE EXPLANATION

Khi gọi `generate_comprehensive_explanation()`, hệ thống sẽ tạo tất cả 5 levels:

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "levels": {
    "level1_rule": { ... },
    "level2_embedding": { ... },
    "level3_humanized": { ... },
    "level4_counterfactual": { ... },
    "level5_confidence": { ... }
  },
  "summary": {
    "final_status": "OK",
    "confidence_percent": 87.0,
    "title_match_percent": 75.0,
    "skill_match_percent": 60.0
  }
}
```

---

## DATABASE SCHEMA

### ProcessedCandidateRecommendation (Extended)

```sql
ALTER TABLE processed_candidate_recommendations 
ADD COLUMN rule_scores TEXT,                    -- JSON: Level 1 results
ADD COLUMN embedding_scores TEXT,               -- JSON: Level 2 results
ADD COLUMN explanation_text TEXT,               -- Level 3 humanized text
ADD COLUMN comprehensive_explanation TEXT,    -- JSON: All levels
ADD COLUMN confidence_score FLOAT;             -- Level 5 confidence

CREATE INDEX idx_processed_candidate_confidence 
ON processed_candidate_recommendations(confidence_score);
```

---

## USAGE

### 1. Trong Test Script

```python
from src.utils.explanation_generator import ExplanationGenerator
from src.utils.rule_matcher import RuleMatcher

# Initialize
rule_matcher = RuleMatcher()
explanation_generator = ExplanationGenerator()

# Get rule result
rule_result = rule_matcher.evaluate_match(...)

# Compute embedding similarities
embedding_scores = {
    'title_similarity': title_sim,
    'skills_similarity': skills_sim,
    'experience_requirement_similarity': exp_req_sim,
    'combined_similarity': combined_sim
}

# Generate comprehensive explanation
explanation = explanation_generator.generate_comprehensive_explanation(
    rule_result=rule_result,
    embedding_scores=embedding_scores,
    candidate_title=candidate.title,
    job_title=job.title,
    candidate_skills=candidate_skills,
    job_requirements=job.requirement,
    rule_matcher=rule_matcher
)
```

### 2. Lưu vào Database

```python
from src.utils.explanation_storage import save_explanation_to_db

save_explanation_to_db(
    db=db,
    candidate_id=candidate_id,
    job_id=job_id,
    comprehensive_explanation=explanation,
    similarity_score=similarity,
    rank=rank
)
```

### 3. Đọc từ Database

```python
from src.utils.explanation_storage import get_explanation_from_db

explanation = get_explanation_from_db(
    db=db,
    candidate_id=candidate_id,
    job_id=job_id
)
```

---

## AUDIT LOGGING

### AuditLogger

```python
from src.utils.explanation_generator import AuditLogger

audit_logger = AuditLogger()

# Log explanation
audit_logger.log_explanation(
    candidate_id=candidate_id,
    job_id=job_id,
    explanation=comprehensive_explanation,
    features_used=['title_embedding', 'skills_embedding', 'experience_embedding'],
    rules_triggered=['title_similarity', 'skill_overlap']
)

# Get all logs
logs = audit_logger.get_audit_logs()
```

**Audit Log Structure**:
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "candidate_id": "candidate_123",
  "job_id": "job_456",
  "features_used": ["title_embedding", "skills_embedding", "experience_embedding"],
  "rules_triggered": ["title_similarity", "skill_overlap"],
  "explanation_summary": {
    "final_status": "OK",
    "confidence": 87.0
  },
  "full_explanation": { ... }
}
```

---

## MIGRATION

### Chạy Migration Script

```bash
python scripts/add_explanation_fields_migration.py
```

Script sẽ:
1. Kiểm tra các cột đã tồn tại
2. Thêm các cột explanation nếu chưa có
3. Tạo index trên confidence_score

---

## TESTING

### Chạy Test với Explanation

```bash
python scripts/test_two_tower_precomputed.py --max-candidates 5 --top-k 10
```

Output sẽ bao gồm:
- Level 1: Rule matching details
- Level 2: Embedding similarities
- Level 3: Humanized explanation
- Level 4: Counterfactual suggestions
- Level 5: Confidence score
- Full JSON explanation (debug mode)

---

## VISUAL EXPLAINABILITY (Level 3 - Visual)

Đã có sẵn scripts:
- `visualize_embeddings_tsne.py`
- `visualize_tsne_production.py`

Có thể sử dụng để tạo:
- 2D/3D visualization của embeddings
- Highlight candidate và jobs phù hợp
- Cluster analysis

**Ví dụ explanation**:
"Candidate nằm trong cluster Backend. Các job gần nhất trong không gian vector là Backend Python, Backend Golang, Microservices Engineer."

---

## BEST PRACTICES

### 1. Lưu Explanation cho mọi Recommendation
- Giúp trace lại lịch sử
- Giải trình khi khách hàng hỏi
- Cải thiện model dựa trên feedback

### 2. Hiển thị Level 3 (Humanized) trong UI
- Dễ đọc nhất cho end users
- Có thể kèm Level 1 và Level 5 cho chi tiết

### 3. Sử dụng Level 4 (Counterfactual) để Engagement
- Gợi ý cách cải thiện profile
- Tăng user interaction

### 4. Audit Logging cho Compliance
- Ghi lại tất cả decisions
- Đảm bảo transparency
- Giảm rủi ro "AI kỳ thị"

---

## FUTURE ENHANCEMENTS

1. **Translation Service**: Tích hợp translation API cho explanation_text_en
2. **Recency Score**: Tính dựa trên thời gian job posting
3. **Popularity Score**: Tính dựa trên số lượng applications
4. **A/B Testing**: Test các weights khác nhau cho confidence score
5. **Feedback Loop**: Thu thập user feedback để cải thiện explanations

---

## KẾT LUẬN

Hệ thống explainability cung cấp:
- ✅ 5 levels giải thích đầy đủ
- ✅ Database storage cho audit trail
- ✅ Human-readable explanations
- ✅ Counterfactual suggestions
- ✅ Confidence scoring
- ✅ Audit logging

**Hệ thống không còn là blackbox!**









