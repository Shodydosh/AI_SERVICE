# Rule Matcher Improvements Documentation

## 📋 Tổng Quan

File `src/utils/rule_matcher.py` đã được cải tiến toàn diện để tăng độ chính xác, giảm false-positive, và hỗ trợ tốt hơn cho dữ liệu tiếng Việt + Anh.

## 🔄 Các Thay Đổi Chính

### 1. Rule 1: Title Match - Cải Tiến

#### Thay đổi từ Char Jaccard → Character Sequence Similarity
- **Trước**: Sử dụng Char Jaccard (tính overlap ký tự)
- **Sau**: Sử dụng Ratcliff/Obershelp algorithm (SequenceMatcher) để tính similarity theo sequence
- **Lợi ích**: Giảm trường hợp char overlap cao nhưng nghĩa sai

#### Bổ sung TF-IDF Cosine Similarity
- Sử dụng `TfidfVectorizer` từ scikit-learn (nếu có)
- Hỗ trợ tốt hơn cho tiếng Anh
- Fallback nếu không có scikit-learn

#### Bổ sung Semantic Embedding Similarity
- Sử dụng SBERT model (`paraphrase-MiniLM-L6-v2`) để tính semantic similarity
- Hỗ trợ hiểu nghĩa của text, không chỉ từ khóa
- Fallback nếu không có sentence-transformers

#### Tính Final Title Score
```python
final_title_score = max(
    token_jaccard,
    sequence_similarity,
    tfidf_cosine_similarity,
    semantic_similarity
)
```

#### Nâng ngưỡng từ 0.75 → 0.82
- Match phải chặt chẽ hơn
- Giảm false-positive

### 2. Rule 2: Skill Matching - Cải Tiến

#### Loại bỏ Generic Skills
- Blacklist các skill quá chung chung: "office", "excel", "english", "communication"
- Chỉ tính nếu job yêu cầu cụ thể

#### Category-Level Matching
- Nhóm skills theo category:
  - **Frontend**: react, vue, angular, javascript...
  - **Backend**: python, java, nodejs, go...
  - **DevOps**: aws, docker, kubernetes...
  - **Database**: postgresql, mysql, mongodb...
  - **Mobile**: react native, flutter, ios, android...
  - **Data**: spark, hadoop, tensorflow, pytorch...
- Nếu skill trong cùng category với requirement → bonus điểm

#### Scoring System (thay cho boolean)
- **exact match**: +1.0
- **synonym match**: +0.8
- **partial match**: +0.5
- **pattern match**: +0.6
- **same-category**: +0.7 (bonus)
- **Rule 2 PASS** khi `total_score >= 1.2`

#### Chuẩn hóa Skill Mạnh Hơn
- Remove accents
- Normalize variants: "reactjs", "react.js", "react js" → "react"
- Normalize versions: "python3", "python 3.9" → "python"
- Remove prefixes: "Experience with React" → "react"

### 3. Final Decision Logic - Cải Tiến

#### Logic Mới
```python
if final_title_score >= 0.82:
    return "OK"
elif skill_score >= 1.2:
    return "OK"
else:
    return "NG"
```

#### Đảm Bảo
- Title không giống → phải có skill thật sự match
- Skill match ít → không cho pass nếu title sai quá xa

### 4. Cấu Trúc Code

#### Tách Rõ Các Hàm
- `normalize_text()`: Chuẩn hóa text
- `compute_title_similarity()`: Tính title similarity với nhiều metrics
- `compute_skill_score()`: Tính skill matching score
- `final_decision()`: Logic quyết định cuối cùng
- `normalize_skill()`: Chuẩn hóa skill name
- `get_skill_category()`: Lấy category của skill
- `is_generic_skill()`: Kiểm tra generic skill

#### Unit Tests
- File `tests/test_rule_matcher.py` với 14 test cases
- Cover tất cả các hàm chính
- Test cả tiếng Việt và tiếng Anh

#### Comment Đầy Đủ
- Tất cả hàm đều có docstring
- Giải thích rõ ràng parameters và return values

#### Fallback Support
- Nếu không có scikit-learn → TF-IDF disabled
- Nếu không có sentence-transformers → Semantic similarity disabled
- Hệ thống vẫn hoạt động với các metrics cơ bản

## 🚀 Cách Sử Dụng

### Basic Usage
```python
from src.utils.rule_matcher import RuleMatcher

matcher = RuleMatcher(
    title_overlap_threshold=0.82,
    skill_score_threshold=1.2,
    use_semantic=True,
    use_tfidf=True
)

result = matcher.evaluate_match(
    candidate_title="Senior Python Developer",
    candidate_skills=["Python", "FastAPI", "PostgreSQL"],
    job_title="Senior Python Developer",
    job_requirements="Python, FastAPI, PostgreSQL required",
    job_description=None
)

print(result['final_status'])  # "OK" or "NG"
print(result['final_title_score'])  # 0.85
print(result['skill_score'])  # 2.4
```

### Advanced Usage
```python
# Chỉ dùng Rule 1
rule1_status, explanation, details = matcher.rule1_title_match(
    "Python Developer",
    "Python Programmer"
)

# Chỉ dùng Rule 2
rule2_status, explanation, details = matcher.rule2_skill_match(
    ["Python", "FastAPI"],
    "We need Python developers",
    None
)

# Tính skill score riêng
score, details = matcher.compute_skill_score(
    ["Python", "React"],
    "We need Python and React developers",
    None
)
```

## 📊 So Sánh Trước/Sau

### Trước
- Char Jaccard + Token Jaccard
- Boolean PASS/FAIL cho skills
- Threshold 0.75
- Không có category matching
- Không filter generic skills

### Sau
- Sequence Similarity + Token Jaccard + TF-IDF + Semantic
- Scoring system cho skills (0-∞)
- Threshold 0.82
- Category-level matching với bonus
- Filter generic skills
- Chuẩn hóa skill mạnh hơn

## ✅ Compatibility

- **Backward Compatible**: API `evaluate_match()` vẫn giữ nguyên signature
- **Two-Tower Pipeline**: Không ảnh hưởng đến Two-Tower matching service
- **Optional Dependencies**: Hoạt động được ngay cả khi không có scikit-learn/sentence-transformers

## 🧪 Testing

Chạy unit tests:
```bash
python -m unittest tests.test_rule_matcher -v
```

Tất cả 14 tests đều pass ✅

## 📝 Notes

- Semantic model (`paraphrase-MiniLM-L6-v2`) sẽ được download lần đầu khi sử dụng
- TF-IDF vectorizer được tạo mới mỗi lần tính similarity (có thể optimize sau)
- Generic skills blacklist có thể mở rộng thêm
- Skill categories có thể mở rộng thêm













