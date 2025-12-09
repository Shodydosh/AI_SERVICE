# GIẢI THÍCH CHI TIẾT RULES MATCHING

## TỔNG QUAN

Hệ thống matching sử dụng **RuleMatcher** class với 2 rules chính để đánh giá sự phù hợp giữa CV (Candidate) và Job:

- **Rule 1**: Title Match - So sánh title của candidate với title của job
- **Rule 2**: Skill Match - So sánh skills của candidate với requirements của job

**File:** `src/utils/rule_matcher.py`

---

## CẤU HÌNH MẶC ĐỊNH

```python
RuleMatcher(
    title_overlap_threshold: float = 0.60,  # 60% - ngưỡng cho title match
    skill_score_threshold: float = 0.8,      # 0.8 - ngưỡng cho skill match
    use_semantic: bool = True,               # Sử dụng semantic similarity
    use_tfidf: bool = True                   # Sử dụng TF-IDF similarity
)
```

---

## RULE 1: TITLE MATCH

### Mục đích
Đánh giá mức độ tương đồng giữa title của candidate và title của job.

### Quy trình xử lý

#### Bước 1: Normalize Text
Chuẩn hóa cả 2 titles trước khi so sánh:

1. **Lowercase**: Chuyển tất cả về chữ thường
   - Ví dụ: "Python Developer" → "python developer"

2. **Bỏ dấu tiếng Việt**: Chuyển ký tự có dấu về không dấu
   - Ví dụ: "Kế toán" → "ke toan", "Lập trình viên" → "lap trinh vien"
   - Mapping đầy đủ: à→a, á→a, đ→d, ế→e, ...

3. **Bỏ ký tự đặc biệt**: Chỉ giữ alphanumeric và spaces
   - Ví dụ: "Python/Java Developer" → "python java developer"

4. **Bỏ stopwords tiếng Việt**: Loại bỏ các từ không có ý nghĩa
   - Stopwords: và, của, cho, với, từ, trong, là, được, có, một, các, như, ...
   - Ví dụ: "Lập trình viên Python" → "lap trinh vien python" → "lap trinh vien python" (không có stopwords)

5. **Bỏ từ có độ dài <= 1**: Loại bỏ ký tự đơn lẻ

#### Bước 2: Tính Multiple Similarity Metrics

Hệ thống tính **4 metrics** khác nhau và lấy **MAX** làm final score:

##### 2.1. Token Jaccard Similarity
- **Công thức**: `J(A, B) = |A ∩ B| / |A ∪ B|`
- **Ý nghĩa**: Tỷ lệ tokens chung so với tổng tokens
- **Ví dụ**:
  - Candidate: "python developer"
  - Job: "senior python developer"
  - Tokens candidate: {python, developer}
  - Tokens job: {senior, python, developer}
  - Intersection: {python, developer} = 2
  - Union: {senior, python, developer} = 3
  - Jaccard = 2/3 = 0.667

##### 2.2. Sequence Similarity (Ratcliff/Obershelp)
- **Thuật toán**: So sánh chuỗi ký tự, tìm các subsequence chung dài nhất
- **Ý nghĩa**: Đo độ tương đồng về mặt chuỗi ký tự
- **Ví dụ**:
  - "python developer" vs "python dev"
  - SequenceMatcher tìm các đoạn chung và tính ratio

##### 2.3. TF-IDF Cosine Similarity
- **Công thức**: Cosine similarity giữa 2 TF-IDF vectors
- **Ý nghĩa**: Đo độ quan trọng của từ trong context
- **Cách tính**:
  1. Tạo TF-IDF vector cho mỗi title
  2. Tính cosine similarity giữa 2 vectors
  3. Kết quả: 0.0 - 1.0
- **Lưu ý**: Chỉ tính nếu `use_tfidf=True`

##### 2.4. Semantic Similarity (SBERT)
- **Model**: Sử dụng SentenceTransformer để encode titles thành embeddings
- **Model ưu tiên**: `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base` (hỗ trợ tiếng Việt)
- **Fallback**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Công thức**: Cosine similarity giữa 2 embeddings
- **Ý nghĩa**: Hiểu ngữ nghĩa, không chỉ từ khóa
- **Ví dụ**: "Lập trình viên Python" và "Python Developer" có semantic similarity cao
- **Lưu ý**: Chỉ tính nếu `use_semantic=True` và model load thành công

#### Bước 3: Final Title Score
```python
final_title_score = max(
    token_jaccard,
    sequence_similarity,
    tfidf_similarity if use_tfidf else 0.0,
    semantic_similarity if use_semantic else 0.0
)
```

**Lấy MAX** của tất cả metrics → Đảm bảo không bỏ sót match tốt.

#### Bước 4: Đánh giá PASS/FAIL
```python
if final_title_score >= 0.60:  # 60%
    status = "PASS"
else:
    status = "FAIL"
```

### Output của Rule 1
```python
{
    "status": "PASS" | "FAIL",
    "score": 0.75,  # final_title_score
    "threshold": 0.60,
    "reasons": [
        "Final title score 75.00% >= threshold 60%",
        "Best metric: semantic_similarity = 0.75"
    ],
    "debug": {
        "token_jaccard": 0.667,
        "sequence_similarity": 0.65,
        "tfidf_similarity": 0.72,
        "semantic_similarity": 0.75,
        "token_analysis": {
            "matched_tokens": ["python", "developer"],
            "candidate_only_tokens": [],
            "job_only_tokens": ["senior"]
        }
    }
}
```

---

## RULE 2: SKILL MATCH

### Mục đích
Đánh giá mức độ phù hợp giữa skills của candidate và requirements của job.

### Quy trình xử lý

#### Bước 1: Chuẩn bị dữ liệu
1. **Combine job text**: Gộp `job_requirements` và `job_description` thành 1 text
2. **Normalize job text**: Áp dụng `normalize_text()` như Rule 1
3. **Extract candidate skills**: Tách skills từ text (split by comma, semicolon, newline)

#### Bước 2: Normalize Skill
Mỗi skill được normalize trước khi match:

1. **Lowercase**: "Python" → "python"
2. **Bỏ prefix/suffix**: 
   - "Experience with Python" → "python"
   - "Python proficiency" → "python"
3. **Normalize variants**:
   - "react.js", "reactjs", "react js" → "reactjs"
4. **Bỏ version numbers**:
   - "python3", "python 3.9" → "python"
5. **Bỏ dấu tiếng Việt**: Giống Rule 1
6. **Bỏ ký tự đặc biệt**: Chỉ giữ alphanumeric và spaces

#### Bước 3: Match từng skill (theo thứ tự ưu tiên)

Với mỗi skill của candidate, hệ thống kiểm tra theo thứ tự sau (dừng khi tìm thấy match đầu tiên):

##### 3.1. Exact Match (+1.0 điểm)
- **Điều kiện**: Skill normalized có trong job text normalized
- **Ví dụ**:
  - Candidate skill: "python"
  - Job text: "... python required ..."
  - → Match! +1.0 điểm

##### 3.2. Synonym Match (+0.8 điểm)
- **Điều kiện**: Bất kỳ variation/synonym của skill có trong job text
- **Variations bao gồm**:
  - **Synonyms từ SKILL_SYNONYMS**: 
    - "react" → ["reactjs", "react.js", "frontend", "ui framework"]
    - "nodejs" → ["node.js", "node", "backend"]
  - **Vietnamese-English translations từ VIETNAMESE_ENGLISH_SKILLS**:
    - "ke toan" → ["accounting", "accountant", "bookkeeping"]
    - "lap trinh" → ["programming", "coding", "development"]
- **Ví dụ**:
  - Candidate skill: "react"
  - Job text: "... reactjs required ..."
  - → Match via synonym! +0.8 điểm

##### 3.3. Pattern Match (+0.6 điểm)
- **Điều kiện**: Regex pattern của skill match trong job text
- **Cách tạo pattern**: Escape skill và cho phép spaces/dots/hyphens giữa các từ
- **Ví dụ**:
  - Skill: "react native"
  - Pattern: `react[\s\.\-]?native`
  - Job text: "... react-native ..."
  - → Match! +0.6 điểm

##### 3.4. Partial Match (+0.5 điểm)
- **Điều kiện**: 
  - Skill có nhiều từ (multi-word)
  - Ít nhất 2 từ trong skill có trong job text
- **Ví dụ**:
  - Candidate skill: "machine learning"
  - Job text: "... machine intelligence and deep learning ..."
  - Matched words: "machine" (1), "learning" (1) = 2/2
  - → Match! +0.5 điểm

##### 3.5. Category-level Match (+0.7 điểm hoặc +0.2 bonus)
- **Điều kiện**: Skill thuộc category và có skill khác trong cùng category xuất hiện trong job
- **Skill Categories**:
  - `frontend`: react, vue, angular, javascript, html, css, ...
  - `backend`: python, java, nodejs, go, django, flask, ...
  - `devops`: aws, docker, kubernetes, jenkins, ...
  - `database`: postgresql, mysql, mongodb, redis, ...
  - `mobile`: react native, flutter, ios, android, ...
  - `data`: python, pandas, spark, tensorflow, ...
- **Logic**:
  - Nếu skill chưa match → +0.7 điểm (category match)
  - Nếu skill đã match (exact/synonym/pattern/partial) → +0.2 bonus
- **Ví dụ**:
  - Candidate skill: "react"
  - Job text: "... vue.js and angular required ..."
  - React thuộc category "frontend"
  - Vue và Angular cũng thuộc "frontend"
  - → Category match! +0.7 điểm

#### Bước 4: Tính Total Skill Score
```python
total_score = sum(score_contribution của tất cả matched skills)
```

**Lưu ý**:
- Mỗi skill chỉ được tính 1 lần (match đầu tiên tìm thấy)
- Generic skills (office, excel, english, ...) bị bỏ qua trừ khi job text có đề cập cụ thể

#### Bước 5: Đánh giá PASS/FAIL
```python
if skill_score >= 0.8:
    status = "PASS"
else:
    status = "FAIL"
```

### Output của Rule 2
```python
{
    "status": "PASS",
    "score": 2.6,  # total_score
    "threshold": 0.8,
    "reasons": [
        "Total skill score 2.60 >= threshold 0.8",
        "Matched 3 out of 5 skills",
        "Exact matches (2): python, fastapi",
        "Synonym matches (1): react"
    ],
    "debug": {
        "matched_skills": ["python", "fastapi", "react"],
        "exact_matches": ["python", "fastapi"],
        "synonym_matches": ["react"],
        "partial_matches": [],
        "regex_matches": [],
        "category_matches": [],
        "skill_contributions": [
            {
                "skill": "python",
                "type": "exact",
                "score": 1.0,
                "details": "Exact match: 'python' found in job text"
            },
            {
                "skill": "fastapi",
                "type": "exact",
                "score": 1.0,
                "details": "Exact match: 'fastapi' found in job text"
            },
            {
                "skill": "react",
                "type": "synonym",
                "score": 0.8,
                "details": "Synonym match: 'react' matched via 'reactjs'",
                "synonym_used": "reactjs"
            }
        ],
        "total_candidate_skills": 5,
        "match_count": 3,
        "total_score": 2.6,
        "categories_found": ["frontend"]
    }
}
```

---

## FINAL DECISION LOGIC

### Quy trình
Sau khi có kết quả từ Rule 1 và Rule 2, hệ thống đưa ra quyết định cuối cùng:

```python
def final_decision(rule1_result, rule2_result):
    title_score = rule1_result['score']
    skill_score = rule2_result['score']
    title_pass = rule1_result['status'] == "PASS"
    skill_pass = rule2_result['status'] == "PASS"
    
    # Decision logic
    if title_score >= 0.60:
        final_status = "OK"
        reason = "Title strong match (score >= 60%)"
    elif skill_score >= 0.8:
        final_status = "OK"
        reason = "Skill match đủ mạnh (score >= 0.8) despite low title similarity"
    else:
        final_status = "NG"
        reason = "Cả title và skill không đạt"
    
    return {
        'final_status': final_status,
        'reason': reason,
        'rule1': rule1_result,
        'rule2': rule2_result
    }
```

### Logic quyết định

1. **Nếu title_score >= 60%** → **OK**
   - Title match tốt → Đủ điều kiện match
   - Không cần kiểm tra skill

2. **Nếu title_score < 60% NHƯNG skill_score >= 0.8** → **OK**
   - Title không match nhưng skills phù hợp → Vẫn OK
   - Ví dụ: "Backend Developer" vs "Python Developer" (title khác nhưng skills match)

3. **Nếu cả 2 đều không đạt** → **NG**
   - title_score < 60% VÀ skill_score < 0.8 → Không phù hợp

### Output cuối cùng
```python
{
    "final_status": "OK" | "NG",
    "reason": "Title strong match (score: 75.00% >= 60%)",
    "final_title_score": 0.75,
    "skill_score": 2.6,
    "rule1": { ... },
    "rule2": { ... }
}
```

---

## CÁC TÍNH NĂNG ĐẶC BIỆT

### 1. Cross-language Matching
- **Hỗ trợ tiếng Việt**: Bỏ dấu, translate Vietnamese-English
- **Ví dụ**: "Kế toán" ↔ "accounting", "Lập trình viên" ↔ "developer"

### 2. Skill Normalization
- **Xử lý variants**: react.js, reactjs, react js → reactjs
- **Bỏ version**: python3, python 3.9 → python
- **Xử lý prefix/suffix**: "Experience with Python" → "python"

### 3. Category-level Matching
- **Nhóm skills theo category**: frontend, backend, devops, database, mobile, data
- **Match category**: Nếu có skill cùng category trong job → vẫn tính điểm

### 4. Generic Skills Filtering
- **Blacklist**: office, excel, word, english, communication, teamwork, ...
- **Logic**: Bỏ qua trừ khi job text có đề cập cụ thể

### 5. Explainability
- Mỗi rule trả về **reasons** và **debug info** chi tiết
- Giúp hiểu tại sao match/không match

---

## VÍ DỤ THỰC TẾ

### Ví dụ 1: Match tốt
**Candidate:**
- Title: "Python Developer"
- Skills: ["python", "fastapi", "postgresql"]

**Job:**
- Title: "Senior Python Developer"
- Requirements: "Python, FastAPI, PostgreSQL required"

**Kết quả:**
- Rule 1: title_score = 0.75 (PASS) → Final: **OK**
- Rule 2: skill_score = 3.0 (PASS) → Final: **OK**

### Ví dụ 2: Title không match nhưng skill match
**Candidate:**
- Title: "Backend Developer"
- Skills: ["python", "django", "postgresql"]

**Job:**
- Title: "Python Developer"
- Requirements: "Python, Django, PostgreSQL required"

**Kết quả:**
- Rule 1: title_score = 0.55 (FAIL)
- Rule 2: skill_score = 3.0 (PASS) → Final: **OK** (skill đủ mạnh)

### Ví dụ 3: Không match
**Candidate:**
- Title: "Frontend Developer"
- Skills: ["react", "vue", "javascript"]

**Job:**
- Title: "Java Developer"
- Requirements: "Java, Spring Boot required"

**Kết quả:**
- Rule 1: title_score = 0.30 (FAIL)
- Rule 2: skill_score = 0.0 (FAIL) → Final: **NG**

---

## TỔNG KẾT

### Rule 1: Title Match
- **4 metrics**: Jaccard, Sequence, TF-IDF, Semantic
- **Final score**: MAX của 4 metrics
- **Threshold**: 0.60 (60%)
- **PASS nếu**: score >= 0.60

### Rule 2: Skill Match
- **5 loại match**: Exact (1.0), Synonym (0.8), Pattern (0.6), Partial (0.5), Category (0.7)
- **Total score**: Tổng điểm của tất cả matched skills
- **Threshold**: 0.8
- **PASS nếu**: score >= 0.8

### Final Decision
- **OK nếu**: title_score >= 0.60 HOẶC skill_score >= 0.8
- **NG nếu**: Cả 2 đều không đạt

---

**Lưu ý**: Các thresholds (0.60, 0.8) có thể điều chỉnh khi khởi tạo RuleMatcher.



