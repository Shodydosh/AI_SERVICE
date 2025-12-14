# BÁO CÁO RULES MATCHING

## TỔNG QUAN

Hệ thống sử dụng **RuleMatcher** class để validate CV-Job matches với 2 rules chính.

**File:** `src/utils/rule_matcher.py`

**Class:** `RuleMatcher`

---

## KHỞI TẠO

```python
def __init__(
    self,
    title_overlap_threshold: float = 0.60,  # Default: 0.60 (60%)
    skill_score_threshold: float = 0.8,      # Default: 0.8
    use_semantic: bool = True,
    use_tfidf: bool = True
):
```

**Parameters:**
- `title_overlap_threshold`: Ngưỡng tối thiểu cho title match (default: 0.60)
- `skill_score_threshold`: Ngưỡng tối thiểu cho skill match (default: 0.8)
- `use_semantic`: Có sử dụng semantic similarity không (default: True)
- `use_tfidf`: Có sử dụng TF-IDF similarity không (default: True)

---

## RULE 1: TITLE MATCH

### Phương thức: `rule1_title_match(candidate_title: str, job_title: str)`

### Logic:

1. **Tính title similarity** bằng `compute_title_similarity()`:
   - Token Jaccard similarity
   - Sequence similarity (Ratcliff/Obershelp)
   - TF-IDF cosine similarity (nếu `use_tfidf=True`)
   - Semantic similarity (nếu `use_semantic=True`)
   - **Final score = max của tất cả metrics**

2. **Điều kiện PASS:**
   ```python
   if final_score >= self.title_overlap_threshold:  # >= 0.60
       status = "PASS"
   else:
       status = "FAIL"
   ```

3. **Normalize text:**
   - Lowercase
   - Bỏ dấu tiếng Việt (à→a, á→a, đ→d, ...)
   - Bỏ ký tự đặc biệt, chỉ giữ alphanumeric và spaces
   - Bỏ stopwords tiếng Việt
   - Bỏ từ có độ dài <= 1

### Return:

```python
{
    "status": "PASS" | "FAIL",
    "score": <float>,  # final_title_score
    "threshold": 0.60,
    "reasons": [<list of reasons>],
    "debug": {
        "token_jaccard": <float>,
        "sequence_similarity": <float>,
        "tfidf_similarity": <float>,
        "semantic_similarity": <float>,
        "token_analysis": {
            "candidate_tokens": [...],
            "job_tokens": [...],
            "matched_tokens": [...],
            "candidate_only_tokens": [...],
            "job_only_tokens": [...]
        }
    }
}
```

---

## RULE 2: SKILL MATCH

### Phương thức: `rule2_skill_match(candidate_skills: List[str], job_requirements: Optional[str], job_description: Optional[str])`

### Logic:

1. **Extract skills từ candidate:**
   - Split bằng comma, semicolon, hoặc newline
   - Normalize mỗi skill

2. **Normalize skill:**
   - Lowercase
   - Bỏ prefixes: "experience with", "knowledge of", "proficient in", ...
   - Bỏ suffixes: "experience", "knowledge", "skill", ...
   - Normalize variants: "reactjs", "react.js", "react js" → "react"
   - Bỏ version numbers: "python3", "python 3.9" → "python"
   - Bỏ dấu tiếng Việt
   - Bỏ ký tự đặc biệt

3. **Tính skill score** bằng `compute_skill_score()`:

   **Với mỗi skill:**
   
   a. **Exact match (+1.0):**
      ```python
      if skill_normalized in job_text_normalized:
          score_contribution = 1.0
          match_type = "exact"
      ```
   
   b. **Synonym match (+0.8):**
      ```python
      # Check trong SKILL_SYNONYMS và VIETNAMESE_ENGLISH_SKILLS
      for variation in skill_variations:
          if variation_normalized in job_text_normalized:
              score_contribution = 0.8
              match_type = "synonym"
              break
      ```
   
   c. **Pattern match (+0.6):**
      ```python
      skill_pattern = re.escape(skill_normalized).replace(r'\ ', r'[\s\.\-]?')
      if re.search(skill_pattern, job_text, re.IGNORECASE):
          score_contribution = 0.6
          match_type = "pattern"
      ```
   
   d. **Partial match (+0.5):**
      ```python
      skill_words = skill_normalized.split()
      matched_words = sum(1 for word in skill_words if word in job_text_normalized)
      if matched_words >= min(2, len(skill_words)):
          score_contribution = 0.5
          match_type = "partial"
      ```
   
   e. **Category match (+0.7):**
      ```python
      # Nếu skill thuộc category (frontend, backend, devops, ...)
      # và có skill khác trong cùng category xuất hiện trong job
      if skill_category and related_skill_in_job:
          score_contribution = 0.7
          match_type = "category"
      ```

4. **Tổng score = sum của tất cả score_contribution**

5. **Điều kiện PASS:**
   ```python
   if skill_score >= self.skill_score_threshold:  # >= 0.8
       status = "PASS"
   else:
       status = "FAIL"
   ```

6. **Generic skills blacklist:**
   - Các skills generic như "office", "excel", "word", "english", "communication", ... sẽ bị bỏ qua trừ khi được đề cập rõ ràng trong job

### Return:

```python
{
    "status": "PASS" | "FAIL",
    "score": <float>,  # total_score
    "threshold": 0.8,
    "reasons": [<list of reasons>],
    "debug": {
        "matched_skills": [...],
        "exact_matches": [...],
        "synonym_matches": [...],
        "partial_matches": [...],
        "regex_matches": [...],
        "category_matches": [...],
        "skill_contributions": [
            {
                "skill": "...",
                "normalized": "...",
                "type": "exact|synonym|pattern|partial|category",
                "score": <float>,
                "details": "..."
            }
        ],
        "total_candidate_skills": <int>,
        "match_count": <int>,
        "total_score": <float>,
        "categories_found": [...]
    }
}
```

---

## FINAL DECISION

### Phương thức: `final_decision(rule1_result: Dict, rule2_result: Dict)`

### Logic:

```python
title_score = rule1_result['score']
skill_score = rule2_result['score']
title_pass = rule1_result['status'] == "PASS"
skill_pass = rule2_result['status'] == "PASS"

# Decision logic
if title_score >= self.title_overlap_threshold:  # >= 0.60
    final_status = "OK"
    reason = f"Title strong match (score: {title_score:.2%} >= {self.title_overlap_threshold:.0%})"
elif skill_score >= self.skill_score_threshold:  # >= 0.8
    final_status = "OK"
    reason = f"Skill match đủ mạnh (score: {skill_score:.2f} >= {self.skill_score_threshold}) despite low title similarity ({title_score:.2%})"
else:
    final_status = "NG"
    reason = f"Cả title và skill không đạt: title score {title_score:.2%} < {self.title_overlap_threshold:.0%}, skill score {skill_score:.2f} < {self.skill_score_threshold}"
```

**Điều kiện OK:**
- Title score >= 0.60 **HOẶC**
- Skill score >= 0.8

**Điều kiện NG:**
- Cả title score < 0.60 **VÀ** skill score < 0.8

### Return:

```python
{
    "final_status": "OK" | "NG",
    "reason": <explanation string>,
    "rule1": rule1_result,
    "rule2": rule2_result
}
```

---

## EVALUATE MATCH (Main Method)

### Phương thức: `evaluate_match(candidate_title: str, candidate_skills: List[str], job_title: str, job_requirements: Optional[str] = None, job_description: Optional[str] = None)`

### Flow:

1. Gọi `rule1_title_match(candidate_title, job_title)`
2. Gọi `rule2_skill_match(candidate_skills, job_requirements, job_description)`
3. Gọi `final_decision(rule1_result, rule2_result)`
4. Thêm convenience fields: `final_title_score`, `skill_score`

### Return:

```python
{
    "final_status": "OK" | "NG",
    "reason": <explanation>,
    "final_title_score": <float>,
    "skill_score": <float>,
    "rule1": {
        "status": "PASS" | "FAIL",
        "score": <float>,
        "threshold": 0.60,
        "reasons": [...],
        "debug": {...}
    },
    "rule2": {
        "status": "PASS" | "FAIL",
        "score": <float>,
        "threshold": 0.8,
        "reasons": [...],
        "debug": {...}
    }
}
```

---

## SKILL CATEGORIES

Hệ thống có các skill categories để hỗ trợ category-level matching:

```python
SKILL_CATEGORIES = {
    'frontend': {'react', 'reactjs', 'vue', 'angular', 'javascript', 'typescript', 'html', 'css', ...},
    'backend': {'python', 'java', 'nodejs', 'go', 'rust', 'php', 'django', 'flask', 'fastapi', ...},
    'devops': {'aws', 'azure', 'docker', 'kubernetes', 'jenkins', 'terraform', ...},
    'database': {'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch', ...},
    'mobile': {'react native', 'flutter', 'ios', 'android', ...},
    'data': {'python', 'pandas', 'numpy', 'spark', 'tensorflow', 'pytorch', ...}
}
```

---

## SKILL SYNONYMS

Hệ thống có mapping synonyms để match các biến thể của skill:

```python
SKILL_SYNONYMS = {
    'react': ['reactjs', 'react.js', 'frontend', 'ui framework', ...],
    'python': ['python programming', 'python development', 'python3', ...],
    'nodejs': ['node.js', 'node', 'backend', ...],
    ...
}
```

---

## VIETNAMESE-ENGLISH SKILLS

Hệ thống có mapping tiếng Việt - tiếng Anh:

```python
VIETNAMESE_ENGLISH_SKILLS = {
    'ke toan': ['accounting', 'accountant', 'bookkeeping', ...],
    'lap bao cao tai chinh': ['financial reporting', 'financial statements', ...],
    'giao tiep': ['communication', 'interpersonal skills'],
    ...
}
```

---

## THRESHOLDS TỔNG HỢP

| Rule | Threshold | Default Value | Điều kiện PASS |
|------|-----------|---------------|----------------|
| Rule 1 (Title) | `title_overlap_threshold` | 0.60 (60%) | `final_title_score >= 0.60` |
| Rule 2 (Skill) | `skill_score_threshold` | 0.8 | `total_score >= 0.8` |
| Final Decision | - | - | `title_score >= 0.60` **HOẶC** `skill_score >= 0.8` |

---

## SCORING BREAKDOWN

### Title Score:
- **Token Jaccard**: `len(intersection) / len(union)`
- **Sequence Similarity**: Ratcliff/Obershelp algorithm
- **TF-IDF Similarity**: Cosine similarity của TF-IDF vectors
- **Semantic Similarity**: Cosine similarity của embeddings (SBERT)
- **Final**: `max(token_jaccard, sequence_similarity, tfidf_similarity, semantic_similarity)`

### Skill Score:
- **Exact match**: +1.0
- **Synonym match**: +0.8
- **Pattern match**: +0.6
- **Partial match**: +0.5
- **Category match**: +0.7 (có thể +0.2 bonus nếu đã có match khác)
- **Total**: Sum của tất cả contributions

---

## GHI CHÚ

- Hệ thống hỗ trợ tiếng Việt và tiếng Anh
- Generic skills (office, excel, communication, ...) bị filter trừ khi được đề cập rõ trong job
- Category matching giúp match skills cùng domain (ví dụ: react → frontend category)
- Vietnamese-English translation giúp match skills giữa 2 ngôn ngữ













