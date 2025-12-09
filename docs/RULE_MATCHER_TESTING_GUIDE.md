# Rule Matcher Testing Guide

## 🧪 Hướng Dẫn Testing

### 1. Chạy Unit Tests

```bash
# Chạy tất cả tests
python -m unittest tests.test_rule_matcher -v

# Chạy test cụ thể
python -m unittest tests.test_rule_matcher.TestRuleMatcher.test_rule1_title_match -v
```

### 2. Test Cases Coverage

#### ✅ Text Normalization
- Vietnamese accents removal
- Stopwords removal
- Special characters handling
- Empty string handling

#### ✅ Skill Normalization
- Version normalization (python3 → python)
- Variant normalization (react.js → reactjs)
- Prefix removal (Experience with React → react)
- Accent removal

#### ✅ Title Similarity
- Sequence similarity (Ratcliff/Obershelp)
- Token Jaccard
- TF-IDF cosine similarity (if available)
- Semantic similarity (if available)
- Final score calculation

#### ✅ Skill Matching
- Exact match scoring
- Synonym match scoring
- Partial match scoring
- Pattern match scoring
- Category-level matching
- Generic skill filtering

#### ✅ Final Decision
- Title pass → OK
- Skill pass (title fail) → OK
- Both fail → NG

### 3. Manual Testing Examples

#### Example 1: Good Match
```python
from src.utils.rule_matcher import RuleMatcher

matcher = RuleMatcher()

result = matcher.evaluate_match(
    candidate_title="Senior Python Developer",
    candidate_skills=["Python", "FastAPI", "PostgreSQL", "Docker"],
    job_title="Senior Python Developer",
    job_requirements="Python, FastAPI, PostgreSQL, Docker required",
    job_description="We are looking for an experienced Python developer..."
)

print(f"Status: {result['final_status']}")  # Should be "OK"
print(f"Title Score: {result['final_title_score']:.2f}")  # Should be >= 0.82
print(f"Skill Score: {result['skill_score']:.2f}")  # Should be >= 1.2
```

#### Example 2: Title Match Only
```python
result = matcher.evaluate_match(
    candidate_title="Python Developer",
    candidate_skills=["Python"],
    job_title="Python Developer",
    job_requirements="Java required",
    job_description=None
)

print(f"Status: {result['final_status']}")  # Should be "OK" (title match)
print(f"Title Score: {result['final_title_score']:.2f}")  # Should be >= 0.82
print(f"Skill Score: {result['skill_score']:.2f}")  # Should be < 1.2
```

#### Example 3: Skill Match Only
```python
result = matcher.evaluate_match(
    candidate_title="Backend Developer",
    candidate_skills=["Python", "FastAPI", "Django", "PostgreSQL"],
    job_title="Software Engineer",
    job_requirements="Python, FastAPI, Django, PostgreSQL required",
    job_description=None
)

print(f"Status: {result['final_status']}")  # Should be "OK" (skill match)
print(f"Title Score: {result['final_title_score']:.2f}")  # Should be < 0.82
print(f"Skill Score: {result['skill_score']:.2f}")  # Should be >= 1.2
```

#### Example 4: Poor Match
```python
result = matcher.evaluate_match(
    candidate_title="Python Developer",
    candidate_skills=["Python"],
    job_title="Java Developer",
    job_requirements="Java, Spring Boot required",
    job_description=None
)

print(f"Status: {result['final_status']}")  # Should be "NG"
print(f"Title Score: {result['final_title_score']:.2f}")  # Should be < 0.82
print(f"Skill Score: {result['skill_score']:.2f}")  # Should be < 1.2
```

#### Example 5: Vietnamese Text
```python
result = matcher.evaluate_match(
    candidate_title="Nhà Phát Triển Python",
    candidate_skills=["Python", "FastAPI"],
    job_title="Nha Phat Trien Python",
    job_requirements="Yêu cầu Python và FastAPI",
    job_description=None
)

print(f"Status: {result['final_status']}")  # Should be "OK"
print(f"Title Score: {result['final_title_score']:.2f}")  # Should handle Vietnamese
print(f"Skill Score: {result['skill_score']:.2f}")  # Should handle Vietnamese
```

### 4. Testing với Two-Tower Pipeline

Đảm bảo RuleMatcher vẫn tương thích với Two-Tower matching service:

```python
from src.services.two_tower_matching_service import TwoTowerMatchingService
from src.database.connection import SessionLocal

db = SessionLocal()
service = TwoTowerMatchingService(db)

# Service sẽ sử dụng RuleMatcher internally nếu cần
# (hiện tại đã loại bỏ rule matching khỏi service, nhưng vẫn tương thích)
```

### 5. Performance Testing

```python
import time
from src.utils.rule_matcher import RuleMatcher

matcher = RuleMatcher()

# Test với nhiều pairs
test_cases = [
    ("Python Developer", ["Python"], "Python Developer", "Python required", None),
    ("Java Developer", ["Java"], "Java Developer", "Java required", None),
    # ... thêm nhiều test cases
]

start_time = time.time()
for case in test_cases:
    matcher.evaluate_match(*case)
end_time = time.time()

print(f"Processed {len(test_cases)} cases in {end_time - start_time:.2f}s")
print(f"Average: {(end_time - start_time) / len(test_cases) * 1000:.2f}ms per case")
```

### 6. Edge Cases Testing

```python
# Empty strings
result = matcher.evaluate_match("", [], "", None, None)

# None values
result = matcher.evaluate_match(None, None, None, None, None)

# Very long texts
long_title = "Senior " * 100 + "Python Developer"
result = matcher.evaluate_match(long_title, ["Python"], "Python Developer", "Python", None)

# Special characters
result = matcher.evaluate_match(
    "Python & Java Developer!!!",
    ["Python", "Java"],
    "Python Java Developer",
    "Python Java",
    None
)
```

## ✅ Expected Results

### All Tests Should Pass
- 14 unit tests
- All edge cases handled
- Vietnamese and English support
- Fallback when optional dependencies missing

### Performance
- Title similarity: < 100ms (without semantic)
- Skill scoring: < 50ms per candidate
- Full evaluation: < 150ms per pair

### Accuracy Improvements
- **Before**: ~70% accuracy, many false-positives
- **After**: Expected ~85%+ accuracy, fewer false-positives
- Better separation between high/medium/low similarity pairs








