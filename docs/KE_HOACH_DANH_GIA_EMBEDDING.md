# Kế Hoạch Đánh Giá Embedding - Nghiên Cứu So Sánh

## 📋 Mục Tiêu Nghiên Cứu

Đánh giá và so sánh **5 phương pháp embedding khác nhau** để tìm ra phương pháp tối ưu nhất cho hệ thống khuyến nghị việc làm tiếng Việt.

---

## 🎯 5 Phương Pháp Embedding

### 1. **Baseline: SimCSE Vietnamese (Hiện tại)**
- **Model:** `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base`
- **Method:** Simple text concatenation → Single embedding
- **Dimension:** 768
- **Đặc điểm:** Standard approach, baseline để so sánh

### 2. **Weighted Embeddings**
- **Model:** SimCSE Vietnamese
- **Method:** Field-specific weights → Weighted combination
- **Weights:** Skills (7.0), Experience (6.5), Title (7.0), etc.
- **Đặc điểm:** Ưu tiên các trường quan trọng

### 3. **Field-Specific Embeddings**
- **Model:** SimCSE Vietnamese
- **Method:** Tách riêng từng field → Multiple embeddings → Concatenate
- **Fields:** Skills, Experience, Education, Summary (riêng biệt)
- **Đặc điểm:** Preserve field semantics

### 4. **Multi-Vector Embeddings**
- **Model:** SimCSE Vietnamese
- **Method:** Tạo nhiều vectors cho một entity → Average hoặc Max pooling
- **Vectors:** Title vector, Skills vector, Experience vector, Combined vector
- **Đặc điểm:** Capture multiple aspects

### 5. **Ensemble Embeddings**
- **Models:** Multiple models (SimCSE, PhoBERT, etc.)
- **Method:** Combine embeddings từ nhiều models
- **Combination:** Weighted average hoặc concatenation
- **Đặc điểm:** Leverage strengths of multiple models

---

## 📊 Metrics Đánh Giá

### 1. **Similarity Quality Metrics**
- **Top-K Accuracy:** % correct matches trong top K
- **Mean Reciprocal Rank (MRR):** Average of 1/rank của correct match
- **NDCG@K:** Normalized Discounted Cumulative Gain
- **Precision@K:** % relevant trong top K

### 2. **Performance Metrics**
- **Embedding Generation Time:** Thời gian tạo embedding
- **Search Time:** Thời gian tìm kiếm
- **Memory Usage:** Memory footprint
- **Storage Size:** Kích thước lưu trữ

### 3. **Semantic Quality Metrics**
- **Cosine Similarity Distribution:** Phân phối similarity scores
- **Diversity:** Độ đa dạng của kết quả
- **Coverage:** % candidates có matches tốt (>0.8 similarity)

---

## 🔬 Phương Pháp Đánh Giá

### Dataset
- **JD Dataset:** 14,634 jobs
- **Candidate Dataset:** 14,634 candidates
- **Test Set:** 100-500 sample pairs (ground truth)

### Evaluation Process
1. Generate embeddings với 5 phương pháp
2. Lưu vào PostgreSQL (separate tables)
3. Build FAISS indices cho mỗi phương pháp
4. Run similarity search
5. Evaluate với ground truth
6. Compare results

---

## 📈 Kết Quả Mong Đợi

### Best Method Selection Criteria
1. **Highest Accuracy:** Top-K accuracy, MRR, NDCG
2. **Good Performance:** Acceptable generation/search time
3. **Reasonable Memory:** Không quá tốn memory
4. **Practical:** Dễ implement và maintain

---

## 📝 Báo Cáo Kết Quả

Báo cáo sẽ bao gồm:
- So sánh chi tiết 5 phương pháp
- Metrics cho từng phương pháp
- Recommendations
- Implementation guide

