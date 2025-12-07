# BÁO CÁO HỆ THỐNG AI SERVICE

## Cấu trúc báo cáo

Báo cáo được chia thành **6 sections chính**, mỗi section gồm nhiều chapters:

### Section 1: Tổng quan hệ thống
- Giới thiệu hệ thống
- Mục tiêu và phạm vi
- Công nghệ sử dụng

### Section 2: Kiến trúc và Mô hình
- Kiến trúc Two-Tower Architecture
- Cấu trúc phân lớp
- Two-Tower Model Architecture
- Embedding Encoders
- Embedding Service và Caching

### Section 3: Matching và Tìm kiếm
- Two-Tower Matching Service
- Vector Search với FAISS
- Rule-based Matching

### Section 4: Data và Database
- Data Processing Layer
- Data Preprocessing Utilities
- Database Connection và Configuration
- Database Models
- Database Repositories

### Section 5: API và Services
- FastAPI Application
- API Routes
- Matching Services
- Embedding Services
- Scheduler Services

### Section 6: Training, Utilities và Kết luận
- Training Pipeline
- Evaluation Metrics
- Utilities và Helpers
- Kết luận

---

## Quy tắc viết báo cáo

1. **Ưu tiên tuyệt đối mô tả dựa trên codebase**: Chỉ mô tả đúng những gì đang tồn tại trong codebase
2. **Không sáng tạo thêm**: Không giả định, không suy diễn logic không có trong code
3. **Cấu trúc mỗi chapter**:
   - **A. Phần dựa trên codebase**: Mô tả chính xác các file, function, class, logic, data flow, config...
   - **B. (Tùy chọn) Phần bổ sung ngoài codebase**: Chỉ viết khi được yêu cầu, đánh dấu rõ "(Kiến thức bổ sung – không từ codebase)"

---

**Chờ yêu cầu**: Vui lòng yêu cầu viết chapter cụ thể theo format:
```
Viết Chapter X: <tên chapter>
```
