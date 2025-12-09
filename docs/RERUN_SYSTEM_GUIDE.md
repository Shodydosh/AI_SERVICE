# Hướng Dẫn Chạy Lại Toàn Bộ Hệ Thống

## 🚀 Quick Start

### Chạy Toàn Bộ Hệ Thống (Khuyến Nghị)

```bash
python scripts/run_complete_system.py
```

Script này sẽ tự động:
1. ✅ Khởi tạo database tables
2. ✅ Process embeddings (JD và Candidates)
3. ✅ Test enhanced matching với tất cả features

### Chạy Từng Bước

```bash
# Bước 1: Init database
python scripts/run_complete_system.py --skip-processing

# Bước 2: Process embeddings
python scripts/run_complete_system.py --skip-init --skip-testing

# Bước 3: Test matching
python scripts/run_complete_system.py --skip-init --skip-processing
```

## 📋 Các Options

```bash
python scripts/run_complete_system.py \
    --skip-init              # Skip database initialization
    --skip-processing        # Skip embedding processing
    --jd-file PATH           # Path to JD CSV file
    --candidate-file PATH    # Path to Candidate CSV file
    --batch-size 50          # Batch size for processing
    --test-candidate-id ID   # Candidate ID for testing
    --test-top-k 10          # Number of top matches
```

## ✅ Tất Cả Features Đã Tích Hợp

### Core Features
- ✅ Hybrid Search (Semantic + Keyword)
- ✅ Reranking (Cross-encoder)
- ✅ Dynamic Filtering
- ✅ Contextual Embeddings
- ✅ Negative Signals
- ✅ Caching

### New Features
- ✅ Explainability
- ✅ Diversity & Fairness
- ✅ Multi-Criteria Optimization
- ✅ Metrics Dashboard
- ✅ A/B Testing

## 📊 Output

Hệ thống sẽ hiển thị:
- Database initialization status
- Embedding processing progress
- Matching results với explanations
- Metrics dashboard
- A/B testing metrics

## 🔧 Troubleshooting

### Nếu không tìm thấy jobs:
- Kiểm tra xem đã process embeddings chưa
- Kiểm tra candidate_id có tồn tại trong database
- Kiểm tra logs để xem lỗi cụ thể

### Nếu lỗi database:
- Kiểm tra PostgreSQL đang chạy
- Kiểm tra connection settings trong `config/settings.py`
- Kiểm tra quyền của database user

## 📝 Notes

- Script tự động detect file paths nếu không chỉ định
- Tất cả features được bật mặc định
- Có thể tắt từng feature trong code nếu cần

