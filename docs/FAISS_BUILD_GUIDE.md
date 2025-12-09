# 🔍 HƯỚNG DẪN BUILD VÀ SỬ DỤNG FAISS INDICES

## 📋 TỔNG QUAN

Hệ thống sử dụng FAISS (Facebook AI Similarity Search) để tăng tốc độ tìm kiếm vector similarity. FAISS indices được lưu vào disk và tự động load khi khởi động matching service.

---

## 🎯 2 CÁCH BUILD FAISS INDICES

### Cách 1: Tự động build sau khi xử lý embeddings (Khuyến nghị)

Sử dụng option `--build-faiss` khi xử lý embeddings:

```bash
# Xử lý embeddings và tự động build FAISS
python scripts/process_multi_field_embeddings.py \
    --jd-file data/filtered/jds_with_skills.csv \
    --candidate-file data/filtered/candidates_with_skills.csv \
    --batch-size 100 \
    --build-faiss
```

**Ưu điểm:**
- Tự động build ngay sau khi embeddings được lưu vào PostgreSQL
- Đảm bảo FAISS indices luôn đồng bộ với dữ liệu
- Một lệnh duy nhất cho toàn bộ workflow

---

### Cách 2: Build riêng biệt (Khi cần rebuild)

Sử dụng script riêng để build FAISS indices:

```bash
# Build FAISS indices từ embeddings đã có trong PostgreSQL
python scripts/build_multi_field_faiss.py
```

**Các tùy chọn:**

```bash
# Với các tham số tùy chỉnh
python scripts/build_multi_field_faiss.py \
    --index-type HNSW \
    --base-path indices/multi_field \
    --ef-search 128 \
    --ef-construction 200 \
    --M 32
```

**Tham số:**
- `--index-type`: Loại index (Flat, IVF, HNSW) - Mặc định: HNSW
- `--base-path`: Đường dẫn lưu indices - Mặc định: `indices/multi_field`
- `--ef-search`: Tham số ef_search cho HNSW - Mặc định: 128
- `--ef-construction`: Tham số ef_construction cho HNSW - Mặc định: 200
- `--M`: Tham số M cho HNSW - Mặc định: 32

**Khi nào dùng:**
- Rebuild indices sau khi cập nhật embeddings
- Thay đổi tham số FAISS index
- Chỉ build indices mà không cần xử lý embeddings mới

---

## 📁 CẤU TRÚC FILE FAISS

Sau khi build, các file sau sẽ được tạo tại `indices/multi_field/`:

```
indices/multi_field/
├── jd_title_index.faiss              # FAISS index cho title embeddings
├── jd_skills_index.faiss             # FAISS index cho skills embeddings
├── jd_requirement_index.faiss        # FAISS index cho requirement embeddings
├── jd_title_id_map.pkl               # Map: index position -> job_id
├── jd_skills_id_map.pkl
├── jd_requirement_id_map.pkl
├── jd_title_reverse_map.pkl          # Map: job_id -> index position
├── jd_skills_reverse_map.pkl
└── jd_requirement_reverse_map.pkl
```

---

## 🔄 WORKFLOW HOÀN CHỈNH

### Bước 1: Xử lý và lưu embeddings vào PostgreSQL

```bash
python scripts/process_multi_field_embeddings.py \
    --jd-file data/filtered/jds_with_skills.csv \
    --candidate-file data/filtered/candidates_with_skills.csv \
    --batch-size 100 \
    --build-faiss
```

Workflow này sẽ:
1. ✅ Khởi tạo database tables (nếu chưa có)
2. ✅ Generate 3 embeddings cho mỗi record (Title, Skills, Requirements)
3. ✅ Lưu embeddings vào PostgreSQL
4. ✅ Build và lưu FAISS indices (nếu có `--build-faiss`)

---

### Bước 2: Sử dụng FAISS trong matching

Khi khởi động matching service, FAISS indices sẽ tự động được load:

```python
# Trong MultiFilterMatchingService.__init__
# Tự động load indices nếu có, nếu không thì build từ database
if (base_path / "jd_title_index.faiss").exists():
    self.faiss_manager.load_indices(base_path)
    logger.info("Loaded existing multi-field FAISS indices")
else:
    logger.warning("Multi-field FAISS indices not found. Building from database...")
    self.faiss_manager.build_indices_from_db(db)
```

---

## ⚙️ TỐI ƯU HIỆU NĂNG

### Chọn Index Type

1. **HNSW** (Khuyến nghị - Mặc định)
   - Tốt nhất cho datasets lớn (>10k records)
   - Tìm kiếm nhanh với độ chính xác cao
   - Sử dụng nhiều RAM hơn

2. **Flat**
   - Đơn giản, chính xác 100%
   - Chậm hơn với datasets lớn
   - Phù hợp datasets nhỏ (<10k records)

3. **IVF**
   - Cân bằng giữa tốc độ và độ chính xác
   - Cần training trước

### Tùy chỉnh tham số HNSW

```bash
# Tăng độ chính xác (chậm hơn một chút)
python scripts/build_multi_field_faiss.py \
    --ef-search 256 \
    --ef-construction 400 \
    --M 64

# Tăng tốc độ (giảm độ chính xác một chút)
python scripts/build_multi_field_faiss.py \
    --ef-search 64 \
    --ef-construction 100 \
    --M 16
```

**Tham số HNSW:**
- `ef_search`: Tăng → Chính xác hơn nhưng chậm hơn (64-256)
- `ef_construction`: Tăng → Build lâu hơn nhưng tốt hơn (100-400)
- `M`: Số lượng kết nối (16-64)

---

## 🔧 TROUBLESHOOTING

### Lỗi: "No embeddings found in database"

**Nguyên nhân:** Chưa có embeddings trong PostgreSQL

**Giải pháp:**
```bash
# Xử lý embeddings trước
python scripts/process_multi_field_embeddings.py \
    --jd-file data/filtered/jds_with_skills.csv \
    --candidate-file data/filtered/candidates_with_skills.csv
```

---

### Lỗi: "Index file not found"

**Nguyên nhân:** FAISS indices chưa được build

**Giải pháp:**
```bash
# Build indices
python scripts/build_multi_field_faiss.py
```

---

### Indices cũ không đồng bộ với database

**Nguyên nhân:** Embeddings đã được cập nhật nhưng indices chưa rebuild

**Giải pháp:**
```bash
# Rebuild indices
python scripts/build_multi_field_faiss.py
```

**Lưu ý:** Hệ thống sẽ tự động build từ database nếu không tìm thấy indices file, nhưng tốt hơn là rebuild thủ công để kiểm soát.

---

### Out of Memory khi build indices

**Nguyên nhân:** Dataset quá lớn

**Giải pháp:**
1. Giảm batch_size trong `build_indices_from_db()`
2. Sử dụng index type nhẹ hơn (IVF thay vì HNSW)
3. Giảm tham số M trong HNSW

---

## 📊 SO SÁNH HIỆU NĂNG

| Index Type | Tốc độ | Độ chính xác | RAM Usage | Phù hợp cho |
|------------|--------|--------------|-----------|-------------|
| **Flat** | Chậm | 100% | Thấp | <10k records |
| **IVF** | Trung bình | 95-98% | Trung bình | 10k-100k records |
| **HNSW** | **Rất nhanh** | **98-99%** | **Cao** | **>10k records (Khuyến nghị)** |

---

## 🔄 REBUILD INDICES KHI CẦN

Khi nào cần rebuild:

1. ✅ Sau khi thêm/xóa/cập nhật embeddings trong database
2. ✅ Khi thay đổi tham số FAISS index
3. ✅ Khi indices file bị lỗi hoặc mất
4. ✅ Sau khi restore database từ backup

**Lệnh rebuild:**

```bash
# Xóa indices cũ (tùy chọn)
rm -rf indices/multi_field/*

# Rebuild
python scripts/build_multi_field_faiss.py
```

---

## 📝 GHI CHÚ QUAN TRỌNG

1. **FAISS indices phải đồng bộ với PostgreSQL**
   - Luôn rebuild sau khi cập nhật embeddings
   - Kiểm tra timestamp nếu cần

2. **Backup indices**
   - Indices file có thể lớn (>100MB)
   - Nên backup cùng với database

3. **Tự động load**
   - Matching service tự động load indices khi khởi động
   - Nếu không tìm thấy, sẽ build từ database (chậm hơn)

4. **Multi-field indices**
   - Hệ thống dùng 3 indices riêng biệt cho 3 fields
   - Tất cả đều cần được build và lưu

---

## 🚀 QUICK START

```bash
# 1. Xử lý embeddings và build FAISS (tất cả trong một)
python scripts/process_multi_field_embeddings.py \
    --jd-file data/filtered/jds_with_skills.csv \
    --candidate-file data/filtered/candidates_with_skills.csv \
    --build-faiss

# 2. Hoặc build riêng sau
python scripts/build_multi_field_faiss.py

# 3. Kiểm tra indices đã được tạo
ls -lh indices/multi_field/
```

---

**Version**: 1.0.0  
**Last Updated**: 2025-01-XX


