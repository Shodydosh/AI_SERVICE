# 🔧 Tùy Chỉnh 5 Variations của SimCSE_Vietnamese

## 📋 Các Thông Số Có Thể Thay Đổi

### Hiện Tại (5 Variations)

| Variation | Batch Size | Normalize | Use Tokenization | Mô Tả |
|-----------|------------|-----------|------------------|-------|
| v1_bs32_norm | 32 | True | None (default=True) | Standard |
| v2_bs64_norm | 64 | True | None (default=True) | Large batch |
| v3_bs128_norm | 128 | True | None (default=True) | Very large batch |
| v4_bs32_norm_false | 32 | False | None (default=True) | No normalization |
| v5_bs16_norm | 16 | True | None (default=True) | Small batch |

### Các Thông Số Có Thể Thay Đổi

1. **batch_size**: Số lượng texts xử lý cùng lúc
   - Giá trị: 8, 16, 32, 64, 128, 256, ...
   - Ảnh hưởng: Tốc độ xử lý và memory usage
   - Batch lớn hơn → Nhanh hơn nhưng dùng nhiều memory hơn

2. **normalize**: Có normalize embeddings hay không
   - Giá trị: `True` hoặc `False`
   - Ảnh hưởng: Chất lượng similarity matching
   - Normalize=True → Embeddings có norm=1, tốt cho cosine similarity

3. **use_tokenization**: Có dùng Vietnamese tokenization không
   - Giá trị: `True`, `False`, hoặc `None` (dùng model default)
   - Ảnh hưởng: Chất lượng embeddings cho tiếng Việt
   - True → Tokenize tiếng Việt trước khi encode (tốt hơn cho tiếng Việt)

---

## 🔧 Cách Thay Đổi

### Option 1: Sửa Trực Tiếp trong Code

**File:** `src/embeddings/parameter_variations.py`

```python
# 5 parameter variation configurations
PARAMETER_VARIATIONS = [
    {
        "name_suffix": "v1_bs32_norm",
        "batch_size": 32,        # ← Thay đổi ở đây
        "normalize": True,       # ← Thay đổi ở đây
        "use_tokenization": None, # ← Thay đổi ở đây (True/False/None)
        "description": "Standard: batch_size=32, normalize=True"
    },
    {
        "name_suffix": "v2_bs64_norm",
        "batch_size": 64,        # ← Có thể thay thành 128, 256, ...
        "normalize": True,
        "use_tokenization": True, # ← Có thể set True/False
        "description": "Large batch: batch_size=64, normalize=True"
    },
    # ... các variations khác
]
```

### Option 2: Tạo Variations Tùy Chỉnh

**Ví dụ:** Tạo variations với batch_size khác nhau:

```python
PARAMETER_VARIATIONS = [
    {
        "name_suffix": "v1_bs8_norm",
        "batch_size": 8,      # Nhỏ hơn → ít memory hơn
        "normalize": True,
        "use_tokenization": True,
        "description": "Small batch: batch_size=8, normalize=True"
    },
    {
        "name_suffix": "v2_bs32_norm",
        "batch_size": 32,
        "normalize": True,
        "use_tokenization": True,
        "description": "Standard: batch_size=32, normalize=True"
    },
    {
        "name_suffix": "v3_bs64_norm",
        "batch_size": 64,
        "normalize": True,
        "use_tokenization": True,
        "description": "Large batch: batch_size=64, normalize=True"
    },
    {
        "name_suffix": "v4_bs256_norm",
        "batch_size": 256,    # Rất lớn → nhanh nhưng cần nhiều memory
        "normalize": True,
        "use_tokenization": True,
        "description": "Very large batch: batch_size=256, normalize=True"
    },
    {
        "name_suffix": "v5_bs32_no_norm",
        "batch_size": 32,
        "normalize": False,   # Không normalize
        "use_tokenization": True,
        "description": "No normalization: batch_size=32, normalize=False"
    }
]
```

### Option 3: Test Nhiều Combinations

**Ví dụ:** Test tất cả combinations của batch_size và normalize:

```python
PARAMETER_VARIATIONS = []

# Test các batch sizes
batch_sizes = [16, 32, 64, 128]

# Test với normalize=True và False
for bs in batch_sizes:
    # Variation với normalize=True
    PARAMETER_VARIATIONS.append({
        "name_suffix": f"bs{bs}_norm",
        "batch_size": bs,
        "normalize": True,
        "use_tokenization": True,
        "description": f"batch_size={bs}, normalize=True"
    })
    
    # Variation với normalize=False
    PARAMETER_VARIATIONS.append({
        "name_suffix": f"bs{bs}_no_norm",
        "batch_size": bs,
        "normalize": False,
        "use_tokenization": True,
        "description": f"batch_size={bs}, normalize=False"
    })
```

---

## 📊 Ảnh Hưởng Của Từng Thông Số

### batch_size

| Batch Size | Tốc Độ | Memory Usage | Phù Hợp Cho |
|------------|--------|--------------|------------|
| 8-16 | Chậm | Thấp | Máy yếu, ít RAM |
| 32-64 | Trung bình | Trung bình | Máy trung bình |
| 128-256 | Nhanh | Cao | Máy mạnh, nhiều RAM |
| 512+ | Rất nhanh | Rất cao | GPU, server mạnh |

### normalize

- **True**: 
  - ✅ Embeddings có norm=1 (unit vectors)
  - ✅ Tốt cho cosine similarity
  - ✅ Chuẩn hóa giúp so sánh công bằng
  - ⚠️ Có thể mất một chút thông tin

- **False**:
  - ✅ Giữ nguyên magnitude của embeddings
  - ⚠️ Có thể ảnh hưởng đến similarity nếu vectors có độ lớn khác nhau

### use_tokenization

- **True**: 
  - ✅ Tokenize tiếng Việt trước khi encode
  - ✅ Tốt hơn cho tiếng Việt (tách từ đúng cách)
  - ⚠️ Cần thư viện `pyvi`

- **False**: 
  - ✅ Không tokenize, dùng trực tiếp
  - ⚠️ Có thể không tối ưu cho tiếng Việt

- **None**: 
  - ✅ Dùng default của model
  - ✅ Linh hoạt, model tự quyết định

---

## 🎯 Gợi Ý Cấu Hình

### Cho Máy Yếu (RAM < 8GB)
```python
PARAMETER_VARIATIONS = [
    {"name_suffix": "v1_bs8_norm", "batch_size": 8, "normalize": True, "use_tokenization": True},
    {"name_suffix": "v2_bs16_norm", "batch_size": 16, "normalize": True, "use_tokenization": True},
    {"name_suffix": "v3_bs32_norm", "batch_size": 32, "normalize": True, "use_tokenization": True},
]
```

### Cho Máy Trung Bình (RAM 8-16GB)
```python
PARAMETER_VARIATIONS = [
    {"name_suffix": "v1_bs32_norm", "batch_size": 32, "normalize": True, "use_tokenization": True},
    {"name_suffix": "v2_bs64_norm", "batch_size": 64, "normalize": True, "use_tokenization": True},
    {"name_suffix": "v3_bs128_norm", "batch_size": 128, "normalize": True, "use_tokenization": True},
]
```

### Cho Máy Mạnh/GPU (RAM > 16GB)
```python
PARAMETER_VARIATIONS = [
    {"name_suffix": "v1_bs64_norm", "batch_size": 64, "normalize": True, "use_tokenization": True},
    {"name_suffix": "v2_bs128_norm", "batch_size": 128, "normalize": True, "use_tokenization": True},
    {"name_suffix": "v3_bs256_norm", "batch_size": 256, "normalize": True, "use_tokenization": True},
    {"name_suffix": "v4_bs512_norm", "batch_size": 512, "normalize": True, "use_tokenization": True},
]
```

### Test Normalization
```python
PARAMETER_VARIATIONS = [
    {"name_suffix": "v1_bs32_norm", "batch_size": 32, "normalize": True, "use_tokenization": True},
    {"name_suffix": "v2_bs32_no_norm", "batch_size": 32, "normalize": False, "use_tokenization": True},
]
```

---

## 📝 Lưu Ý

1. **Sau khi thay đổi**: Cần chạy lại benchmark để có kết quả mới
2. **Số lượng variations**: Có thể tạo nhiều hơn 5 variations nếu muốn
3. **Testing**: Nên test với sample size nhỏ trước (50-100) để kiểm tra
4. **Memory**: Batch size lớn cần nhiều memory, cẩn thận OOM errors

---

## 🚀 Ví Dụ Thực Tế

### Tạo 10 Variations với Nhiều Combinations

```python
PARAMETER_VARIATIONS = [
    # Batch size variations với normalize=True
    {"name_suffix": "v1_bs16_norm", "batch_size": 16, "normalize": True, "use_tokenization": True},
    {"name_suffix": "v2_bs32_norm", "batch_size": 32, "normalize": True, "use_tokenization": True},
    {"name_suffix": "v3_bs64_norm", "batch_size": 64, "normalize": True, "use_tokenization": True},
    {"name_suffix": "v4_bs128_norm", "batch_size": 128, "normalize": True, "use_tokenization": True},
    
    # Batch size variations với normalize=False
    {"name_suffix": "v5_bs16_no_norm", "batch_size": 16, "normalize": False, "use_tokenization": True},
    {"name_suffix": "v6_bs32_no_norm", "batch_size": 32, "normalize": False, "use_tokenization": True},
    {"name_suffix": "v7_bs64_no_norm", "batch_size": 64, "normalize": False, "use_tokenization": True},
    
    # Tokenization variations
    {"name_suffix": "v8_bs32_norm_no_tok", "batch_size": 32, "normalize": True, "use_tokenization": False},
    {"name_suffix": "v9_bs64_norm_no_tok", "batch_size": 64, "normalize": True, "use_tokenization": False},
    
    # Combination
    {"name_suffix": "v10_bs128_norm_no_tok", "batch_size": 128, "normalize": True, "use_tokenization": False},
]
```

Sau đó chạy benchmark sẽ test tất cả 10 variations này!

