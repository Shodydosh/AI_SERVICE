# Translation Backends - Hướng Dẫn Chọn Backend Dịch Thuật

## 📋 Tổng Quan

VietnameseTranslator hỗ trợ nhiều backend dịch thuật khác nhau. Bạn có thể chọn backend phù hợp với nhu cầu:

| Backend | Offline | Chất lượng | Tốc độ | Yêu cầu | Khuyến nghị |
|---------|---------|-----------|--------|---------|-------------|
| **argostranslate** | ✅ | ⭐⭐⭐ | ⚡⚡⚡ | Thấp | ⭐⭐⭐⭐⭐ |
| **EasyNMT** | ✅ | ⭐⭐⭐⭐ | ⚡⚡ | GPU (tùy chọn) | ⭐⭐⭐⭐ |
| **transformers** | ✅ | ⭐⭐⭐⭐⭐ | ⚡ | RAM/GPU | ⭐⭐⭐ |
| **deep_translator** | ❌ | ⭐⭐⭐⭐⭐ | ⚡⚡⚡ | Internet | ⭐⭐⭐ |

## 🚀 Cài Đặt

### Option 1: argostranslate (Khuyến nghị - Offline, miễn phí)

```bash
pip install argostranslate
```

**Ưu điểm:**
- ✅ Chạy offline, không cần internet
- ✅ Miễn phí, open source
- ✅ Tự động tải package ngôn ngữ lần đầu
- ✅ Không cần API key
- ✅ Tốc độ nhanh

**Nhược điểm:**
- ⚠️ Chất lượng dịch tốt nhưng không bằng Google Translate
- ⚠️ Cần tải package ngôn ngữ (~100MB) lần đầu

**Sử dụng:**
```python
from src.utils.vietnamese_translator import VietnameseTranslator

translator = VietnameseTranslator(backend='argostranslate')
# Hoặc để tự động chọn
translator = VietnameseTranslator(backend='auto')
```

---

### Option 2: EasyNMT (Neural Models)

```bash
pip install easynmt
```

**Ưu điểm:**
- ✅ Chạy offline sau khi tải model
- ✅ Chất lượng dịch tốt (neural models)
- ✅ Hỗ trợ nhiều ngôn ngữ
- ✅ Có thể fine-tune

**Nhược điểm:**
- ⚠️ Tải model lần đầu (~500MB-2GB)
- ⚠️ Cần RAM/GPU cho model lớn
- ⚠️ Chậm hơn argostranslate

**Sử dụng:**
```python
translator = VietnameseTranslator(backend='easynmt')
```

---

### Option 3: Transformers với Helsinki-NLP

```bash
pip install transformers torch
```

**Ưu điểm:**
- ✅ Chất lượng dịch rất tốt
- ✅ Chạy offline sau khi tải model
- ✅ Có thể chọn model phù hợp

**Nhược điểm:**
- ⚠️ Tải model lần đầu (~300MB-1GB)
- ⚠️ Cần RAM/GPU
- ⚠️ Chậm hơn các option khác

**Sử dụng:**
```python
translator = VietnameseTranslator(backend='transformers')
```

---

### Option 4: deep_translator (Google Translate)

```bash
pip install deep-translator
```

**Ưu điểm:**
- ✅ Chất lượng dịch tốt nhất
- ✅ Tốc độ nhanh
- ✅ Không cần tải model

**Nhược điểm:**
- ❌ Cần internet
- ❌ Có thể bị rate limit
- ❌ Phụ thuộc vào Google

**Sử dụng:**
```python
translator = VietnameseTranslator(backend='deep_translator')
```

---

## 🎯 Khuyến Nghị

### Cho Production (Offline):
```bash
pip install argostranslate
```
→ Sử dụng `backend='argostranslate'` hoặc `backend='auto'`

### Cho Development/Testing:
```bash
pip install deep-translator
```
→ Sử dụng `backend='deep_translator'` (nếu có internet)

### Cho Chất Lượng Cao (Offline):
```bash
pip install easynmt
```
→ Sử dụng `backend='easynmt'`

---

## ⚙️ Cấu Hình

### Tự động chọn backend tốt nhất:
```python
translator = VietnameseTranslator(backend='auto')
```

### Chỉ định backend cụ thể:
```python
translator = VietnameseTranslator(backend='argostranslate')
```

### Kiểm tra backend nào đang được sử dụng:
```python
translator = VietnameseTranslator()
print(f"Using backend: {translator.backend_type}")
```

---

## 🔧 Troubleshooting

### Lỗi: "No translation backend available"
**Giải pháp:** Cài đặt ít nhất một backend:
```bash
pip install argostranslate
```

### Lỗi: argostranslate không tìm thấy package
**Giải pháp:** Package sẽ tự động tải lần đầu khi chạy. Đảm bảo có kết nối internet lần đầu.

### Lỗi: EasyNMT/Transformers quá chậm
**Giải pháp:** 
- Sử dụng GPU nếu có
- Hoặc chuyển sang argostranslate (nhanh hơn)

### Lỗi: deep_translator bị rate limit
**Giải pháp:**
- Thêm delay giữa các request
- Hoặc chuyển sang backend offline (argostranslate)

---

## 📝 Ví Dụ Sử Dụng

```python
from src.utils.vietnamese_translator import VietnameseTranslator

# Khởi tạo với auto-select
translator = VietnameseTranslator(backend='auto')

# Dịch text
text = "Nhân Viên Kế Toán tại công ty FPT"
translated = translator.translate(text)
print(translated)  # "Accounting Staff at FPT company"

# Dịch batch
texts = ["Kỹ sư phần mềm", "Quản lý dự án"]
translated_batch = translator.translate_batch(texts)
print(translated_batch)  # ["Software Engineer", "Project Manager"]
```

---

## 🔄 Migration từ Google Translate

Nếu bạn đang dùng Google Translate và muốn chuyển sang offline:

1. **Cài đặt argostranslate:**
   ```bash
   pip install argostranslate
   ```

2. **Code tự động sẽ chuyển sang argostranslate** nếu bạn dùng `backend='auto'`

3. **Hoặc chỉ định rõ:**
   ```python
   translator = VietnameseTranslator(backend='argostranslate')
   ```

Không cần thay đổi code khác, API giống nhau!









