# 🔧 Fix Lỗi "Paging File is Too Small"

## 🔍 Vấn Đề

**Lỗi:** `The paging file is too small for this operation to complete. (os error 1455)`

**Nguyên nhân:**
- Models được load vào memory nhưng không được unload sau khi benchmark
- Mỗi model chiếm ~500MB-1GB RAM
- Khi chạy nhiều variations, memory bị đầy
- Windows paging file (virtual memory) quá nhỏ

## ✅ Giải Pháp Đã Áp Dụng

### 1. **Memory Cleanup Sau Mỗi Variation**

Code đã được cập nhật để:
- ✅ Unload model sau khi benchmark xong
- ✅ Xóa embeddings arrays khỏi memory
- ✅ Clear PyTorch cache (nếu dùng GPU)
- ✅ Force garbage collection

### 2. **Giảm Memory Usage**

- ✅ Giảm số lượng test samples (20 → 10 JDs và candidates)
- ✅ Giảm batch test size (100 → 50)
- ✅ Thêm delay giữa các variations để cho phép cleanup

### 3. **Cấu Hình Hệ Thống**

Nếu vẫn gặp lỗi, có thể tăng paging file size:

#### Windows:

1. Mở **System Properties**:
   - Right-click **This PC** → **Properties**
   - Click **Advanced system settings**
   - Tab **Advanced** → Click **Settings** trong **Performance**

2. Tab **Advanced** → Click **Change** trong **Virtual memory**

3. Bỏ chọn **Automatically manage paging file size for all drives**

4. Chọn drive chính (thường là C:)

5. Chọn **Custom size**:
   - **Initial size (MB):** 4096 (4GB)
   - **Maximum size (MB):** 8192 (8GB) hoặc lớn hơn

6. Click **Set** → **OK**

7. **Restart** máy tính

## 🚀 Cách Chạy An Toàn

### Option 1: Chạy Từng Model Một

Thay vì chạy tất cả 50 variations, chạy từng model một:

```bash
# Model 1: SimCSE_Vietnamese (variations 1-5)
python scripts/run_full_benchmark_50_variations.py \
    --variations 1 2 3 4 5 \
    --sample-size 20

# Model 2: Multilingual_MPNet (variations 6-10)
python scripts/run_full_benchmark_50_variations.py \
    --variations 6 7 8 9 10 \
    --sample-size 20
```

### Option 2: Giảm Sample Size

```bash
# Chạy với sample size nhỏ hơn
python scripts/run_full_benchmark_50_variations.py \
    --sample-size 20  # Giảm từ 50 xuống 20
```

### Option 3: Chạy Từng Variation

```bash
# Chạy chỉ 1 variation tại một thời điểm
python scripts/benchmark_from_csv.py \
    --candidate-file "data/raw/candidates_dataset.csv" \
    --jd-file "data/raw/job_data.csv" \
    --sample-size 20 \
    --variations 1
```

## 📊 Kiểm Tra Memory Usage

Tạo script để kiểm tra memory trước khi chạy:

```python
# scripts/check_memory.py
import psutil
import os

process = psutil.Process(os.getpid())
mem_info = process.memory_info()
mem_mb = mem_info.rss / 1024 / 1024

print(f"Current memory usage: {mem_mb:.2f} MB")

# Check available memory
mem = psutil.virtual_memory()
print(f"Total RAM: {mem.total / 1024**3:.2f} GB")
print(f"Available RAM: {mem.available / 1024**3:.2f} GB")
print(f"Used RAM: {mem.percent:.1f}%")

if mem.available < 2 * 1024**3:  # Less than 2GB
    print("⚠️ Warning: Low available memory!")
    print("Consider:")
    print("  1. Close other applications")
    print("  2. Reduce sample size")
    print("  3. Run variations one at a time")
```

## 🔍 Debug Memory Issues

Nếu vẫn gặp lỗi:

1. **Kiểm tra memory hiện tại:**
   ```powershell
   Get-ComputerInfo | Select-Object TotalPhysicalMemory, CsTotalPhysicalMemory
   ```

2. **Kiểm tra paging file:**
   ```powershell
   Get-WmiObject -Class Win32_PageFileUsage | Select-Object Name, AllocatedBaseSize, CurrentUsage
   ```

3. **Monitor memory trong khi chạy:**
   - Mở Task Manager (Ctrl+Shift+Esc)
   - Tab Performance → Memory
   - Xem memory usage trong real-time

## ⚙️ Tối Ưu Thêm

### Nếu có GPU:

Enable GPU để giảm RAM usage:

```python
# Models sẽ được load lên GPU thay vì RAM
# Giảm RAM usage đáng kể
```

Xem hướng dẫn: `docs/ENABLE_GPU.md` (nếu có)

### Nếu có đủ RAM:

Có thể tăng lại sample size sau khi fix memory issues:

```bash
python scripts/run_full_benchmark_50_variations.py \
    --sample-size 50  # Tăng lại nếu memory đủ
```

## 📝 Summary

**Đã fix:**
- ✅ Auto cleanup models sau mỗi variation
- ✅ Giảm memory footprint
- ✅ Garbage collection tự động

**Cần làm thêm nếu vẫn lỗi:**
- ⚠️ Tăng paging file size (Windows)
- ⚠️ Chạy từng model một
- ⚠️ Giảm sample size

**Khuyến nghị:**
- Chạy với `--sample-size 20` để test
- Nếu OK, tăng dần lên 50
- Nếu vẫn lỗi, chạy từng model một




