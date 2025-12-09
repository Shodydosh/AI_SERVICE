# Tóm tắt Hệ thống Embedding Cache - 12 Giờ

## ✅ Đã hoàn thành

### 1. Embedding Cache Manager
- **File**: `src/services/embedding_cache_manager.py`
- **Tính năng**:
  - In-memory cache với TTL 12 giờ
  - Content hash để phát hiện thay đổi
  - Thread-safe operations
  - Auto-expiration

### 2. Optimized Embedding Service
- **File**: `src/services/embedding_service.py`
- **Tính năng**:
  - Cache-first approach (cache → database → compute)
  - Batch processing cho PostgreSQL
  - Efficient storage với batch inserts
  - Non-blocking realtime queries

### 3. Background Scheduler
- **File**: `src/services/embedding_scheduler.py`
- **Tính năng**:
  - Tự động refresh embeddings mỗi 12 giờ
  - Batch processing
  - Background worker không block realtime

### 4. Database Migration
- **File**: `scripts/create_embedding_timestamp_migration.py`
- **Thêm cột**:
  - `embedding_timestamp`: Timestamp khi embedding được compute
  - `content_hash`: MD5 hash để phát hiện thay đổi nội dung
  - Indexes cho performance

### 5. Scheduler Runner
- **File**: `scripts/run_embedding_scheduler.py`
- **Modes**:
  - Continuous: Background worker
  - Once: Chạy một lần (cho cron jobs)

## 📊 Luồng hoạt động

### Realtime Query (Fast Path)
```
1. User request → TwoTowerMatchingService
2. Check cache → Hit? Return immediately
3. Check database → Fresh (< 12h)? Return
4. Compute embedding → Save → Cache → Return
```

### Background Refresh (12h Cycle)
```
1. Scheduler wakes up every 12 hours
2. Query database for embeddings older than 12h
3. Batch process (100 items/batch)
4. Update embeddings in database
5. Update cache
6. Sleep until next cycle
```

## 🚀 Cách sử dụng

### 1. Chạy Migration (Đã chạy)
```bash
python scripts/create_embedding_timestamp_migration.py
```

### 2. Sử dụng trong Code

#### Option A: OptimizedEmbeddingService (cho Multi-Field)
```python
from src.services.embedding_service import OptimizedEmbeddingService
from src.database.connection import SessionLocal

db = SessionLocal()
service = OptimizedEmbeddingService(db, cache_ttl_hours=12.0)

# Get embedding với cache
embeddings = service.get_candidate_embedding(
    candidate_id="123",
    title="Software Engineer",
    skills="Python",
    experience="5 years"
)
```

#### Option B: TwoTowerMatchingService (đã tích hợp cache)
```python
from src.services.two_tower_matching_service import TwoTowerMatchingService
from src.database.connection import SessionLocal

db = SessionLocal()
service = TwoTowerMatchingService(db, use_cache=True)

# Tìm jobs - tự động dùng cache
matches = service.find_jobs_for_candidate("123", top_k=10)
```

### 3. Setup Background Scheduler

#### Option A: Cron Job (Recommended)
```bash
# Chạy mỗi 12 giờ
0 */12 * * * cd /path/to/AI_SERVICE && python scripts/run_embedding_scheduler.py --once
```

#### Option B: Continuous Worker
```bash
# Background daemon
nohup python scripts/run_embedding_scheduler.py > scheduler.log 2>&1 &
```

#### Option C: Systemd Service (Linux)
```bash
sudo systemctl enable embedding-scheduler
sudo systemctl start embedding-scheduler
```

## 📈 Performance Benefits

### Before (Không có cache)
- Mỗi query: Compute embedding (~100-500ms)
- Database load: High
- Response time: 100-500ms

### After (Với cache)
- Cache hit: < 1ms
- Database hit (fresh): ~10-50ms
- Cache miss: ~100-500ms (nhưng cache cho lần sau)
- Response time: < 1ms (cache) hoặc 10-50ms (database)

### Background Refresh
- Không block realtime queries
- Batch processing: 100 items/batch
- Tự động refresh mỗi 12 giờ

## 🔧 Configuration

### Cache TTL
```python
# Default: 12 hours
service = OptimizedEmbeddingService(db, cache_ttl_hours=12.0)

# Custom: 6 hours (fresh hơn)
service = OptimizedEmbeddingService(db, cache_ttl_hours=6.0)
```

### Batch Size
```python
# Default: 100
scheduler = EmbeddingScheduler(batch_size=100)

# Larger batch: 200 (nhanh hơn nhưng tốn memory)
scheduler = EmbeddingScheduler(batch_size=200)
```

### Scheduler Interval
```python
# Default: 12 hours
scheduler = EmbeddingScheduler(refresh_interval_hours=12.0)

# Custom: 6 hours
scheduler = EmbeddingScheduler(refresh_interval_hours=6.0)
```

## 📝 Monitoring

### Cache Statistics
```python
from src.services.embedding_cache_manager import get_cache_manager

cache = get_cache_manager()
stats = cache.get_stats()
print(stats)
```

### Scheduler Logs
```bash
tail -f embedding_scheduler.log
```

### Database Queries
```sql
-- Check embeddings cần refresh
SELECT COUNT(*) 
FROM candidate_multi_embeddings
WHERE embedding_timestamp < NOW() - INTERVAL '12 hours';
```

## ✅ Checklist

- [x] EmbeddingCacheManager với TTL 12h
- [x] OptimizedEmbeddingService với cache-first
- [x] Background scheduler cho 12h cycle
- [x] Database migration (embedding_timestamp, content_hash)
- [x] Batch processing cho PostgreSQL
- [x] Non-blocking realtime queries
- [x] Documentation và examples
- [x] Scheduler runner script

## 🎯 Kết quả

Hệ thống hiện tại:
- ✅ **Lưu embedding vào PostgreSQL hiệu quả** - Batch inserts, optimized
- ✅ **Cache thông minh** - In-memory với TTL 12h
- ✅ **Chỉ re-embed mỗi 12 giờ** - Automatic scheduler
- ✅ **Không làm chậm realtime** - Cache-first, background processing

## 📚 Tài liệu chi tiết

Xem `docs/EMBEDDING_CACHE_SYSTEM.md` để biết thêm chi tiết.








