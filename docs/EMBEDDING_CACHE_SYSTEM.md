# Embedding Cache System - 12-Hour Refresh Cycle

## Tổng quan

Hệ thống cache và quản lý embedding được thiết kế để:
- ✅ **Lưu embedding vào PostgreSQL hiệu quả** - Batch processing, optimized inserts
- ✅ **Cache thông minh** - In-memory cache với TTL 12 giờ
- ✅ **Chỉ re-embed mỗi 12 giờ** - Automatic refresh cycle
- ✅ **Không làm chậm hệ thống realtime** - Cache-first approach, background processing

## Kiến trúc

### 1. EmbeddingCacheManager (`src/services/embedding_cache_manager.py`)

Quản lý cache in-memory với các tính năng:
- **TTL (Time-To-Live)**: 12 giờ mặc định
- **Content Hash**: Phát hiện thay đổi nội dung
- **Thread-safe**: Sử dụng Lock để đảm bảo thread safety
- **Auto-expiration**: Tự động xóa cache hết hạn

**API chính:**
```python
from src.services.embedding_cache_manager import get_cache_manager

cache = get_cache_manager(ttl_hours=12.0)

# Get cached embedding
embedding = cache.get(entity_id, entity_type, content_hash)

# Set cache
cache.set(entity_id, entity_type, embedding, content_hash)

# Check if needs refresh
needs_refresh = cache.needs_refresh(entity_id, entity_type, db_timestamp)
```

### 2. OptimizedEmbeddingService (`src/services/embedding_service.py`)

Service chính để lấy embeddings với caching:
- **Cache-first**: Kiểm tra cache trước, sau đó database, cuối cùng mới compute
- **Batch processing**: Xử lý hàng loạt để tối ưu hiệu năng
- **Efficient storage**: Batch inserts vào PostgreSQL

**Luồng xử lý:**
```
1. Check in-memory cache → Return if valid
2. Check database → Return if fresh (< 12h) and content unchanged
3. Compute new embedding → Save to DB → Cache → Return
```

**API chính:**
```python
from src.services.embedding_service import OptimizedEmbeddingService
from src.database.connection import SessionLocal

db = SessionLocal()
service = OptimizedEmbeddingService(db, cache_ttl_hours=12.0)

# Get candidate embedding (with caching)
embeddings = service.get_candidate_embedding(
    candidate_id="123",
    title="Software Engineer",
    skills="Python, FastAPI",
    experience="5 years"
)

# Get job embedding (with caching)
embeddings = service.get_job_embedding(
    job_id="456",
    title="Backend Developer",
    skills="Python, Django",
    requirement="5+ years experience"
)
```

### 3. EmbeddingScheduler (`src/services/embedding_scheduler.py`)

Background scheduler để refresh embeddings định kỳ:
- **12-hour cycle**: Tự động refresh embeddings cũ hơn 12 giờ
- **Batch processing**: Xử lý hàng loạt để tối ưu
- **Non-blocking**: Chạy background, không ảnh hưởng realtime queries

**Chạy scheduler:**
```bash
# Continuous mode (background worker)
python scripts/run_embedding_scheduler.py

# Run once (for cron jobs)
python scripts/run_embedding_scheduler.py --once

# Custom interval
python scripts/run_embedding_scheduler.py --refresh-interval 12.0 --batch-size 100
```

## Database Schema

### Thêm cột mới vào bảng embedding

```sql
-- candidate_multi_embeddings
ALTER TABLE candidate_multi_embeddings
ADD COLUMN embedding_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW();

ALTER TABLE candidate_multi_embeddings
ADD COLUMN content_hash VARCHAR(64);

CREATE INDEX idx_candidate_multi_embedding_timestamp
ON candidate_multi_embeddings(embedding_timestamp);

-- job_description_multi_embeddings
ALTER TABLE job_description_multi_embeddings
ADD COLUMN embedding_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW();

ALTER TABLE job_description_multi_embeddings
ADD COLUMN content_hash VARCHAR(64);

CREATE INDEX idx_job_multi_embedding_timestamp
ON job_description_multi_embeddings(embedding_timestamp);
```

**Chạy migration:**
```bash
python scripts/create_embedding_timestamp_migration.py
```

## Sử dụng trong Realtime Queries

### Two-Tower Matching Service với Cache

Service đã được tích hợp cache tự động:

```python
from src.services.two_tower_matching_service import TwoTowerMatchingService
from src.database.connection import SessionLocal

db = SessionLocal()
service = TwoTowerMatchingService(db, use_cache=True)

# Tìm jobs cho candidate (sử dụng cache)
matches = service.find_jobs_for_candidate(candidate_id="123", top_k=10)
```

### Manual Embedding Service

Nếu cần lấy embedding trực tiếp:

```python
from src.services.embedding_service import OptimizedEmbeddingService
from src.database.connection import SessionLocal

db = SessionLocal()
service = OptimizedEmbeddingService(db)

# Lấy embedding với cache (không block)
embeddings = service.get_candidate_embedding(
    candidate_id="123",
    title="Software Engineer",
    skills="Python",
    experience="5 years"
)
```

## Background Scheduler Setup

### Option 1: Continuous Background Worker

Chạy như một daemon process:

```bash
# Linux/Mac
nohup python scripts/run_embedding_scheduler.py > scheduler.log 2>&1 &

# Windows (PowerShell)
Start-Process python -ArgumentList "scripts/run_embedding_scheduler.py" -WindowStyle Hidden
```

### Option 2: Cron Job (Recommended)

Chạy định kỳ mỗi 12 giờ:

```bash
# Crontab entry (chạy mỗi 12 giờ)
0 */12 * * * cd /path/to/AI_SERVICE && python scripts/run_embedding_scheduler.py --once
```

### Option 3: Systemd Service (Linux)

Tạo service file `/etc/systemd/system/embedding-scheduler.service`:

```ini
[Unit]
Description=Embedding Refresh Scheduler
After=network.target postgresql.service

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/AI_SERVICE
ExecStart=/usr/bin/python3 scripts/run_embedding_scheduler.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable và start:
```bash
sudo systemctl enable embedding-scheduler
sudo systemctl start embedding-scheduler
sudo systemctl status embedding-scheduler
```

## Monitoring

### Cache Statistics

```python
from src.services.embedding_cache_manager import get_cache_manager

cache = get_cache_manager()
stats = cache.get_stats()
print(stats)
# {
#     'total_entries': 1000,
#     'valid_entries': 950,
#     'expired_entries': 50,
#     'cache_ttl_hours': 12.0
# }
```

### Scheduler Logs

Logs được ghi vào `embedding_scheduler.log`:

```bash
tail -f embedding_scheduler.log
```

### Database Queries

Kiểm tra embeddings cần refresh:

```sql
-- Candidates cần refresh
SELECT candidate_id, embedding_timestamp, 
       NOW() - embedding_timestamp as age
FROM candidate_multi_embeddings
WHERE embedding_timestamp < NOW() - INTERVAL '12 hours'
LIMIT 10;

-- Jobs cần refresh
SELECT job_id, embedding_timestamp,
       NOW() - embedding_timestamp as age
FROM job_description_multi_embeddings
WHERE embedding_timestamp < NOW() - INTERVAL '12 hours'
LIMIT 10;
```

## Performance Tuning

### Cache TTL

Điều chỉnh TTL theo nhu cầu:

```python
# TTL ngắn hơn (6 giờ) - fresh hơn nhưng compute nhiều hơn
service = OptimizedEmbeddingService(db, cache_ttl_hours=6.0)

# TTL dài hơn (24 giờ) - ít compute hơn nhưng có thể stale
service = OptimizedEmbeddingService(db, cache_ttl_hours=24.0)
```

### Batch Size

Tăng batch size để xử lý nhanh hơn (nhưng tốn memory):

```python
# Batch size lớn hơn
scheduler = EmbeddingScheduler(batch_size=200)
```

### Max Items Per Cycle

Giới hạn số items xử lý mỗi cycle:

```python
scheduler = EmbeddingScheduler(max_items_per_cycle=5000)
```

## Troubleshooting

### Cache không hoạt động

1. Kiểm tra cache stats:
```python
cache = get_cache_manager()
print(cache.get_stats())
```

2. Clear expired cache:
```python
cache.clear_expired()
```

### Embeddings không được refresh

1. Kiểm tra scheduler logs:
```bash
tail -f embedding_scheduler.log
```

2. Chạy manual refresh:
```python
from src.services.embedding_scheduler import EmbeddingScheduler
scheduler = EmbeddingScheduler()
stats = scheduler.run_refresh_cycle()
print(stats)
```

### Database connection issues

1. Kiểm tra connection pool:
```python
from src.database.connection import engine
print(engine.pool.status())
```

2. Tăng pool size nếu cần:
```python
# Trong src/database/connection.py
engine = create_engine(
    db_url,
    pool_size=20,
    max_overflow=10
)
```

## Best Practices

1. **Luôn sử dụng cache** cho realtime queries
2. **Chạy scheduler** như background worker hoặc cron job
3. **Monitor cache stats** để tối ưu TTL
4. **Batch processing** cho large datasets
5. **Logging** để debug và monitor

## Migration Guide

### Bước 1: Chạy migration

```bash
python scripts/create_embedding_timestamp_migration.py
```

### Bước 2: Update code sử dụng OptimizedEmbeddingService

Thay vì:
```python
encoder = CandidateTowerEncoder()
embeddings = encoder.encode_candidate(...)
```

Sử dụng:
```python
service = OptimizedEmbeddingService(db)
embeddings = service.get_candidate_embedding(...)
```

### Bước 3: Setup scheduler

```bash
# Test run
python scripts/run_embedding_scheduler.py --once

# Setup cron job
crontab -e
# Add: 0 */12 * * * cd /path/to/AI_SERVICE && python scripts/run_embedding_scheduler.py --once
```

### Bước 4: Monitor và tune

- Monitor cache hit rate
- Adjust TTL nếu cần
- Tune batch size cho performance








