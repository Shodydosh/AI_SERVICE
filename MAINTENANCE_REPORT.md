# 📋 BÁO CÁO BẢO TRÌ DỰ ÁN - MAINTENANCE REPORT

**Ngày tạo:** 2025-12-04  
**Phạm vi:** Toàn bộ dự án AI_SERVICE

---

## 🔍 PHÂN TÍCH TỔNG QUAN

### 1. FILES CẦN XÓA (Unnecessary Files)

#### 1.1. Root Level - Markdown Files (Đã xóa một số, còn lại cần xem xét)
- ✅ `readme.md` - Có thể merge vào docs chính
- ✅ `check_progress.bat` - Script cũ, không dùng
- ✅ `run_benchmark_parameter_variations.bat` - Có thể merge vào script Python
- ✅ `RUN_POSTGRESQL_WORKFLOW.bat` - Có thể merge vào script Python
- ✅ `test_db_connection.py` - Test file ở root, nên move vào tests/

#### 1.2. Scripts - Duplicate/Redundant Workflow Scripts
- ⚠️ `run_full_workflow_3_fields.py` - Có thể merge với `run_complete_system.py`
- ⚠️ `run_full_workflow_with_logging.py` - Có thể merge với `run_complete_system.py`
- ⚠️ `show_10_samples_with_recommendations.py` - Tương tự `test_jd_to_jd_matching.py`
- ⚠️ `test_enhanced_matching.py` - Có thể merge với `test_all_features.py`
- ⚠️ `test_complete_system.py` - Có thể merge với `test_all_features.py`

#### 1.3. Data Files - Processed/Sample Files
- ⚠️ `data/processed/*.csv` - Nhiều sample files có thể xóa, chỉ giữ processed chính
  - `candidates_sample_10.csv`
  - `candidates_sample_2.csv`
  - `candidates_sample_20.csv`
  - `job_data_sample_10.csv`
  - `job_data_sample_2.csv`
  - `job_data_sample_20.csv`
  - `candidates_dataset.csv` (duplicate với raw)
  - `job_data.csv` (duplicate với raw)

#### 1.4. Old Indices
- ⚠️ `indices/candidate_index.faiss` - Old single-field index
- ⚠️ `indices/candidate_index.pkl` - Old single-field index
- ⚠️ `indices/jd_index.faiss` - Old single-field index
- ⚠️ `indices/jd_index.pkl` - Old single-field index
- ✅ Giữ lại `indices/multi_field/` - Đang sử dụng

#### 1.5. Log Files
- ⚠️ `logs/architecture_evaluation.log` - Log cũ
- ⚠️ `logs/workflow_*.log` - Có thể archive hoặc xóa logs cũ > 30 ngày

#### 1.6. Reports - Old Analysis Files
- ⚠️ `reports/benchmark_csv/analysis/` - Cần kiểm tra nội dung
- ⚠️ `reports/benchmark_csv/logs/` - Logs cũ có thể xóa
- ⚠️ `reports/benchmark_variations/logs/` - Logs cũ có thể xóa

#### 1.7. Docs - Duplicate/Obsolete Documentation
- ⚠️ `docs/AUTO_FAISS_BUILD_SUMMARY.md` - Có thể merge vào FAISS_BUILD_GUIDE.md
- ⚠️ `docs/ENHANCED_SYSTEM_SUMMARY.md` - Có thể merge vào COMPLETE_SYSTEM_OVERVIEW.md
- ⚠️ `docs/SYSTEM_FLOW_SUMMARY.md` - Có thể merge vào SYSTEM_FLOW_QUICK.md
- ⚠️ `docs/PROJECT_WORKFLOW.md` - Có thể merge vào RUN_AND_DEBUG_GUIDE.md
- ⚠️ `docs/FIX_MEMORY_ERROR.md` - Fix log cũ, có thể archive

### 2. CODE CẦN MERGE/SIMPLIFY

#### 2.1. Workflow Scripts - Merge 3 scripts thành 1
**Files:**
- `run_full_workflow_3_fields.py`
- `run_complete_system.py`
- `run_full_workflow_with_logging.py`

**Đề xuất:** Merge thành `scripts/run_full_workflow.py` với:
- Tất cả features từ 3 scripts
- Logging tích hợp
- Options để skip steps

#### 2.2. Test Scripts - Consolidate
**Files:**
- `test_all_features.py`
- `test_enhanced_matching.py`
- `test_complete_system.py`

**Đề xuất:** Merge thành `scripts/test_system.py` với:
- Test tất cả features
- Test enhanced matching
- Test complete system
- Có thể chọn test nào để chạy

#### 2.3. Repository Classes - Check for Duplication
**Files:**
- `src/database/repository.py` - Old repository?
- `src/database/multi_field_repository.py` - Current repository

**Cần kiểm tra:** Nếu `repository.py` không dùng nữa thì xóa

#### 2.4. Embedding Generators - Check for Duplication
**Files trong `src/embeddings/`:**
- `generator.py` - Old?
- `multi_field_generator.py` - Current
- `field_mapping_embedding.py` - Old?
- `advanced_field_mapping_embedding.py` - Old?
- `improved_field_mapping_embedding.py` - Old?

**Cần kiểm tra:** Chỉ giữ lại `multi_field_generator.py` nếu các file khác không dùng

#### 2.5. Matching Services - Consolidate
**Files:**
- `src/services/matching_service.py` - Old?
- `src/services/multi_filter_matching_service.py` - Current
- `src/services/enhanced_multi_filter_matching_service.py` - Enhanced
- `src/services/enhanced_matching_with_all_features.py` - Full features

**Cần kiểm tra:** Xem `matching_service.py` có còn dùng không

#### 2.6. Data Processing Utils
**Files:**
- `src/utils/clean_data.py` - Standalone function
- `src/utils/data_preprocessor.py` - Class-based

**Đề xuất:** Merge `clean_data.py` vào `data_preprocessor.py`

### 3. CODE QUALITY IMPROVEMENTS

#### 3.1. Fix t-SNE Parameter
**File:** `scripts/visualize_embeddings_tsne.py`
- ❌ `n_iter=1000` - Parameter không tồn tại trong scikit-learn mới
- ✅ Sửa thành `max_iter=1000`

#### 3.2. Remove Unused Imports
- Cần scan tất cả files để tìm unused imports
- Sử dụng tools như `pylint` hoặc `autoflake`

#### 3.3. Remove Commented Code
- Scan và xóa commented-out code blocks
- Chỉ giữ comments có ý nghĩa

#### 3.4. Consolidate Constants
- Tìm các magic numbers và constants trùng lặp
- Tạo `src/config/constants.py` nếu cần

### 4. PROJECT STRUCTURE IMPROVEMENTS

#### 4.1. Move Test Files
- `test_db_connection.py` (root) → `tests/test_db_connection.py`

#### 4.2. Organize Scripts
Tạo subdirectories trong `scripts/`:
- `scripts/workflow/` - Workflow scripts
- `scripts/test/` - Test scripts
- `scripts/analysis/` - Analysis scripts
- `scripts/utils/` - Utility scripts

#### 4.3. Clean Data Directory
- `data/processed/` - Chỉ giữ files chính, xóa samples
- `data/sample/` - Có thể xóa nếu không cần

### 5. DOCUMENTATION CONSOLIDATION

#### 5.1. Merge Similar Docs
- `SYSTEM_FLOW_QUICK.md` + `SYSTEM_FLOW_SUMMARY.md` → `SYSTEM_FLOW.md`
- `COMPLETE_SYSTEM_OVERVIEW.md` + `ENHANCED_SYSTEM_SUMMARY.md` → `SYSTEM_OVERVIEW.md`
- `PROJECT_WORKFLOW.md` → Merge vào `RUN_AND_DEBUG_GUIDE.md`

#### 5.2. Archive Old Fix Docs
- Move `FIX_MEMORY_ERROR.md` vào `docs/archive/` hoặc xóa

---

## 📊 TỔNG KẾT ĐỀ XUẤT

### Files để XÓA (Ước tính: ~25-30 files)
1. **Root:** 5 files (bat, test files, readme cũ)
2. **Scripts:** 5-7 duplicate/redundant scripts
3. **Data:** 7 sample CSV files
4. **Indices:** 4 old index files
5. **Logs:** 2+ old log files
6. **Docs:** 4-5 duplicate/obsolete docs

### Files để MERGE (Ước tính: ~10 files → 3-4 files)
1. **Workflow scripts:** 3 → 1
2. **Test scripts:** 3 → 1
3. **Docs:** 4-5 → 2-3

### Code Improvements
1. Fix t-SNE parameter
2. Remove unused imports
3. Remove commented code
4. Consolidate constants

### Structure Improvements
1. Move test files
2. Organize scripts into subdirectories
3. Clean data directory

---

## ⚠️ LƯU Ý QUAN TRỌNG

1. **Backup trước khi xóa:** Đảm bảo có backup hoặc git commit trước khi xóa
2. **Kiểm tra dependencies:** Đảm bảo không có file nào đang được import
3. **Test sau khi merge:** Chạy test suite sau khi merge code
4. **Documentation:** Cập nhật docs sau khi thay đổi structure

---

## ✅ XÁC NHẬN CẦN THIẾT

**Bạn có muốn tôi thực hiện các thay đổi này không?**

1. Xóa các files không cần thiết
2. Merge các scripts trùng lặp
3. Sửa code quality issues
4. Cải thiện project structure
5. Consolidate documentation

**Vui lòng xác nhận để tiếp tục!**

