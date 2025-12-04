# 🔧 BÁO CÁO PHÂN TÍCH BẢO TRÌ DỰ ÁN

**Ngày:** 2025-12-04  
**Phạm vi:** Full project scan và analysis

---

## 📊 TỔNG QUAN

### Thống kê hiện tại:
- **Scripts:** 28 Python files
- **Services:** 23 service files
- **Embeddings:** 11 embedding files
- **Docs:** 19 markdown files
- **Data files:** ~20 CSV files (bao gồm samples)

---

## 🗑️ 1. FILES CẦN XÓA

### 1.1 Root Level Files (5 files)
```
✅ readme.md                    - Generic, có thể merge vào docs
✅ check_progress.bat           - Script cũ, không dùng
✅ run_benchmark_parameter_variations.bat - Có thể thay bằng Python script
✅ RUN_POSTGRESQL_WORKFLOW.bat  - Có thể thay bằng Python script
✅ test_db_connection.py        - Nên move vào tests/
```

### 1.2 Duplicate/Redundant Scripts (5-7 files)
```
⚠️ run_full_workflow_3_fields.py      - Merge với run_complete_system.py
⚠️ run_full_workflow_with_logging.py  - Merge với run_complete_system.py
⚠️ show_10_samples_with_recommendations.py - Tương tự test_jd_to_jd_matching.py
⚠️ test_enhanced_matching.py           - Merge với test_all_features.py
⚠️ test_complete_system.py             - Merge với test_all_features.py
```

### 1.3 Old/Unused Source Files (Cần verify)
```
⚠️ src/database/repository.py          - Old repository? Check if used
⚠️ src/services/matching_service.py    - Old service? Check if used
⚠️ src/embeddings/generator.py         - Old generator? Check if used
⚠️ src/embeddings/field_mapping_embedding.py - Old? Check if used
⚠️ src/embeddings/advanced_field_mapping_embedding.py - Old? Check if used
⚠️ src/embeddings/improved_field_mapping_embedding.py - Old? Check if used
```

### 1.4 Data Sample Files (7 files)
```
✅ data/processed/candidates_sample_10.csv
✅ data/processed/candidates_sample_2.csv
✅ data/processed/candidates_sample_20.csv
✅ data/processed/job_data_sample_10.csv
✅ data/processed/job_data_sample_2.csv
✅ data/processed/job_data_sample_20.csv
✅ data/processed/candidates_dataset.csv (duplicate với raw)
✅ data/processed/job_data.csv (duplicate với raw)
```

### 1.5 Old Index Files (4 files)
```
✅ indices/candidate_index.faiss       - Old single-field index
✅ indices/candidate_index.pkl        - Old single-field index
✅ indices/jd_index.faiss              - Old single-field index
✅ indices/jd_index.pkl               - Old single-field index
```

### 1.6 Log Files (2+ files)
```
⚠️ logs/architecture_evaluation.log   - Log cũ
⚠️ logs/workflow_*.log                - Archive logs > 30 days
```

### 1.7 Documentation - Duplicate/Obsolete (4-5 files)
```
⚠️ docs/AUTO_FAISS_BUILD_SUMMARY.md   - Merge vào FAISS_BUILD_GUIDE.md
⚠️ docs/ENHANCED_SYSTEM_SUMMARY.md     - Merge vào COMPLETE_SYSTEM_OVERVIEW.md
⚠️ docs/SYSTEM_FLOW_SUMMARY.md         - Merge vào SYSTEM_FLOW_QUICK.md
⚠️ docs/PROJECT_WORKFLOW.md            - Merge vào RUN_AND_DEBUG_GUIDE.md
⚠️ docs/FIX_MEMORY_ERROR.md            - Archive hoặc xóa (fix log cũ)
```

---

## 🔀 2. FILES CẦN MERGE

### 2.1 Workflow Scripts → 1 file
**Merge:**
- `run_full_workflow_3_fields.py`
- `run_complete_system.py`
- `run_full_workflow_with_logging.py`

**Thành:** `scripts/run_full_workflow.py`
- Tích hợp tất cả features
- Logging tích hợp
- Options để skip steps

### 2.2 Test Scripts → 1 file
**Merge:**
- `test_all_features.py`
- `test_enhanced_matching.py`
- `test_complete_system.py`

**Thành:** `scripts/test_system.py`
- Test suite đầy đủ
- Có thể chọn test nào để chạy

### 2.3 Documentation Consolidation
**Merge:**
- `SYSTEM_FLOW_QUICK.md` + `SYSTEM_FLOW_SUMMARY.md` → `SYSTEM_FLOW.md`
- `COMPLETE_SYSTEM_OVERVIEW.md` + `ENHANCED_SYSTEM_SUMMARY.md` → `SYSTEM_OVERVIEW.md`
- `PROJECT_WORKFLOW.md` → Merge vào `RUN_AND_DEBUG_GUIDE.md`

---

## 🛠️ 3. CODE QUALITY FIXES

### 3.1 Fixed Issues
- ✅ **t-SNE parameter:** Đã sửa `n_iter` → `max_iter` trong `visualize_embeddings_tsne.py`

### 3.2 Cần Fix
- ⚠️ **Unused imports:** Scan và remove
- ⚠️ **Commented code:** Remove commented-out blocks
- ⚠️ **Magic numbers:** Extract to constants
- ⚠️ **Duplicate code:** Consolidate similar functions

---

## 📁 4. STRUCTURE IMPROVEMENTS

### 4.1 Move Files
```
test_db_connection.py (root) → tests/test_db_connection.py
```

### 4.2 Organize Scripts (Optional)
Tạo subdirectories:
```
scripts/
  workflow/     - run_full_workflow.py, etc.
  test/         - test_system.py, etc.
  analysis/     - analyze_*.py
  utils/        - cleanup, check, etc.
```

### 4.3 Clean Data Directory
- Xóa sample files trong `data/processed/`
- Chỉ giữ: `candidate_processed.csv`, `jd_processed.csv`

---

## 📝 5. SUMMARY

### Tổng số files sẽ xóa: ~30-35 files
- Root: 5 files
- Scripts: 5-7 files (sau khi merge)
- Data: 7 sample files
- Indices: 4 old files
- Logs: 2+ files
- Docs: 4-5 files

### Tổng số files sẽ merge: ~10 files → 3-4 files
- Workflow: 3 → 1
- Test: 3 → 1
- Docs: 4-5 → 2-3

### Code improvements:
- Fix t-SNE parameter ✅
- Remove unused imports
- Remove commented code
- Consolidate constants

---

## ⚠️ XÁC NHẬN CẦN THIẾT

**Bạn có muốn tôi thực hiện các thay đổi này không?**

1. ✅ Xóa files không cần thiết (~30-35 files)
2. ✅ Merge duplicate scripts (~10 → 3-4 files)
3. ✅ Fix code quality issues
4. ✅ Cải thiện project structure
5. ✅ Consolidate documentation

**Vui lòng xác nhận để tiếp tục!**

