"""Script để xóa các file không liên quan đến workflow chính."""
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Core workflow files - GIỮ LẠI
CORE_FILES = {
    # Database & Setup
    'init_multi_field_tables.py',
    'check_postgresql_setup.py',
    
    # Data Processing
    'filter_data_with_skills.py',
    'process_multi_field_embeddings.py',
    'process_all_raw_data.py',
    'process_to_processed.py',
    
    # Workflow
    'run_full_workflow_3_fields.py',
    'run_full_optimization_benchmark.py',
    
    # Testing & Matching
    'test_multi_filter_matching.py',
    'test_best_variation_samples.py',
    'test_all_features.py',
    'test_enhanced_matching.py',
    
    # Analysis (cần cho benchmark)
    'analyze_benchmark_results.py',
    'analyze_all_variations.py',
    'analyze_improvements.py',
    'monitor_benchmark_progress.py',
    'update_comparison_csv.py',
    
    # Utilities
    'check_database_status.py',
    'database_optimization.py',
    
    # Enhanced services test
    'test_all_features.py',
    'test_enhanced_matching.py',
}

# Files cần xóa - không liên quan đến workflow
FILES_TO_DELETE = [
    # Old test scripts
    'test_5_candidates_detailed.py',
    'test_5_candidates_sample.py',
    'test_5_candidates_with_logging.py',
    'test_90_percent_similarity.py',
    'test_and_log.py',
    'test_benchmark_csv_simple.py',
    'test_benchmark_csv.py',
    'test_benchmark.py',
    'test_cosine_similarity.py',
    'test_data_check.py',
    'test_embeddings_sample.py',
    'test_evaluation_simple.py',
    'test_final.py',
    'test_improved_similarity.py',
    'test_multi_field_complete.py',
    'test_multi_field_filter.py',
    'test_multi_field_final.py',
    'test_multi_field_output.py',
    'test_multi_field_simple.py',
    'test_multi_field_with_logging.py',
    'test_simcse_vietnamese_model.py',
    'test_simcse_vietnamese.py',
    'test_system_simple.py',
    'test_system.py',
    'test_vietnamese_embedding_quality.py',
    'test_vietnamese_model.py',
    'test_weighted_embeddings.py',
    
    # Old benchmark scripts
    'run_full_benchmark_50_variations.py',
    'benchmark_field_mapping.py',
    'benchmark_model_variations.py',
    'benchmark_models.py',
    'benchmark_system.py',
    'benchmark_with_logging.py',
    'auto_run_benchmark.py',
    'run_benchmark_minimal.py',
    'run_missing_variations.py',
    
    # Old check/verify scripts
    'check_all_embeddings_quality.py',
    'check_benchmark_progress.py',
    'check_benchmark_status.py',
    'check_candidate_data.py',
    'check_candidate_details.py',
    'check_candidate_fields.py',
    'check_consistency_data.py',
    'check_database_data.py',
    'check_db_connection.py',
    'check_embedding_progress.py',
    'check_embedding_quality_batch.py',
    'check_embedding_quality.py',
    'check_embeddings_saved.py',
    'check_jd_embeddings_quality.py',
    'check_job_fields.py',
    'check_logs_for_errors.py',
    'check_memory.py',
    'check_raw_data.py',
    'clean_data.py',
    'clean_nan_values.py',
    'clear_embeddings.py',
    'compare_candidate_embeddings.py',
    'compare_embedding_methods.py',
    'compare_embedding_models.py',
    'debug_consistency_tests.py',
    'debug_model_loading.py',
    'find_candidates_with_data.py',
    'full_system_check.py',
    'quick_check_status.py',
    'quick_test_workflow.py',
    'verify_candidate_data.py',
    'validate_data.py',
    'view_benchmark_log.py',
    'view_status.py',
    
    # Old analysis scripts
    'analyze_matching_quality.py',
    
    # Old evaluation scripts
    'evaluate_architecture_3_levels.py',
    'evaluate_embeddings_research.py',
    'evaluate_system_comprehensive.py',
    
    # Old processing scripts
    'process_datasets.py',
    'process_multi_field_datasets.py',
    'generate_embeddings_from_processed.py',
    'generate_embeddings.py',
    'generate_field_mapping_embeddings.py',
    'generate_processed_recommendations.py',
    'reembed_all_data.py',
    'rerun_system.py',
    
    # Old utility scripts
    'data_pipeline.py',
    'preprocess_data.py',
    'prepare_evaluation_datasets.py',
    'select_embedding_model.py',
    'show_updated_results.py',
    'optimize_system.py',
    
    # Batch/PowerShell scripts (giữ lại setup_venv)
    'check_benchmark_status.bat',
    'check_benchmark_status.ps1',
    'run_architecture_evaluation.ps1',
    'run_benchmark_auto.bat',
    'run_benchmark_auto.ps1',
    'run_benchmark_csv.ps1',
    'run_benchmark_logging.ps1',
    'run_benchmark.bat',
    'run_benchmark.ps1',
    'run_full_benchmark_50.bat',
    'run_full_benchmark_50.ps1',
    'watch_benchmark.ps1',
    
    # Old scripts
    'benchmark_from_csv.py',  # Đã có run_full_optimization_benchmark.py
    'match_candidate_to_jobs.py',
    'migrate_db_schema.py',
    'manage_faiss.py',
    'download_datasets.py',
    'init_db.py',  # Đã có init_multi_field_tables.py
]


def cleanup_scripts():
    """Xóa các file không cần thiết trong scripts/."""
    scripts_dir = Path(__file__).parent
    deleted_count = 0
    skipped_count = 0
    
    logger.info("=" * 80)
    logger.info("CLEANUP UNUSED FILES")
    logger.info("=" * 80)
    logger.info(f"Scripts directory: {scripts_dir}")
    logger.info("")
    
    for filename in FILES_TO_DELETE:
        file_path = scripts_dir / filename
        
        if file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"✓ Deleted: {filename}")
                deleted_count += 1
            except Exception as e:
                logger.error(f"✗ Error deleting {filename}: {e}")
                skipped_count += 1
        else:
            logger.debug(f"  Not found: {filename}")
            skipped_count += 1
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"SUMMARY: Deleted {deleted_count} files, Skipped {skipped_count} files")
    logger.info("=" * 80)
    
    # List remaining files
    remaining = [f.name for f in scripts_dir.glob("*.py") if f.name not in CORE_FILES and f.name != "cleanup_unused_files.py"]
    if remaining:
        logger.info("")
        logger.info("Remaining files (not in CORE_FILES list):")
        for f in sorted(remaining):
            logger.info(f"  - {f}")


def cleanup_reports():
    """Xóa các log files cũ trong reports/."""
    reports_dir = Path(__file__).parent.parent / "reports"
    
    if not reports_dir.exists():
        return
    
    logger.info("")
    logger.info("Cleaning up old log files in reports/...")
    
    # Xóa log files cũ
    log_dirs = [
        reports_dir / "benchmark_csv" / "logs",
        reports_dir / "benchmark_variations" / "logs",
    ]
    
    deleted_logs = 0
    for log_dir in log_dirs:
        if log_dir.exists():
            for log_file in log_dir.glob("*.log"):
                try:
                    log_file.unlink()
                    deleted_logs += 1
                    logger.info(f"✓ Deleted log: {log_file.name}")
                except Exception as e:
                    logger.error(f"✗ Error deleting {log_file}: {e}")
    
    if deleted_logs > 0:
        logger.info(f"Deleted {deleted_logs} log files")


if __name__ == "__main__":
    cleanup_scripts()
    cleanup_reports()
    logger.info("")
    logger.info("✅ Cleanup completed!")

