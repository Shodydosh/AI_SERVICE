# MỤC LỤC BÁO CÁO HỆ THỐNG AI SERVICE

## 1. TỔNG QUAN HỆ THỐNG

1.1. Giới thiệu hệ thống AI Job Recommendation Service
1.2. Kiến trúc tổng quan (Two-Tower Architecture)
1.3. Luồng hoạt động chính của hệ thống
1.4. Công nghệ và công cụ sử dụng

## 2. KIẾN TRÚC HỆ THỐNG

2.1. Kiến trúc Two-Tower Architecture
    2.1.1. Job Tower Encoder
    2.1.2. Candidate Tower Encoder
    2.1.3. Multi-field Embeddings (3 embeddings per record)
2.2. Cấu trúc phân lớp (Layered Architecture)
    2.2.1. API Layer (FastAPI)
    2.2.2. Service Layer (Business Logic)
    2.2.3. Embedding Layer (Sentence Transformers)
    2.2.4. Database Layer (PostgreSQL)
    2.2.5. Vector Search Layer (FAISS)
2.3. Entry Point và Application Bootstrap
    2.3.1. Main entry point (main.py)
    2.3.2. FastAPI application initialization
    2.3.3. Configuration management

## 3. MÔ HÌNH VÀ EMBEDDINGS

3.1. Two-Tower Model Architecture
    3.1.1. TwoTowerModel class (src/models/two_tower_model.py)
    3.1.2. CandidateTower neural network
    3.1.3. JobTower neural network
    3.1.4. Model parameters và hyperparameters
3.2. Embedding Encoders
    3.2.1. JobTowerEncoder (src/embeddings/job_tower_encoder.py)
        3.2.1.1. Title embedding generation
        3.2.1.2. Skills embedding generation
        3.2.1.3. Requirement embedding generation
        3.2.1.4. Vietnamese text preprocessing
    3.2.2. CandidateTowerEncoder (src/embeddings/candidate_tower_encoder.py)
        3.2.2.1. Title embedding generation
        3.2.2.2. Skills embedding generation
        3.2.2.3. Experience embedding generation
        3.2.2.4. Vietnamese text preprocessing
3.3. Embedding Model Selection
    3.3.1. Model selector (src/embeddings/model_selector.py)
    3.3.2. Vietnamese SimCSE model (VoVanPhuc/sup-SimCSE-VietNamese-phobert-base)
    3.3.3. Embedding dimension (768 dimensions)
3.4. Embedding Service
    3.4.1. OptimizedEmbeddingService (src/services/embedding_service.py)
    3.4.2. Batch processing
    3.4.3. Embedding caching mechanism
    3.4.4. Non-blocking realtime queries

## 4. HỆ THỐNG MATCHING VÀ TÌM KIẾM

4.1. Two-Tower Matching Service
    4.1.1. TwoTowerMatchingService (src/services/two_tower_matching_service.py)
    4.1.2. Tìm jobs cho candidate (find_jobs_for_candidate)
    4.1.3. Tìm candidates cho job (find_candidates_for_job)
    4.1.4. Model loading và inference
4.2. Vector Search với FAISS
    4.2.1. TwoTowerFAISSManager (src/vector_search/two_tower_faiss_manager.py)
    4.2.2. 6 FAISS indices riêng biệt
        4.2.2.1. Job title index
        4.2.2.2. Job skills index
        4.2.2.3. Job requirement index
        4.2.2.4. Candidate title index
        4.2.2.5. Candidate skills index
        4.2.2.6. Candidate experience index
    4.2.3. HNSW index configuration (M=32, ef_search=128)
    4.2.4. Index building và management
    4.2.5. Similarity search operations
4.3. Rule-based Matching
    4.3.1. RuleMatcher (src/utils/rule_matcher.py)
    4.3.2. Title similarity matching
    4.3.3. Skills matching (exact, fuzzy, category-level)
    4.3.4. Experience matching
    4.3.5. Rule scoring và validation
    4.3.6. Vietnamese text handling

## 5. XỬ LÝ DỮ LIỆU

5.1. Data Processing Layer
    5.1.1. JDProcessor (src/data_processing/jd_processor.py)
        5.1.1.1. Job description dataset processing
        5.1.1.2. Data validation
        5.1.1.3. Field extraction
    5.1.2. CandidateProcessor (src/data_processing/candidate_processor.py)
        5.1.2.1. Candidate dataset processing
        5.1.2.2. Data validation
        5.1.2.3. Field extraction
5.2. Data Preprocessing Utilities
    5.2.1. DataValidator (src/utils/data_validator.py)
    5.2.2. DataPreprocessor (src/utils/data_preprocessor.py)
    5.2.3. CleanData (src/utils/clean_data.py)
    5.2.4. ThreeFieldExtractor (src/utils/three_field_extractor.py)
5.3. Data Quality và Validation
    5.3.1. Raw data quality checking
    5.3.2. Data validation scripts
    5.3.3. Preprocessing pipeline
    5.3.4. Report generation

## 6. LỚP DATABASE

6.1. Database Connection và Configuration
    6.1.1. Database connection (src/database/connection.py)
    6.1.2. SQLAlchemy engine và session management
    6.1.3. Connection pooling
6.2. Database Models
    6.2.1. Legacy models
        6.2.1.1. JobDescriptionEmbedding
        6.2.1.2. CandidateEmbedding
    6.2.2. Multi-field embedding models
        6.2.2.1. JobDescriptionMultiEmbedding
        6.2.2.2. CandidateMultiEmbedding
    6.2.3. Two-Tower models
        6.2.3.1. JobDescriptionTwoTower
        6.2.3.2. CandidateTwoTower
    6.2.4. Recommendation models
        6.2.4.1. ProcessedCandidateRecommendation
        6.2.4.2. Explainability fields
    6.2.5. Evaluation models (src/database/evaluation_models.py)
6.3. Database Repositories
    6.3.1. MultiFieldEmbeddingRepository (src/database/multi_field_repository.py)
    6.3.2. TwoTowerRepository (src/database/two_tower_repository.py)
    6.3.3. CRUD operations
    6.3.4. Query optimization
6.4. Database Migrations
    6.4.1. Alembic configuration
    6.4.2. Schema migrations
    6.4.3. Data migrations

## 7. API LAYER

7.1. FastAPI Application
    7.1.1. Main application (src/api/main.py)
    7.1.2. Application configuration
    7.1.3. CORS middleware
    7.1.4. Static files serving
7.2. API Routes
    7.2.1. Two-Tower routes (src/api/two_tower_routes.py)
        7.2.1.1. POST /api/v2/search/jobs
        7.2.1.2. POST /api/v2/search/candidates
        7.2.1.3. POST /api/v2/index/job
        7.2.1.4. POST /api/v2/index/candidate
        7.2.1.5. POST /api/v2/reindex
        7.2.1.6. GET /api/v2/health
    7.2.2. Request/Response handling
    7.2.3. Error handling
7.3. API Schemas
    7.3.1. Two-Tower schemas (src/api/two_tower_schemas.py)
    7.3.2. Request models (Pydantic)
    7.3.3. Response models (Pydantic)
    7.3.4. Data validation

## 8. SERVICES VÀ BUSINESS LOGIC

8.1. Matching Services
    8.1.1. TwoTowerMatchingService
    8.1.2. Matching pipeline (3-stage)
    8.1.3. Score calculation và ranking
8.2. Embedding Services
    8.2.1. OptimizedEmbeddingService
    8.2.2. EmbeddingCacheManager (src/services/embedding_cache_manager.py)
    8.2.3. Cache TTL và invalidation
8.3. Scheduler Services
    8.3.1. EmbeddingScheduler (src/services/embedding_scheduler.py)
    8.3.2. Scheduled embedding updates (12-hour cycle)
    8.3.3. Background tasks

## 9. UTILITIES VÀ HELPERS

9.1. Text Processing Utilities
    9.1.1. VietnameseTranslator (src/utils/vietnamese_translator.py)
    9.1.2. TextEnhancer (src/utils/text_enhancer.py)
    9.1.3. Vietnamese tokenization (pyvi)
9.2. Embedding Utilities
    9.2.1. EmbeddingLoader (src/utils/embedding_loader.py)
    9.2.2. Model loading và caching
9.3. Explanation Utilities
    9.3.1. ExplanationGenerator (src/utils/explanation_generator.py)
        9.3.1.1. Level 1: Rule matching explanation
        9.3.1.2. Level 2: Embedding similarity explanation
        9.3.1.3. Level 3: Humanized explanation
        9.3.1.4. Level 4: Counterfactual explanation
        9.3.1.5. Level 5: Confidence score calculation
    9.3.2. ExplanationStorage (src/utils/explanation_storage.py)
9.4. Data Utilities
    9.4.1. ColumnMapper (src/utils/column_mapper.py)
    9.4.2. ReportGenerator (src/utils/report_generator.py)
9.5. Logging Utilities
    9.5.1. LoggingUTF8 (src/utils/logging_utf8.py)
    9.5.2. Logging configuration

## 10. TRAINING VÀ EVALUATION

10.1. Training Pipeline
    10.1.1. TrainingPipeline (src/models/training_pipeline.py)
    10.1.2. GroundTruthDataset
    10.1.3. Training loop
    10.1.4. Model checkpointing
    10.1.5. Loss functions
10.2. Evaluation Metrics
    10.2.1. TwoTowerEvaluator (src/models/evaluation_metrics.py)
    10.2.2. Accuracy metrics
    10.2.3. Recall@K
    10.2.4. Precision metrics
10.3. Ground Truth Building
    10.3.1. GroundTruthBuilder (src/models/ground_truth_builder.py)
    10.3.2. Ground truth dataset generation
10.4. Training Scripts
    10.4.1. train_two_tower.py (scripts/train_two_tower.py)
    10.4.2. Training data preparation
    10.4.3. Model training execution

## 11. TESTING VÀ BENCHMARKING

11.1. Evaluation Scripts
    11.1.1. evaluate_two_tower.py
    11.1.2. evaluate_two_tower_detailed.py
    11.1.3. evaluate_two_tower_simple.py
    11.1.4. compare_models.py
11.2. Testing Scripts
    11.2.1. test_two_tower_precomputed.py
    11.2.2. test_two_tower_standalone.py
    11.2.3. test_two_tower_with_vietnamese.py
    11.2.4. test_complete_system.py
    11.2.5. test_enhanced_matching.py
    11.2.6. test_multi_filter_matching.py
    11.2.7. test_50_candidates.py
    11.2.8. test_all_features.py
11.3. Benchmark Scripts
    11.3.1. run_full_optimization_benchmark.py
    11.3.2. analyze_benchmark_results.py
    11.3.3. analyze_all_variations.py
    11.3.4. analyze_improvements.py
    11.3.5. monitor_benchmark_progress.py
11.4. Unit Tests
    11.4.1. test_embeddings.py
    11.4.2. test_rule_matcher.py
    11.4.3. test_debug_samples.py

## 12. SCRIPTS VÀ TOOLS

12.1. Data Processing Scripts
    12.1.1. process_all_raw_data.py
    12.1.2. process_to_processed.py
    12.1.3. process_multi_field_embeddings.py
    12.1.4. filter_data_with_skills.py
12.2. Database Scripts
    12.2.1. init_multi_field_tables.py
    12.2.2. migrate_to_two_tower_schema.py
    12.2.3. add_explanation_fields_migration.py
    12.2.4. create_embedding_timestamp_migration.py
    12.2.5. check_database_status.py
    12.2.6. debug_database_connection.py
    12.2.7. database_optimization.py
12.3. Indexing Scripts
    12.3.1. batch_reindex_two_tower.py
    12.3.2. build_multi_field_faiss.py
    12.3.3. incremental_upsert_two_tower.py
12.4. Workflow Scripts
    12.4.1. run_complete_system.py
    12.4.2. run_full_workflow_3_fields.py
    12.4.3. run_full_workflow_with_logging.py
    12.4.4. rerun_pipeline_with_translation.py
12.5. Recommendation Scripts
    12.5.1. recommend_jobs_for_candidates.py
    12.5.2. show_10_samples_with_recommendations.py
    12.5.3. print_recommendations_details.py
12.6. Ground Truth Scripts
    12.6.1. generate_ground_truth_500_pairs.py
12.7. Visualization Scripts
    12.7.1. visualize_embeddings_tsne.py
    12.7.2. visualize_embeddings_tsne_test.py
    12.7.3. visualize_tsne_production.py
    12.7.4. visualize_tsne_simple.py
12.8. Utility Scripts
    12.8.1. check_title_similarity.py
    12.8.2. check_postgresql_setup.py
    12.8.3. cleanup_unused_files.py
    12.8.4. update_comparison_csv.py
    12.8.5. print_full_test_results.py
12.9. Scheduler Scripts
    12.9.1. run_embedding_scheduler.py
12.10. Candidate Creation Scripts
    12.10.1. test_candidate_creation.py
    12.10.2. test_candidate_creation_direct.py

## 13. TWO-TOWER MODULE

13.1. Two-Tower Implementation
    13.1.1. Model definition (two_tower/model.py)
    13.1.2. Data handling (two_tower/data.py)
    13.1.3. Loss functions (two_tower/loss.py)
    13.1.4. Training utilities (two_tower/train.py, train_improved.py)
    13.1.5. Inference utilities (two_tower/inference.py)
    13.1.6. Evaluation utilities (two_tower/evaluate.py)
    13.1.7. General utilities (two_tower/utils.py)
13.2. CV-Job Matching
    13.2.1. CV-Job matcher (two_tower/cv_job_matcher.py)
    13.2.2. Training data creation (two_tower/create_training_data.py)
    13.2.3. Run CV-Job matching (two_tower/run_cv_job_matching.py)
13.3. Testing và Demo
    13.3.1. Basic tests (two_tower/test_basic.py)
    13.3.2. Improved tests (two_tower/test_improved.py)
    13.3.3. Two-tower tests (two_tower/test_two_tower.py)
    13.3.4. Demo script (two_tower/demo.py)
    13.3.5. Example inference (two_tower/example_inference.py)
13.4. Export và Optimization
    13.4.1. ONNX export (two_tower/export_onnx.py)
13.5. Documentation
    13.5.1. README files
    13.5.2. Quick start guides
    13.5.3. Test results documentation

## 14. CẤU HÌNH VÀ INFRASTRUCTURE

14.1. Configuration Management
    14.1.1. Settings (config/settings.py)
    14.1.2. Environment variables
    14.1.3. Database configuration
    14.1.4. Embedding model configuration
    14.1.5. API configuration
14.2. Logging Configuration
    14.2.1. Log levels
    14.2.2. Log formats
    14.2.3. Log files
14.3. Database Setup
    14.3.1. PostgreSQL configuration
    14.3.2. pgvector extension
    14.3.3. Database initialization
14.4. FAISS Configuration
    14.4.1. Index types
    14.4.2. Index parameters
    14.4.3. Index storage

## 15. DOCUMENTATION

15.1. System Documentation
    15.1.1. System flow documentation
    15.1.2. Architecture documentation
    15.1.3. Workflow guides
15.2. Feature Documentation
    15.2.1. Two-Tower architecture guides
    15.2.2. Embedding system documentation
    15.2.3. Rule matching documentation
    15.2.4. Explainability system guide
15.3. Operational Documentation
    15.3.1. Setup guides
    15.3.2. Running guides
    15.3.3. Debugging guides
    15.3.4. PostgreSQL workflow guide
    15.3.5. FAISS build guide
15.4. API Documentation
    15.4.1. API candidate creation guide
    15.4.2. Swagger/OpenAPI documentation
15.5. Benchmark Documentation
    15.5.1. Benchmark parameter variations
    15.5.2. Run 50 variations benchmark guide
15.6. Maintenance Documentation
    15.6.1. Maintenance analysis
    15.6.2. Maintenance reports

## 16. DATA VÀ OUTPUTS

16.1. Data Structure
    16.1.1. Raw data (data/raw/)
    16.1.2. Processed data (data/processed/)
    16.1.3. Filtered data (data/filtered/)
    16.1.4. Sample data (data/sample/)
    16.1.5. Training data (JSON format)
16.2. Model Outputs
    16.2.1. Trained models (outputs/, outputs_improved/, outputs_vo/)
    16.2.2. Model checkpoints
    16.2.3. Job embeddings
16.3. FAISS Indices
    16.3.1. Candidate indices (indices/candidate_index.*)
    16.3.2. JD indices (indices/jd_index.*)
    16.3.3. Multi-field indices (indices/multi_field/)
16.4. Reports và Logs
    16.4.1. Evaluation results
    16.4.2. Benchmark reports
    16.4.3. System logs
    16.4.4. Recommendation outputs

## 17. DEPENDENCIES VÀ REQUIREMENTS

17.1. Python Dependencies
    17.1.1. Core dependencies (FastAPI, SQLAlchemy, etc.)
    17.1.2. ML dependencies (PyTorch, Sentence Transformers, etc.)
    17.1.3. Database dependencies (psycopg2, pgvector)
    17.1.4. Vector search dependencies (FAISS)
    17.1.5. Vietnamese NLP dependencies (pyvi)
17.2. System Requirements
    17.2.1. Python version
    17.2.2. PostgreSQL version
    17.2.3. Operating system compatibility
17.3. Configuration Files
    17.3.1. requirements.txt
    17.3.2. pyproject.toml
    17.3.3. alembic.ini




