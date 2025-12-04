@echo off
echo ================================================================================
echo POSTGRESQL WORKFLOW: 3-FIELD MULTI-FILTER MATCHING
echo ================================================================================
echo.

echo [Step 0/5] Checking PostgreSQL setup...
python scripts\check_postgresql_setup.py
if errorlevel 1 (
    echo.
    echo ERROR: PostgreSQL setup check failed!
    echo Please fix the issues above before continuing.
    pause
    exit /b 1
)
echo.

echo [Step 1/5] Initializing database tables...
python scripts\init_multi_field_tables.py
if errorlevel 1 (
    echo ERROR: Failed to initialize database tables
    pause
    exit /b 1
)
echo.

echo [Step 2/5] Processing job descriptions (10 records for testing)...
python scripts\process_multi_field_embeddings.py --jd-file data\raw\job_data.csv --batch-size 10
if errorlevel 1 (
    echo WARNING: Job processing failed, but continuing...
)
echo.

echo [Step 3/5] Processing candidates (10 records for testing)...
python scripts\process_multi_field_embeddings.py --candidate-file data\raw\candidates_dataset.csv --batch-size 10
if errorlevel 1 (
    echo WARNING: Candidate processing failed, but continuing...
)
echo.

echo [Step 4/5] Testing matching service...
python scripts\test_multi_filter_matching.py --candidate-id "15001" --top-k 5
if errorlevel 1 (
    echo WARNING: Matching test failed
)
echo.

echo ================================================================================
echo WORKFLOW COMPLETED
echo ================================================================================
echo.
echo Next steps:
echo - Check logs in logs/ directory
echo - Verify data in PostgreSQL database
echo - Process full dataset with larger batch size
echo.
pause



