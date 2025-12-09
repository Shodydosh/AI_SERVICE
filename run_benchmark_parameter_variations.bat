@echo off
echo ================================================================================
echo BENCHMARK PARAMETER VARIATIONS
echo ================================================================================
echo.

echo Testing first 5 variations (Model 1: SimCSE_Vietnamese)...
echo Sample size: 20 records
echo.

python scripts\benchmark_from_csv.py --candidate-file data\raw\candidates_dataset.csv --jd-file data\raw\job_data.csv --sample-size 20 --limit 5

if errorlevel 1 (
    echo.
    echo ERROR: Benchmark failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo BENCHMARK COMPLETED
echo ================================================================================
echo.
echo Check results in:
echo   - reports\benchmark_csv\benchmark_csv_results_*.json
echo   - reports\benchmark_csv\benchmark_csv_results_*.csv
echo   - reports\benchmark_csv\logs\benchmark_csv_*.log
echo.
pause




