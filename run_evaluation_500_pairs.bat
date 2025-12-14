@echo off
echo ========================================
echo   EVALUATION 500 PAIRS - ENHANCED
echo ========================================
echo.

REM Activate virtual environment if exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Run evaluation (simple version - no model loading needed)
python scripts/evaluate_500_pairs_simple.py ^
    --ground-truth-csv ground_truth_500_pairs.csv ^
    --output-html reports/evaluation_500_pairs_report.html ^
    --output-json reports/evaluation_500_pairs_results.json

echo.
echo ========================================
echo   EVALUATION COMPLETED
echo ========================================
echo.
echo HTML Report: reports\evaluation_500_pairs_report.html
echo JSON Results: reports\evaluation_500_pairs_results.json
echo.
pause

