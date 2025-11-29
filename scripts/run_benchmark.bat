@echo off
REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Run benchmark with parameter variations (5 per model)
python scripts/benchmark_model_variations.py --sample-size 50 --use-param-variations

pause

