# PowerShell script to run benchmark
# Activate virtual environment if it exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
} elseif (Test-Path ".venv\Scripts\Activate.ps1") {
    & ".venv\Scripts\Activate.ps1"
}

# Run benchmark with parameter variations (5 per model)
python scripts/benchmark_model_variations.py --sample-size 50 --use-param-variations

Read-Host "Press Enter to exit"

