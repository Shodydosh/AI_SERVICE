# PowerShell script to run benchmark with detailed logging
# Activate virtual environment if it exists
$venvActivated = $false

if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activating venv..." -ForegroundColor Green
    & "venv\Scripts\Activate.ps1"
    $venvActivated = $true
} elseif (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Activating .venv..." -ForegroundColor Green
    & ".venv\Scripts\Activate.ps1"
    $venvActivated = $true
} else {
    Write-Host "No virtual environment found. Please activate manually or install dependencies globally." -ForegroundColor Yellow
}

Write-Host "`nStarting benchmark with detailed logging..." -ForegroundColor Cyan
Write-Host "This will benchmark model variations and log everything for report generation." -ForegroundColor Cyan
Write-Host ""

# Run benchmark with logging
# --limit 10 means test first 10 variations (can be increased)
python scripts/benchmark_with_logging.py --sample-size 30 --limit 10

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nBenchmark failed. Check logs in reports/benchmark_variations/logs/" -ForegroundColor Red
} else {
    Write-Host "`nBenchmark completed successfully!" -ForegroundColor Green
    Write-Host "Check results in:" -ForegroundColor Green
    Write-Host "  - reports/benchmark_variations/benchmark_results_*.json" -ForegroundColor Yellow
    Write-Host "  - reports/benchmark_variations/benchmark_results_*.csv" -ForegroundColor Yellow
    Write-Host "  - reports/benchmark_variations/benchmark_report_*.md" -ForegroundColor Yellow
    Write-Host "  - reports/benchmark_variations/logs/benchmark_*.log" -ForegroundColor Yellow
}

Read-Host "`nPress Enter to exit"

