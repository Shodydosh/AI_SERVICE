# Setup virtual environment for Windows PowerShell

Write-Host "Setting up virtual environment..." -ForegroundColor Green

# Check if already in venv
if ($env:VIRTUAL_ENV) {
    Write-Host "Already in virtual environment. Deactivating first..." -ForegroundColor Yellow
    if (Get-Command deactivate -ErrorAction SilentlyContinue) {
        deactivate
    }
}

# Create venv
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1

# Upgrade pip and setuptools
Write-Host "Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Green
pip install -r requirements.txt

Write-Host ""
Write-Host "Virtual environment setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To activate: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "To deactivate: deactivate" -ForegroundColor Yellow

