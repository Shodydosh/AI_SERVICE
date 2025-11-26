#!/bin/bash
# Setup virtual environment for macOS/Linux

echo "Setting up virtual environment..."

# Check if already in venv
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Already in virtual environment. Deactivating first..."
    deactivate
fi

# Create venv
python3 -m venv venv

# Activate
source venv/bin/activate

# Upgrade pip and setuptools
echo "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✓ Virtual environment setup complete!"
echo ""
echo "To activate: source venv/bin/activate"
echo "To deactivate: deactivate"

