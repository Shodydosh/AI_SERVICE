@echo off
REM Setup virtual environment for Windows CMD

echo Setting up virtual environment...

REM Create venv
python -m venv venv

REM Activate
call venv\Scripts\activate.bat

REM Upgrade pip and setuptools
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Virtual environment setup complete!
echo.
echo To activate: venv\Scripts\activate.bat
echo To deactivate: deactivate

