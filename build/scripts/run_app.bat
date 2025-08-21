@echo off
REM AI-Enabled GUI for Medical Image Analysis - Quick Start Script
REM This script ensures proper Python environment and starts the secure application

echo ===============================================================
echo    AI-Enabled GUI for Medical Image Analysis
echo    Starting Secure Application...
echo ===============================================================

REM Change to the root application directory (go up from build/scripts)
cd /d "%~dp0..\.."

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.9+ and try again
    pause
    exit /b 1
)

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo 🔧 Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo ⚠️  Virtual environment not found. Using system Python.
)

REM Install requirements if needed
if exist "config\requirements.txt" (
    echo 📦 Checking dependencies...
    pip install -r config\requirements.txt >nul 2>&1
)

REM Start the secure application
echo 🚀 Starting secure medical image analysis application...
python src\apps\app_secure.py

echo.
echo Application stopped. Press any key to exit...
pause >nul
