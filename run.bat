@echo off
chcp 65001 >nul

echo ========================================
echo   OSO Forecasting Application
echo ========================================

python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found
    pause
    exit /b 1
)

echo Installing dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo Error installing dependencies
    pause
    exit /b 1
)

echo Creating folders...
if not exist "data\predictions" mkdir data\predictions
if not exist "trained_models" mkdir trained_models
if not exist "logs" mkdir logs

echo.
echo ========================================
echo   Starting application...
echo ========================================
echo.

python main.py

if errorlevel 1 (
    echo Application ended with error
    pause
)