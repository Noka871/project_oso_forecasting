@echo off
chcp 65001 >nul
echo ========================================
echo   Запуск OSO Forecasting Application
echo ========================================
echo.

REM Проверка Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Ошибка: Python не найден!
    echo Установите Python 3.8 или выше
    pause
    exit /b 1
)

REM Проверка и установка зависимостей
echo Установка/проверка зависимостей...
pip install -r requirements.txt

if errorlevel 1 (
    echo Ошибка при установке зависимостей!
    pause
    exit /b 1
)

REM Создание необходимых папок
echo Создание структуры папок...
if not exist "data\predictions" mkdir data\predictions
if not exist "trained_models" mkdir trained_models
if not exist "logs" mkdir logs

REM Обновление pip (опционально)
echo.
echo Обновление pip (рекомендуется)...
python -m pip install --upgrade pip

REM Запуск приложения
echo.
echo ========================================
echo   Запуск приложения...
echo ========================================
echo.
python main.py

if errorlevel 1 (
    echo.
    echo Приложение завершилось с ошибкой
    pause
)