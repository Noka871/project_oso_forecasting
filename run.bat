@echo off
chcp 65001
echo ========================================
echo   🌍 OSO Forecasting - Modern Interface
echo ========================================
echo.

echo Проверка Python...
python --version > nul 2>&1
if errorlevel 1 (
    echo ❌ ОШИБКА: Python не установлен
    echo Установите Python 3.8+ с python.org
    pause
    exit /b 1
)

echo Установка/проверка зависимостей...
pip install -r requirements.txt

echo Запуск современного интерфейса...
python main.py

pause