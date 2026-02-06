@echo off
cd /d "%~dp0\..\src"

if not exist "venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found.
    echo Please run scripts\dev-setup.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

if "%~1"=="" (
    echo KiNet Evaluation Script
    echo.
    echo Usage: evaluate.bat --model ^<path^> --data-dir ^<path^> [options]
    echo.
    echo Options:
    echo   --model              Path to model weights (required)
    echo   --data-dir           Data directory in KiNet format (required)
    echo   --split              Split to evaluate: train or val (default: val)
    echo   --threshold          Detection threshold (default: 0.3)
    echo   --min-distance       Min peak distance in pixels (default: 5)
    echo   --max-match-distance Max matching distance in pixels (default: 10)
    echo   --output             Output JSON file path
    echo.
    pause
    exit /b 0
)

python evaluate.py %*
pause
