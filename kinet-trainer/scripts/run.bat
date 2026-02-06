@echo off
cd /d "%~dp0\..\src"

if not exist "venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found.
    echo Please run scripts\dev-setup.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
python main.py
