@echo off
echo ============================================
echo KiNet Trainer - Development Setup
echo ============================================
echo.

cd /d "%~dp0\..\src"

if not exist "venv" (
    echo Creating virtual environment...

    REM Try Python 3.12 or 3.11
    where py >nul 2>&1
    if %errorlevel%==0 (
        py -3.12 --version >nul 2>&1
        if %errorlevel%==0 (
            echo Using Python 3.12...
            py -3.12 -m venv venv
        ) else (
            py -3.11 --version >nul 2>&1
            if %errorlevel%==0 (
                echo Using Python 3.11...
                py -3.11 -m venv venv
            ) else (
                echo Python 3.11/3.12 not found, using default python...
                python -m venv venv
            )
        )
    ) else (
        echo Python launcher not found, using default python...
        python -m venv venv
    )
) else (
    echo Virtual environment already exists.
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Upgrading pip...
python.exe -m pip install --upgrade pip

echo.
echo Installing dependencies (CPU PyTorch)...
pip install -r ..\..\kinet-trainer\requirements.txt

echo.
echo ============================================
echo Setup complete!
echo.
echo To start the annotation GUI:
echo   scripts\run.bat
echo.
echo For GPU training, run:
echo   scripts\install-gpu.bat
echo ============================================
pause
