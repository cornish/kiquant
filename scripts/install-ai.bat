@echo off
echo ============================================
echo kiQuant AI Detection Setup
echo ============================================
echo.
echo This script installs optional AI dependencies
echo for automatic nucleus detection.
echo.
echo Choose which model(s) to install:
echo   1. CellPose only (~500MB, PyTorch-based) [Recommended]
echo   2. StarDist only (~1GB, TensorFlow-based)
echo   3. Both CellPose and StarDist (~1.5GB)
echo   4. Cancel
echo.

set /p choice="Enter choice (1-4): "

cd /d "%~dp0..\src"

if not exist "venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found.
    echo Please run dev-setup.bat first.
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip first...
python.exe -m pip install --upgrade pip

if "%choice%"=="1" (
    echo.
    echo Installing CellPose...
    pip install "cellpose>=3.0,<4.0"
    goto :done
)

if "%choice%"=="2" (
    echo.
    echo Installing StarDist with TensorFlow...
    pip install tensorflow==2.18.0 stardist
    goto :done
)

if "%choice%"=="3" (
    echo.
    echo Installing CellPose and StarDist...
    pip install tensorflow==2.18.0 "cellpose>=3.0,<4.0" stardist
    goto :done
)

if "%choice%"=="4" (
    echo Cancelled.
    pause
    exit /b 0
)

echo Invalid choice.
pause
exit /b 1

:done
echo.
echo ============================================
echo Installation complete!
echo.
echo Restart kiQuant to use AI detection.
echo ============================================
pause
