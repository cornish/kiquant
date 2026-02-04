@echo off
echo ============================================
echo kiQuant AI Detection Setup (GPU)
echo ============================================
echo.
echo This script installs AI dependencies with
echo NVIDIA GPU acceleration (CUDA).
echo.
echo Requirements:
echo   - NVIDIA GPU with CUDA support
echo   - NVIDIA drivers installed
echo   - Python 3.12 or earlier
echo.
echo Choose which model(s) to install:
echo   1. CellPose only (PyTorch + CUDA)
echo   2. StarDist only (TensorFlow + CUDA)
echo   3. Both CellPose and StarDist
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

echo Upgrading pip...
python.exe -m pip install --upgrade pip

if "%choice%"=="1" (
    echo.
    echo Removing CPU PyTorch if present...
    pip uninstall -y torch torchvision 2>nul
    echo.
    echo Installing PyTorch with CUDA 11.8 support...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    echo.
    echo Installing CellPose...
    pip install "cellpose>=3.0,<4.0"
    goto :done
)

if "%choice%"=="2" (
    echo.
    echo Installing TensorFlow with CUDA support...
    pip install tensorflow[and-cuda]==2.18.0
    echo.
    echo Installing StarDist...
    pip install stardist
    goto :done
)

if "%choice%"=="3" (
    echo.
    echo Removing CPU PyTorch if present...
    pip uninstall -y torch torchvision 2>nul
    echo.
    echo Installing PyTorch with CUDA 11.8 support...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    echo.
    echo Installing CellPose...
    pip install "cellpose>=3.0,<4.0"
    echo.
    echo Installing TensorFlow with CUDA support...
    pip install tensorflow[and-cuda]==2.18.0
    echo.
    echo Installing StarDist...
    pip install stardist
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
echo Verifying GPU detection...
echo.
python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}')" 2>nul
python -c "import tensorflow as tf; print(f'TensorFlow GPUs: {len(tf.config.list_physical_devices(\"GPU\"))}')" 2>nul
echo.
echo If GPUs show as 0 or False, ensure NVIDIA
echo drivers and CUDA toolkit are installed.
echo.
echo Restart kiQuant to use AI detection.
echo ============================================
pause
