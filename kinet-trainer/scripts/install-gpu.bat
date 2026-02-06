@echo off
echo ============================================
echo KiNet Trainer - GPU Setup (CUDA)
echo ============================================
echo.
echo This replaces CPU PyTorch with the CUDA version
echo for GPU-accelerated training.
echo.
echo Requirements:
echo   - NVIDIA GPU with CUDA support
echo   - NVIDIA drivers installed
echo.

cd /d "%~dp0\..\src"

if not exist "venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found.
    echo Please run dev-setup.bat first.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Removing CPU PyTorch...
pip uninstall -y torch torchvision 2>nul

echo.
echo Installing PyTorch with CUDA 11.8 support...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo.
echo Verifying GPU detection...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo.
echo ============================================
echo GPU setup complete!
echo.
echo If CUDA shows as False, ensure NVIDIA
echo drivers and CUDA toolkit are installed.
echo ============================================
pause
