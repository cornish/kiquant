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
    echo KiNet Training Script
    echo.
    echo Usage: train.bat --data-dir ^<path^> [options]
    echo.
    echo By default, fine-tunes from kiQuant's cached model weights
    echo (~/.kiquant/models/ki67net-best.pth) if available.
    echo.
    echo Options:
    echo   --data-dir        Training data directory (required)
    echo   --output-dir      Output directory (default: runs/exp)
    echo   --weights         Pre-trained weights file (default: auto-detect)
    echo   --no-pretrained   Train from scratch instead of fine-tuning
    echo   --resume          Checkpoint to resume interrupted training
    echo   --epochs          Number of epochs (default: 100)
    echo   --batch-size      Batch size (default: 4)
    echo   --crop-size       Training crop size (default: 256)
    echo   --lr              Learning rate (default: 0.001)
    echo   --fg-weight       Foreground loss weight (default: 5.0)
    echo   --num-workers     DataLoader workers (default: 2)
    echo   --seed            Random seed (default: 42)
    echo.
    echo Examples:
    echo   train.bat --data-dir exported_data --epochs 50
    echo   train.bat --data-dir exported_data --weights my_model.pth --epochs 50
    echo   train.bat --data-dir exported_data --no-pretrained --epochs 200
    echo.
    pause
    exit /b 0
)

python -m training.train %*
pause
