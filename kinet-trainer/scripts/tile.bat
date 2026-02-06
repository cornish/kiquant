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
    echo Image Tiling Script
    echo.
    echo Tiles large microscopy images into fixed-size patches for annotation.
    echo.
    echo Usage: tile.bat --source-dir ^<path^> --output-dir ^<path^> [options]
    echo.
    echo Options:
    echo   --source-dir        Directory of source images (required)
    echo   --output-dir        Directory for output tiles (required)
    echo   --tile-size         Tile size in pixels (default: 512)
    echo   --overlap           Overlap between tiles in pixels (default: 0)
    echo   --no-skip-blank     Keep blank/white background tiles
    echo   --blank-threshold   White pixel fraction to skip (default: 0.9)
    echo.
    echo Recommended tile sizes:
    echo   512  - Good balance of context and memory (default)
    echo   720  - Matches original KiNet training patches
    echo   256  - Minimal, for memory-constrained training
    echo.
    echo Examples:
    echo   tile.bat --source-dir C:\slides --output-dir C:\tiles
    echo   tile.bat --source-dir C:\slides --output-dir C:\tiles --tile-size 720
    echo.
    pause
    exit /b 0
)

python -m image_prep %*
pause
