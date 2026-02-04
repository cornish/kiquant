@echo off
echo Building kiQuant executable...

cd /d "%~dp0..\src"
call venv\Scripts\activate.bat

echo Running PyInstaller...
pyinstaller --noconfirm --onefile --windowed ^
    --name "kiQuant" ^
    --add-data "web;web" ^
    --hidden-import "bottle_websocket" ^
    main.py

echo.
if exist "dist\kiQuant.exe" (
    echo Build successful!
    echo Executable: %~dp0..\src\dist\kiQuant.exe
) else (
    echo Build failed!
)
pause
