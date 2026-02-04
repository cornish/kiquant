@echo off
echo Setting up kiQuant development environment...

cd /d "%~dp0..\src"

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r ..\requirements.txt

echo.
echo Setup complete! Run 'scripts\run.bat' to start the application.
pause
