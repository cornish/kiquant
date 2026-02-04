@echo off
echo Setting up kiQuant development environment...

cd /d "%~dp0..\src"

if not exist "venv" (
    echo Creating virtual environment...

    REM Try Python 3.12 or 3.11 (compatible with TensorFlow for AI features)
    where py >nul 2>&1
    if %errorlevel%==0 (
        py -3.12 --version >nul 2>&1
        if %errorlevel%==0 (
            echo Using Python 3.12 for AI library compatibility...
            py -3.12 -m venv venv
        ) else (
            py -3.11 --version >nul 2>&1
            if %errorlevel%==0 (
                echo Using Python 3.11 for AI library compatibility...
                py -3.11 -m venv venv
            ) else (
                echo Python 3.11/3.12 not found, using default python...
                echo Note: StarDist/TensorFlow requires Python 3.12 or earlier
                python -m venv venv
            )
        )
    ) else (
        echo Python launcher not found, using default python...
        python -m venv venv
    )
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r ..\requirements.txt

echo.
echo Setup complete! Run 'scripts\run.bat' to start the application.
echo.
echo To add AI detection, run 'scripts\install-ai.bat'
pause
