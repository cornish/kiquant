@echo off
cd /d "%~dp0..\src"
call venv\Scripts\activate.bat
python main.py
