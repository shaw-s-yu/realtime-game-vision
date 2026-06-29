@echo off
REM Launch UI config editor for Realtime Game Vision
REM Cross-platform UI built with PySide6 - works on Windows Linux macOS

setlocal
cd /d %~dp0..
if not exist .venv\Scripts\python.exe (
    echo Virtual environment not found at .venv\Scripts\python.exe
    echo Run scripts\setup_windows.bat first
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
python -m src.ui_app
pause
