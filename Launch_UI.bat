@echo off
REM Realtime Game Vision - One-click UI launcher for Windows
REM Double-click this file to open src.ui_app (PySide6 UI)
REM
REM How it works:
REM - cd to repo root
REM - Uses .venv\Scripts\python.exe if present
REM - Runs: python -m src.ui_app

setlocal
cd /d %~dp0

if not exist .venv\Scripts\python.exe (
    echo Virtual environment not found at .venv\Scripts\python.exe
    echo.
    echo Please run scripts\setup_windows.bat first to create venv and install dependencies.
    echo.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
python -m src.ui_app
pause
