@echo off
REM Realtime Game Vision - One-click UI launcher for Windows
REM Double-click this file to open src.ui_app (PySide6 UI)
REM
REM How it works:
REM - cd to repo root
REM - Uses .venv\Scripts\python.exe if present
REM - Runs: python -m src.ui_app
REM
REM Qt DPI awareness fix:
REM - qt.conf in repo root sets dpiawareness=1 to avoid
REM   "SetProcessDpiAwarenessContext() failed: Access is denied."
REM - If warning persists, change qt.conf to dpiawareness=0
REM - Do NOT run as Administrator unless needed for dxcam; admin
REM   contexts often deny the DPI awareness API call.

setlocal
cd /d %~dp0

REM Suppress Qt DPI warning by overriding default per-monitor v2
set QT_ENABLE_HIGHDPI_SCALING=0
set QT_SCALE_FACTOR=1
REM Optional: force platform plugin awareness level (0=unaware,1=system,2=per-monitor)
set QT_QPA_PLATFORM=windows:dpiawareness=1

REM Ensure qt.conf is also next to python.exe (Qt application directory)
if exist qt.conf (
    if exist .venv\Scripts\python.exe (
        copy /Y qt.conf .venv\Scripts\qt.conf >nul 2>&1
    )
)

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
