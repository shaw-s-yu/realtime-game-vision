@echo off
if not exist .venv\Scripts\python.exe (
    echo .venv not found, run scripts\setup_windows.bat first
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
python -m src.main --config config.yaml
pause
