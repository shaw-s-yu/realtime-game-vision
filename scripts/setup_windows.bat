@echo off
REM setup_windows.bat - alternative to ps1 for cmd users
echo === Realtime Game Vision Windows Setup ===
python --version
if errorlevel 1 (
    echo Python not found. Install Python 3.10+ and add to PATH.
    pause
    exit /b 1
)

if not exist .venv (
    echo Creating venv...
    python -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip wheel setuptools

REM Change to CPU if no NVIDIA GPU: replace cu121 with cpu below and onnxruntime-gpu with onnxruntime
echo Installing PyTorch CUDA 12.1 - edit this file for CPU version if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu

pip install -r requirements.txt

echo.
echo Setup complete. Activate with .venv\Scripts\activate.bat
echo Run with scripts\run.bat
pause
