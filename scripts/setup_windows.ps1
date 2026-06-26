#Requires -RunAsAdministrator
# setup_windows.ps1 - create venv and install dependencies for realtime-game-vision on Windows
param(
    [bool]$USE_CUDA = $true,
    [string]$PythonVersion = "3.10"
)

$ErrorActionPreference = "Stop"
Write-Host "=== Realtime Game Vision Windows Setup ===" -ForegroundColor Cyan

# Check Python
try {
    $py = Get-Command python -ErrorAction Stop
    $ver = python --version
    Write-Host "Found $ver at $($py.Source)"
} catch {
    Write-Host "Python not found. Install Python 3.10+ from python.org or Microsoft Store, ensure 'Add to PATH' checked." -ForegroundColor Red
    exit 1
}

# Create venv
if (-Not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment .venv ..."
    python -m venv .venv
} else {
    Write-Host ".venv already exists, reusing"
}

$venvPython = ".\.venv\Scripts\python.exe"
$venvPip = ".\.venv\Scripts\pip.exe"

& $venvPython -m pip install --upgrade pip wheel setuptools

if ($USE_CUDA) {
    Write-Host "Installing PyTorch with CUDA 12.1 ..."
    & $venvPip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    Write-Host "Installing ONNX Runtime GPU ..."
    & $venvPip install onnxruntime-gpu
} else {
    Write-Host "Installing PyTorch CPU ..."
    & $venvPip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    Write-Host "Installing ONNX Runtime CPU ..."
    & $venvPip install onnxruntime
}

Write-Host "Installing requirements.txt ..."
& $venvPip install -r requirements.txt

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Green
Write-Host "Activate venv:  .\.venv\Scripts\Activate.ps1"
Write-Host "Test capture:   python -c ""import dxcam; print(dxcam.__version__)"""
Write-Host "Run app:        .\scripts\run.ps1"
Write-Host ""
Write-Host "Optional Ollama for VLM:"
Write-Host "  winget install Ollama.Ollama"
Write-Host "  ollama pull moondream"
Write-Host "  # then set vlm.enabled: true in config.yaml"
