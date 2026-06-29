param()
$ErrorActionPreference = "Stop"
if (-Not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "venv not found, run scripts\setup_windows.ps1 first" -ForegroundColor Red
    exit 1
}
& .\.venv\Scripts\python.exe -m src.ui_launcher
