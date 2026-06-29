param(
    [string]$Config = "config.yaml"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-Not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Virtual environment not found. Run .\scripts\setup_windows.ps1 first" -ForegroundColor Red
    exit 1
}

& .\.venv\Scripts\python.exe -m pip install PySide6 -q
& .\.venv\Scripts\python.exe -m src.ui_app
