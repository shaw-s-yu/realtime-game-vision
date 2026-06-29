#!/usr/bin/env bash
# Cross-platform UI launcher for Linux and macOS
set -e
cd "$(dirname "$0")/.."
if [ ! -f ".venv/bin/python" ]; then
  echo "Virtual environment not found at .venv/bin/python"
  echo "Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi
source .venv/bin/activate
pip install -q PySide6
python -m src.ui_app
