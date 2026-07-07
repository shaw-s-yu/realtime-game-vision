#!/usr/bin/env bash
# Realtime Game Vision - One-click UI launcher
# Double-click this file (or run ./Launch_UI.sh) to open src.ui_app
#
# How it works:
# - cd to repo root
# - Uses .venv/bin/python if present, else python3
# - Auto-installs PySide6 if missing
# - Runs: python -m src.ui_app  (PySide6 UI with Custom/All/Screen tabs)

set -e
cd "$(dirname "$0")"

if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

"$PYTHON" -c "import PySide6" 2>/dev/null || {
    echo "Installing PySide6..."
    "$PYTHON" -m pip install -q PySide6
}

exec "$PYTHON" -m src.ui_app
