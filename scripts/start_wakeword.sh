#!/bin/bash
# Start DARVIS Wake Word Service (Python 3.11)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[STARTUP] Starting wake word service..."

cd "$SCRIPT_DIR/../darvis-wakeword"
uv run python main.py
