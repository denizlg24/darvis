#!/bin/bash
# Start DARVIS TTS Service (Python 3.11)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[STARTUP] Starting TTS service..."

cd "$SCRIPT_DIR/../darvis-tts"
uv run python main.py
