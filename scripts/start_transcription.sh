#!/bin/bash
# Start DARVIS Transcription Service (Python 3.11)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[STARTUP] Starting transcription service..."

cd "$SCRIPT_DIR/../darvis-transcription"
uv run python main.py
