#!/bin/bash
# Start all DARVIS services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSCRIPTION_PID=""
WAKEWORD_PID=""
TTS_PID=""

cleanup() {
    echo ""
    echo "[SHUTDOWN] Stopping TTS service..."
    if [ -n "$TTS_PID" ]; then
        kill $TTS_PID 2>/dev/null || true
    fi
    echo "[SHUTDOWN] Stopping wake word service..."
    if [ -n "$WAKEWORD_PID" ]; then
        kill $WAKEWORD_PID 2>/dev/null || true
    fi
    echo "[SHUTDOWN] Stopping transcription service..."
    if [ -n "$TRANSCRIPTION_PID" ]; then
        kill $TRANSCRIPTION_PID 2>/dev/null || true
    fi
    echo "[SHUTDOWN] All services stopped"
}

trap cleanup EXIT

echo "[STARTUP] Starting all DARVIS services..."
echo ""

# Start transcription service in background
echo "[STARTUP] Launching transcription service in background..."
"$SCRIPT_DIR/start_transcription.sh" &
TRANSCRIPTION_PID=$!

# Give it time to start
sleep 3

# Start wake word service in background
echo "[STARTUP] Launching wake word service in background..."
"$SCRIPT_DIR/start_wakeword.sh" &
WAKEWORD_PID=$!

# Give it time to start
sleep 2

# Start TTS service in background
echo "[STARTUP] Launching TTS service in background..."
"$SCRIPT_DIR/start_tts.sh" &
TTS_PID=$!

# Give it time to start
sleep 3

# Start DARVIS core in foreground
echo "[STARTUP] Launching DARVIS core..."
"$SCRIPT_DIR/start_darvis.sh"
