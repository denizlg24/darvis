#!/bin/bash
# Start DARVIS Core (Python 3.14)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRANSCRIPTION_HOST="${TRANSCRIPTION_HOST:-127.0.0.1}"
TRANSCRIPTION_PORT="${TRANSCRIPTION_PORT:-8001}"
TRANSCRIPTION_URL="http://${TRANSCRIPTION_HOST}:${TRANSCRIPTION_PORT}/health"

WAKEWORD_HOST="${WAKEWORD_HOST:-127.0.0.1}"
WAKEWORD_PORT="${WAKEWORD_PORT:-8002}"
WAKEWORD_URL="http://${WAKEWORD_HOST}:${WAKEWORD_PORT}/health"

MAX_ATTEMPTS=30

# Wait for transcription service
echo "[STARTUP] Waiting for transcription service at $TRANSCRIPTION_URL..."
ATTEMPT=0
READY=false

while [ $ATTEMPT -lt $MAX_ATTEMPTS ] && [ "$READY" = false ]; do
    if curl -s -o /dev/null -w "%{http_code}" "$TRANSCRIPTION_URL" 2>/dev/null | grep -q "200"; then
        READY=true
        echo "[STARTUP] Transcription service is ready"
    else
        ATTEMPT=$((ATTEMPT + 1))
        echo "[STARTUP] Waiting... ($ATTEMPT/$MAX_ATTEMPTS)"
        sleep 1
    fi
done

if [ "$READY" = false ]; then
    echo "[STARTUP] Warning: Transcription service not available, continuing with stub..."
fi

# Wait for wake word service
echo "[STARTUP] Waiting for wake word service at $WAKEWORD_URL..."
ATTEMPT=0
READY=false

while [ $ATTEMPT -lt $MAX_ATTEMPTS ] && [ "$READY" = false ]; do
    if curl -s -o /dev/null -w "%{http_code}" "$WAKEWORD_URL" 2>/dev/null | grep -q "200"; then
        READY=true
        echo "[STARTUP] Wake word service is ready"
    else
        ATTEMPT=$((ATTEMPT + 1))
        echo "[STARTUP] Waiting... ($ATTEMPT/$MAX_ATTEMPTS)"
        sleep 1
    fi
done

if [ "$READY" = false ]; then
    echo "[STARTUP] Warning: Wake word service not available, using hotkey fallback..."
fi

echo "[STARTUP] Starting DARVIS core..."

cd "$SCRIPT_DIR/../darvis-core"
uv run python main.py
