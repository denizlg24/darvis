#!/bin/bash
# Test DARVIS TTS Service

set -e

TTS_HOST="${TTS_HOST:-127.0.0.1}"
TTS_PORT="${TTS_PORT:-8003}"
BASE_URL="http://${TTS_HOST}:${TTS_PORT}"

echo "Testing TTS service at ${BASE_URL}..."
echo ""

# Health check
echo "1. Health check..."
curl -s "${BASE_URL}/health" | python -m json.tool
echo ""

# List voices
echo "2. Available voices..."
curl -s "${BASE_URL}/voices" | python -m json.tool
echo ""

# Synthesize test
echo "3. Synthesizing test audio..."
curl -s -X POST "${BASE_URL}/synthesize" \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello, I am DARVIS, your voice assistant.", "voice": "am_michael"}' \
    --output /tmp/darvis_test.wav

if [ -f /tmp/darvis_test.wav ]; then
    echo "   Audio saved to /tmp/darvis_test.wav"
    ls -la /tmp/darvis_test.wav
    echo ""
    echo "   To play: aplay /tmp/darvis_test.wav (Linux) or afplay /tmp/darvis_test.wav (Mac)"
else
    echo "   ERROR: Failed to save audio file"
    exit 1
fi

echo ""
echo "All tests passed!"
