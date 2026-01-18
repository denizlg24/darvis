#!/bin/bash
# Stop all DARVIS services

echo "[SHUTDOWN] Stopping DARVIS services..."

# Kill processes matching uvicorn, darvis, or wakeword
pkill -f "uvicorn.*transcription" 2>/dev/null || true
pkill -f "uvicorn.*wakeword" 2>/dev/null || true
pkill -f "python.*darvis" 2>/dev/null || true

echo "[SHUTDOWN] All services stopped"
