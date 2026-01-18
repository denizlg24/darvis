# Start DARVIS TTS Service (Python 3.11)
$ErrorActionPreference = "Stop"

Write-Host "[STARTUP] Starting TTS service..."

$ttsDir = Join-Path $PSScriptRoot "..\darvis-tts"
Set-Location $ttsDir

# Run with uv
uv run python main.py
