# Start DARVIS Transcription Service (Python 3.11)
$ErrorActionPreference = "Stop"

Write-Host "[STARTUP] Starting transcription service..."

$transcriptionDir = Join-Path $PSScriptRoot "..\darvis-transcription"
Set-Location $transcriptionDir

# Run with uv
uv run python main.py
