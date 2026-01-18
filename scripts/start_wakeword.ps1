# Start DARVIS Wake Word Service (Python 3.11)
$ErrorActionPreference = "Stop"

Write-Host "[STARTUP] Starting wake word service..."

$wakewordDir = Join-Path $PSScriptRoot "..\darvis-wakeword"
Set-Location $wakewordDir

# Run with uv
uv run python main.py
