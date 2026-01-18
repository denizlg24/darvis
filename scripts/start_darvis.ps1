# Start DARVIS Core (Python 3.14)
$ErrorActionPreference = "Stop"

$transcriptionHost = if ($env:TRANSCRIPTION_HOST) { $env:TRANSCRIPTION_HOST } else { "127.0.0.1" }
$transcriptionPort = if ($env:TRANSCRIPTION_PORT) { $env:TRANSCRIPTION_PORT } else { "8001" }
$transcriptionUrl = "http://${transcriptionHost}:${transcriptionPort}/health"

$wakewordHost = if ($env:WAKEWORD_HOST) { $env:WAKEWORD_HOST } else { "127.0.0.1" }
$wakewordPort = if ($env:WAKEWORD_PORT) { $env:WAKEWORD_PORT } else { "8002" }
$wakewordUrl = "http://${wakewordHost}:${wakewordPort}/health"

$maxAttempts = 30

# Wait for transcription service
Write-Host "[STARTUP] Waiting for transcription service at $transcriptionUrl..."
$attempt = 0
$ready = $false

while ($attempt -lt $maxAttempts -and -not $ready) {
    try {
        $response = Invoke-WebRequest -Uri $transcriptionUrl -TimeoutSec 1 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            $ready = $true
            Write-Host "[STARTUP] Transcription service is ready"
        }
    } catch {
        $attempt++
        Write-Host "[STARTUP] Waiting... ($attempt/$maxAttempts)"
        Start-Sleep -Seconds 1
    }
}

if (-not $ready) {
    Write-Host "[STARTUP] Warning: Transcription service not available, continuing with stub..."
}

# Wait for wake word service
Write-Host "[STARTUP] Waiting for wake word service at $wakewordUrl..."
$attempt = 0
$ready = $false

while ($attempt -lt $maxAttempts -and -not $ready) {
    try {
        $response = Invoke-WebRequest -Uri $wakewordUrl -TimeoutSec 1 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            $ready = $true
            Write-Host "[STARTUP] Wake word service is ready"
        }
    } catch {
        $attempt++
        Write-Host "[STARTUP] Waiting... ($attempt/$maxAttempts)"
        Start-Sleep -Seconds 1
    }
}

if (-not $ready) {
    Write-Host "[STARTUP] Warning: Wake word service not available, using hotkey fallback..."
}

Write-Host "[STARTUP] Starting DARVIS core..."

$coreDir = Join-Path $PSScriptRoot "..\darvis-core"
Set-Location $coreDir

# Run with uv
uv run python main.py
