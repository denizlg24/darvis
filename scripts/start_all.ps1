# Start all DARVIS services
$ErrorActionPreference = "Stop"

$scriptsDir = $PSScriptRoot

Write-Host "[STARTUP] Starting all DARVIS services..."
Write-Host ""

# Start transcription service in background
Write-Host "[STARTUP] Launching transcription service in background..."
$transcriptionJob = Start-Process powershell -ArgumentList "-NoExit", "-File", "$scriptsDir\start_transcription.ps1" -PassThru

# Give it time to start
Start-Sleep -Seconds 3

# Start wake word service in background
Write-Host "[STARTUP] Launching wake word service in background..."
$wakewordJob = Start-Process powershell -ArgumentList "-NoExit", "-File", "$scriptsDir\start_wakeword.ps1" -PassThru

# Give it time to start
Start-Sleep -Seconds 2

# Start DARVIS core in foreground
Write-Host "[STARTUP] Launching DARVIS core..."
try {
    & "$scriptsDir\start_darvis.ps1"
} finally {
    Write-Host ""
    Write-Host "[SHUTDOWN] Stopping wake word service..."
    Stop-Process -Id $wakewordJob.Id -Force -ErrorAction SilentlyContinue
    Write-Host "[SHUTDOWN] Stopping transcription service..."
    Stop-Process -Id $transcriptionJob.Id -Force -ErrorAction SilentlyContinue
    Write-Host "[SHUTDOWN] All services stopped"
}
