# Stop all DARVIS services
Write-Host "[SHUTDOWN] Stopping DARVIS services..."

# Find and stop Python processes running uvicorn or darvis
Get-Process -Name "python*" -ErrorAction SilentlyContinue | ForEach-Object {
    $cmdLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $($_.Id)").CommandLine
    if ($cmdLine -match "uvicorn|darvis|wakeword") {
        Write-Host "[SHUTDOWN] Stopping process $($_.Id): $($_.Name)"
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "[SHUTDOWN] All services stopped"
