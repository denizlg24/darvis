# Test DARVIS TTS Service

$TTS_HOST = if ($env:TTS_HOST) { $env:TTS_HOST } else { "127.0.0.1" }
$TTS_PORT = if ($env:TTS_PORT) { $env:TTS_PORT } else { "8003" }
$BASE_URL = "http://${TTS_HOST}:${TTS_PORT}"

Write-Host "Testing TTS service at ${BASE_URL}..."
Write-Host ""

# Health check
Write-Host "1. Health check..."
$health = Invoke-RestMethod -Uri "${BASE_URL}/health" -Method Get
$health | ConvertTo-Json
Write-Host ""

# List voices
Write-Host "2. Available voices..."
$voices = Invoke-RestMethod -Uri "${BASE_URL}/voices" -Method Get
$voices | ConvertTo-Json -Depth 3
Write-Host ""

# Synthesize test
Write-Host "3. Synthesizing test audio..."
$body = @{
    text = "Hello, I am DARVIS, your voice assistant."
    voice = "am_michael"
} | ConvertTo-Json

$outputPath = "$env:TEMP\darvis_test.wav"
Invoke-RestMethod -Uri "${BASE_URL}/synthesize" -Method Post -Body $body -ContentType "application/json" -OutFile $outputPath

if (Test-Path $outputPath) {
    Write-Host "   Audio saved to $outputPath"
    Get-Item $outputPath | Select-Object Name, Length, LastWriteTime
    Write-Host ""
    Write-Host "   To play, open the file or use a media player"
} else {
    Write-Host "   ERROR: Failed to save audio file"
    exit 1
}

Write-Host ""
Write-Host "All tests passed!"
