# start_qdrant.ps1 — Start the Qdrant vector database server
# Run this BEFORE starting api.py. Keep this terminal open.
#
# Qdrant v1.17.1 standalone server
#   REST API : http://localhost:6333
#   gRPC     : localhost:6334
#   Dashboard: http://localhost:6333/dashboard
#   Storage  : data/qdrant_storage/

$ErrorActionPreference = "Stop"

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$QdrantExe  = Join-Path $ScriptDir "qdrant\qdrant.exe"
$QdrantCfg  = Join-Path $ScriptDir "qdrant\config.yaml"

if (-not (Test-Path $QdrantExe)) {
    Write-Error "qdrant.exe not found at $QdrantExe. Run the setup steps in README."
    exit 1
}

# Check if Qdrant is already running
try {
    $null = Invoke-WebRequest -Uri "http://localhost:6333/" -UseBasicParsing -TimeoutSec 2
    Write-Host "[INFO] Qdrant server is already running on port 6333." -ForegroundColor Yellow
    exit 0
} catch { }

Write-Host "[INFO] Starting Qdrant v1.17.1 server..." -ForegroundColor Cyan
Write-Host "[INFO] Storage : $ScriptDir\data\qdrant_storage" -ForegroundColor Cyan
Write-Host "[INFO] REST API: http://localhost:6333" -ForegroundColor Cyan
Write-Host "[INFO] Press Ctrl+C to stop." -ForegroundColor Cyan
Write-Host ""

Push-Location (Join-Path $ScriptDir "qdrant")
try {
    & $QdrantExe --config-path $QdrantCfg
} finally {
    Pop-Location
}
