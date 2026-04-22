# start_medgraph.ps1 — Start Qdrant server + MedGraph API (two separate terminals)
#
# Usage:
#   .\start_medgraph.ps1              # single worker (default)
#   .\start_medgraph.ps1 -Workers 4  # multi-worker (scales because Qdrant is now a server)
#
# Architecture note:
#   Qdrant runs as a standalone server process (data/qdrant_storage/).
#   Multiple api.py workers connect to it over TCP — no file lock contention.

param(
    [int]$Workers = 1
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# ── 1. Ensure Qdrant is up ────────────────────────────────────────────────────
$qdrantRunning = $false
try {
    $r = Invoke-WebRequest -Uri "http://localhost:6333/" -UseBasicParsing -TimeoutSec 2
    $qdrantRunning = $true
    Write-Host "[OK] Qdrant already running." -ForegroundColor Green
} catch { }

if (-not $qdrantRunning) {
    Write-Host "[INFO] Launching Qdrant server in a new window..." -ForegroundColor Cyan
    Start-Process powershell -ArgumentList "-NoExit", "-File", "$ScriptDir\start_qdrant.ps1" -WindowStyle Normal
    Write-Host "[INFO] Waiting for Qdrant to become ready (/readyz)..."
    $retries = 0
    while ($retries -lt 60) {
        Start-Sleep -Seconds 1
        try {
            $rz = Invoke-WebRequest -Uri "http://127.0.0.1:6333/readyz" -UseBasicParsing -TimeoutSec 1 -ErrorAction Stop
            if ($rz.StatusCode -eq 200) {
                Write-Host "[OK] Qdrant is ready." -ForegroundColor Green
                break
            }
        } catch { $retries++ }
    }
    if ($retries -ge 60) {
        Write-Error "Qdrant did not pass /readyz within 60 seconds. Check qdrant window for errors."
        exit 1
    }
}

# ── 2. Activate venv ─────────────────────────────────────────────────────────
$activate = Join-Path $ScriptDir "venv\Scripts\Activate.ps1"
if (Test-Path $activate) { & $activate }

# ── 3. Start API ──────────────────────────────────────────────────────────────
Set-Location $ScriptDir

if ($Workers -eq 1) {
    Write-Host "[INFO] Starting MedGraph API (single worker) on port 7860..." -ForegroundColor Cyan
    & python src/api.py
} else {
    Write-Host "[INFO] Starting MedGraph API ($Workers workers) on port 7860..." -ForegroundColor Cyan
    Write-Host "[INFO] Multi-worker mode enabled — Qdrant server handles concurrent access." -ForegroundColor Cyan
    & python -m uvicorn src.api:app --host 0.0.0.0 --port 7860 --workers $Workers
}
