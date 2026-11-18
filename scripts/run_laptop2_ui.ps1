# Member D — Laptop 2 Launcher (Evaluator-Facing Streamlit UI)
# Usage:
#   .\scripts\run_laptop2_ui.ps1 -ApiUrl "http://192.168.1.42:8000"
#   .\scripts\run_laptop2_ui.ps1   # defaults to http://127.0.0.1:8000

param(
    [string]$ApiUrl = "http://127.0.0.1:8000"
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:PATHSNAP_API_URL = $ApiUrl
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " PathSnap Laptop 2 — Starting Streamlit UI" -ForegroundColor Cyan
Write-Host " Connecting to Laptop 1 API: $ApiUrl" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

& "$RepoRoot\.venv\Scripts\streamlit.exe" run ui\app.py
