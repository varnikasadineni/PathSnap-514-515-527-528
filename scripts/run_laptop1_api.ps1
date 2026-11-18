# Member D — Laptop 1 Launcher (FastAPI Model, HMM/Viterbi & Server-Side Evaluation)
# Usage:
#   .\scripts\run_laptop1_api.ps1
#   .\scripts\run_laptop1_api.ps1 -EvalDir "data\raw\evaluator_test"

param(
    [string]$EvalDir = "",
    [int]$Port = 8000
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if ($EvalDir -ne "") {
    $env:PATHSNAP_EVAL_DIR = $EvalDir
    Write-Host "Using custom PATHSNAP_EVAL_DIR = $EvalDir" -ForegroundColor Cyan
}

Write-Host "============================================================" -ForegroundColor Green
Write-Host " PathSnap Laptop 1 — Starting FastAPI Backend (port $Port)" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "LAN IPv4 addresses on this laptop (give one to Laptop 2):" -ForegroundColor Yellow
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } | ForEach-Object {
    Write-Host ("  -> http://{0}:{1}" -f $_.IPAddress, $Port) -ForegroundColor White
}
Write-Host "If Laptop 2 cannot connect, run once in Admin PowerShell:" -ForegroundColor DarkGray
Write-Host "  netsh advfirewall firewall add rule name=`"PathSnap API`" dir=in action=allow protocol=TCP localport=$Port" -ForegroundColor DarkGray
Write-Host ""

& "$RepoRoot\.venv\Scripts\uvicorn.exe" api.main:app --host 0.0.0.0 --port $Port
