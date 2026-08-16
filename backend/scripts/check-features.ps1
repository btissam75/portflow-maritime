$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

docker compose -f compose.features.yaml ps

$health = Invoke-RestMethod -Uri "http://localhost:8090/health" -TimeoutSec 5
$ready = Invoke-RestMethod -Uri "http://localhost:8090/ready" -TimeoutSec 10

Write-Host "Feature Builder health:"
Write-Host ($health | ConvertTo-Json -Depth 5)
Write-Host "Feature Builder dependencies:"
Write-Host ($ready | ConvertTo-Json -Depth 5)
