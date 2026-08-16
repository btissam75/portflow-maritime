$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

docker compose -f compose.airflow.yaml down
Write-Host "Airflow stopped. Metadata database and named volumes were preserved."
