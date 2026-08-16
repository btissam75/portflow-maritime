$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example"
}

docker compose config --quiet
docker compose up -d --build

Write-Host "Waiting for services..."
Start-Sleep -Seconds 8
& "$PSScriptRoot\check-stack.ps1"
