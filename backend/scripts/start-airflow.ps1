$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$network = docker network ls --filter "name=^smart-port-backend$" --format "{{.Name}}"
if ($network -ne "smart-port-backend") {
    throw "Base stack is not running. Start compose.yaml first."
}

docker compose -f compose.airflow.yaml config --quiet
if ($LASTEXITCODE -ne 0) {
    throw "Invalid Airflow Compose configuration."
}

Write-Host "Building the Airflow image..."
$baseImage = "apache/airflow:3.3.0-python3.12"
$pullSucceeded = $false

for ($attempt = 1; $attempt -le 4; $attempt++) {
    Write-Host "Pulling $baseImage (attempt $attempt/4)..."
    docker pull $baseImage
    if ($LASTEXITCODE -eq 0) {
        $pullSucceeded = $true
        break
    }
    if ($attempt -lt 4) {
        Start-Sleep -Seconds (10 * $attempt)
    }
}

if (-not $pullSucceeded) {
    throw "Unable to download the Airflow base image after 4 attempts. Check Docker Desktop network/proxy settings."
}

$buildSucceeded = $false
for ($attempt = 1; $attempt -le 3; $attempt++) {
    Write-Host "Building Airflow (attempt $attempt/3)..."
    docker compose -f compose.airflow.yaml --progress plain build
    if ($LASTEXITCODE -eq 0) {
        $buildSucceeded = $true
        break
    }
    if ($attempt -lt 3) {
        Start-Sleep -Seconds (10 * $attempt)
    }
}

if (-not $buildSucceeded) {
    throw "Airflow image build failed. Containers were not started."
}

Write-Host "Starting Airflow services..."
docker compose -f compose.airflow.yaml up -d
if ($LASTEXITCODE -ne 0) {
    throw "Airflow services failed to start."
}

Write-Host "Airflow is starting."
Start-Sleep -Seconds 15
& "$PSScriptRoot\check-airflow.ps1"
