$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$network = docker network ls --filter "name=^smart-port-backend$" --format "{{.Name}}"
if ($network -ne "smart-port-backend") {
    throw "Base stack is not running. Start compose.yaml first."
}

docker compose -f compose.features.yaml config --quiet
if ($LASTEXITCODE -ne 0) {
    throw "Invalid Feature Builder Compose configuration."
}

$buildSucceeded = $false
for ($attempt = 1; $attempt -le 3; $attempt++) {
    Write-Host "Building Feature Builder (attempt $attempt/3)..."
    docker compose -f compose.features.yaml --progress plain build
    if ($LASTEXITCODE -eq 0) {
        $buildSucceeded = $true
        break
    }
    if ($attempt -lt 3) {
        Start-Sleep -Seconds (10 * $attempt)
    }
}

if (-not $buildSucceeded) {
    throw "Feature Builder image build failed."
}

docker compose -f compose.features.yaml up -d --force-recreate
if ($LASTEXITCODE -ne 0) {
    throw "Feature Builder failed to start."
}

Write-Host "Waiting for Feature Builder readiness..."
$readySucceeded = $false
for ($attempt = 1; $attempt -le 30; $attempt++) {
    try {
        $health = Invoke-RestMethod -Uri "http://localhost:8090/health" -TimeoutSec 5
        $ready = Invoke-RestMethod -Uri "http://localhost:8090/ready" -TimeoutSec 10
        if ($health.status -eq "healthy" -and $ready.status -eq "ready") {
            Write-Host "[OK] Feature Builder: http://localhost:8090"
            Write-Host ($ready | ConvertTo-Json -Depth 5)
            $readySucceeded = $true
            break
        }
    }
    catch {
        Write-Host "[WAIT] Feature Builder attempt $attempt/30"
    }
    Start-Sleep -Seconds 5
}

if (-not $readySucceeded) {
    docker compose -f compose.features.yaml logs --tail=200
    throw "Feature Builder is not ready."
}
