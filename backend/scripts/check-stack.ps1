$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

function Test-HttpEndpoint {
    param(
        [string]$Name,
        [string]$Url
    )

    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
        Write-Host ("[OK] {0,-12} {1} ({2})" -f $Name, $Url, $response.StatusCode)
        return $true
    }
    catch {
        Write-Host ("[WAIT] {0,-10} {1}" -f $Name, $Url)
        return $false
    }
}

docker compose ps

$attempts = 18
for ($attempt = 1; $attempt -le $attempts; $attempt++) {
    $minio = Test-HttpEndpoint "MinIO API" "http://localhost:9000/minio/health/live"
    $mlflow = Test-HttpEndpoint "MLflow" "http://localhost:5000/health"
    $grafana = Test-HttpEndpoint "Grafana" "http://localhost:3001/api/health"

    if ($minio -and $mlflow -and $grafana) {
        docker compose exec -T timescaledb psql -U smartport -d maritime -c "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';"
        docker compose exec -T timescaledb psql -U smartport -d maritime -c "SELECT schema_name FROM information_schema.schemata WHERE schema_name IN ('core', 'features', 'serving', 'audit') ORDER BY schema_name;"
        Write-Host "Stack ready."
        exit 0
    }

    if ($attempt -lt $attempts) {
        Start-Sleep -Seconds 5
    }
}

Write-Host "Some services are not healthy. Inspect logs with: docker compose logs --tail=200"
exit 1
