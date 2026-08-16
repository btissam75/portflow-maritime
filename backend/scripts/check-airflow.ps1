$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$ComposeArgs = @("compose", "-f", "compose.airflow.yaml")
& docker @ComposeArgs ps -a

$url = "http://localhost:8080/api/v2/monitor/health"
for ($attempt = 1; $attempt -le 30; $attempt++) {
    try {
        $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Host "[OK] Airflow API: $url"
            & docker @ComposeArgs exec -T airflow-scheduler airflow dags list
            Write-Host "Airflow ready: http://localhost:8080"
            exit 0
        }
    }
    catch {
        Write-Host "[WAIT] Airflow API attempt $attempt/30"
    }
    Start-Sleep -Seconds 5
}

Write-Host "Airflow is not healthy. Inspect: docker compose -f compose.airflow.yaml logs --tail=200"
exit 1
