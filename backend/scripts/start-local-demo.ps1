param(
    [int]$Port = 8092,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

$BackendRoot = Split-Path -Parent $PSScriptRoot
$RepositoryRoot = Split-Path -Parent $BackendRoot
$VirtualEnvironment = Join-Path $BackendRoot ".venv-local"
$VirtualPython = Join-Path $VirtualEnvironment "Scripts\python.exe"
$Requirements = Join-Path $BackendRoot "services\platform_api\requirements.txt"

function Resolve-BootstrapPython {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return $python.Source
    }

    $codexPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    if (Test-Path -LiteralPath $codexPython) {
        return $codexPython
    }

    throw "Python 3 est introuvable. Installez Python 3.12 ou lancez ce script depuis Codex Desktop."
}

if (-not (Test-Path -LiteralPath $VirtualPython)) {
    $BootstrapPython = Resolve-BootstrapPython
    Write-Host "Creation de l'environnement Python local..."
    & $BootstrapPython -m venv $VirtualEnvironment
}

if (-not $SkipInstall) {
    & $VirtualPython -c "import fastapi, uvicorn, psycopg2, pydantic" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installation des dependances de l'API..."
        & $VirtualPython -m pip install --disable-pip-version-check -r $Requirements
    }
}

$env:SMART_PORT_LOCAL_DEMO_MODE = "true"
$env:B56F_CORS_ORIGINS = "http://localhost:3000,http://localhost:4173,http://localhost:8088"

Write-Host ""
Write-Host "PortFlow API locale : http://localhost:$Port"
Write-Host "Mode               : LOCAL DEMO / SHADOW (aucune revendication production)"
Write-Host "Documentation      : http://localhost:$Port/docs"
Write-Host "Arret               : Ctrl+C"
Write-Host ""

Set-Location $BackendRoot
& $VirtualPython -m uvicorn platform_api.main:app `
    --app-dir services `
    --host 0.0.0.0 `
    --port $Port `
    --reload `
    --reload-dir (Join-Path $BackendRoot "services\platform_api")

