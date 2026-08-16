param(
    [switch]$VerifyOnly,
    [switch]$Elevated
)

$ErrorActionPreference = "Stop"

function Assert-LastExitCode {
    param([string]$Message)
    if ($LASTEXITCODE -ne 0) {
        throw "$Message (exit code $LASTEXITCODE)"
    }
}

function Assert-PublishedAsset {
    param(
        [string]$Label,
        [string]$Pattern
    )

    $Matches = @(docker exec spm-maritime-web find /usr/share/nginx/html/assets -maxdepth 1 -type f -name $Pattern 2>$null)
    if ($LASTEXITCODE -ne 0 -or $Matches.Count -eq 0) {
        throw "Missing published asset: $Label ($Pattern)"
    }
}

function Assert-PublishedMarker {
    param(
        [string]$Label,
        [string]$Marker
    )

    docker exec spm-maritime-web grep -RqFs -- $Marker /usr/share/nginx/html/assets 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Missing published marker: $Label [$Marker]"
    }
}

function Test-IsAdministrator {
    $Identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $Principal = New-Object Security.Principal.WindowsPrincipal($Identity)
    return $Principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Invoke-ElevatedScript {
    Write-Host "[ADMIN] Docker requires elevation. Accept the Windows UAC prompt." -ForegroundColor Yellow

    $Arguments = "-NoExit -NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Elevated"
    if ($VerifyOnly) {
        $Arguments += " -VerifyOnly"
    }

    try {
        Start-Process powershell.exe -Verb RunAs -ArgumentList $Arguments -WorkingDirectory $Root -ErrorAction Stop
        Write-Host "[OK] Deployment continued in the Administrator PowerShell window." -ForegroundColor Green
        exit 0
    }
    catch {
        throw "Administrator elevation was cancelled or unavailable. Re-run PowerShell as Administrator. $($_.Exception.Message)"
    }
}

function Get-DockerServerVersion {
    $script:LastDockerProbeError = $null

    foreach ($ApiVersion in @("1.50", "1.49", "1.47", "1.44")) {
        $env:DOCKER_API_VERSION = $ApiVersion
        $PreviousPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"

        try {
            $ProbeOutput = @(& docker version --format '{{.Server.Version}}' 2>&1)
            $ProbeExitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $PreviousPreference
        }

        if ($ProbeExitCode -eq 0) {
            $ServerVersion = ($ProbeOutput | Out-String).Trim()
            if ($ServerVersion) {
                return [pscustomobject]@{
                    ServerVersion = $ServerVersion
                    ApiVersion = $ApiVersion
                }
            }
        }

        $script:LastDockerProbeError = ($ProbeOutput | Out-String).Trim()
    }

    return $null
}

function Restart-DockerDesktopEngine {
    Write-Host "[REPAIR] Restarting Docker Desktop Engine" -ForegroundColor Yellow

    $PreviousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & wsl.exe --shutdown 2>$null
        Get-Process -Name "Docker Desktop", "com.docker.backend" -ErrorAction SilentlyContinue |
            Stop-Process -Force -ErrorAction SilentlyContinue
        $DockerService = Get-Service com.docker.service -ErrorAction Stop
        if ($DockerService.Status -eq "Running") {
            Stop-Service com.docker.service -Force -ErrorAction Stop
        }
        Start-Service com.docker.service -ErrorAction Stop
        Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe" -WindowStyle Hidden -ErrorAction Stop
    }
    catch {
        throw "Unable to restart Docker Desktop. Open PowerShell as Administrator and run this script again. $($_.Exception.Message)"
    }
    finally {
        $ErrorActionPreference = $PreviousPreference
    }
}

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Archive = Join-Path $Root ".live-metocean-src.tar"
$DockerConfig = Join-Path $Root ".docker-empty"

New-Item -ItemType Directory -Force -Path $DockerConfig | Out-Null
$env:DOCKER_CONFIG = $DockerConfig
$env:DOCKER_HOST = "npipe:////./pipe/dockerDesktopLinuxEngine"

Push-Location $Root
try {
    # Docker Desktop can expose its Linux engine while the optional Windows
    # service remains stopped. Probe the engine itself before touching services.
    $DockerProbe = Get-DockerServerVersion

    if ($null -eq $DockerProbe) {
        $DockerService = Get-Service com.docker.service -ErrorAction SilentlyContinue
        if ($null -ne $DockerService -and $DockerService.Status -ne "Running" -and -not (Test-IsAdministrator)) {
            Invoke-ElevatedScript
        }

        if ($null -ne $DockerService -and $DockerService.Status -ne "Running") {
            try {
                Start-Service com.docker.service -ErrorAction Stop
            }
            catch {
                Write-Host "[INFO] Docker service requires elevation; trying the user session" -ForegroundColor Yellow
            }
        }

        Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe" -WindowStyle Hidden -ErrorAction SilentlyContinue
        for ($Attempt = 1; $Attempt -le 6; $Attempt++) {
            $DockerProbe = Get-DockerServerVersion
            if ($null -ne $DockerProbe) { break }
            Write-Host "[WAIT] Docker Engine attempt $Attempt/6" -ForegroundColor Yellow
            Start-Sleep -Seconds 5
        }
    }

    if ($null -eq $DockerProbe) {
        if (-not (Test-IsAdministrator)) {
            Invoke-ElevatedScript
        }
        Restart-DockerDesktopEngine
        for ($Attempt = 1; $Attempt -le 30; $Attempt++) {
            $DockerProbe = Get-DockerServerVersion
            if ($null -ne $DockerProbe) { break }
            Write-Host "[WAIT] Docker Engine recovery $Attempt/30" -ForegroundColor Yellow
            Start-Sleep -Seconds 5
        }
    }

    if ($null -eq $DockerProbe) {
        throw "Docker Desktop Engine is unavailable after automatic recovery. Last error: $script:LastDockerProbeError"
    }

    Write-Host "[OK] Docker Server $($DockerProbe.ServerVersion), API $($DockerProbe.ApiVersion)" -ForegroundColor Green

    if (-not $VerifyOnly) {
        tar -cf $Archive src index.html nginx.conf package.json pnpm-lock.yaml
        Assert-LastExitCode "Unable to package the React sources"

        docker cp $Archive "spm-maritime-web:/tmp/live-metocean-src.tar"
        Assert-LastExitCode "Unable to copy the React sources into spm-maritime-web"

        docker exec spm-maritime-web sh -lc @'
set -eu
rm -rf /tmp/maritime-build/src
tar -xf /tmp/live-metocean-src.tar -C /tmp/maritime-build
cd /tmp/maritime-build
npm run build
rm -rf /usr/share/nginx/html/*
cp -r dist/* /usr/share/nginx/html/
cp /tmp/maritime-build/nginx.conf /etc/nginx/conf.d/default.conf
nginx -t
'@
        Assert-LastExitCode "Frontend compilation or publication failed"

        # A full restart activates the newly copied bundle without relying on
        # an in-place Nginx reload.
        docker restart spm-maritime-web | Out-Null
        Assert-LastExitCode "Unable to restart spm-maritime-web"
    }
    else {
        Write-Host "[VERIFY] Reusing the currently published bundle" -ForegroundColor Cyan
    }

    $ContainerReady = $false
    for ($Attempt = 1; $Attempt -le 20; $Attempt++) {
        # A restarted container can refuse connections for a few seconds.
        # PowerShell 5.1 otherwise promotes wget's transient stderr to a
        # terminating NativeCommandError because this script uses Stop.
        $PreviousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "SilentlyContinue"
        try {
            docker exec spm-maritime-web sh -lc "wget -q -O /dev/null http://127.0.0.1/health >/dev/null 2>&1"
            $ProbeExitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $PreviousErrorActionPreference
        }

        if ($ProbeExitCode -eq 0) {
            $ContainerReady = $true
            break
        }
        Write-Host "[WAIT] Nginx container attempt $Attempt/20" -ForegroundColor Yellow
        Start-Sleep -Seconds 2
    }

    if (-not $ContainerReady) {
        docker logs --tail 80 spm-maritime-web
        throw "Nginx did not become ready inside spm-maritime-web"
    }

    Assert-PublishedAsset -Label "weather page" -Pattern "WeatherPage-*.js"
    Assert-PublishedAsset -Label "capacity page" -Pattern "CapacityPage-*.js"
    Assert-PublishedAsset -Label "control tower page" -Pattern "ControlTowerPage-*.js"
    Assert-PublishedMarker -Label "control tower shell" -Marker "Control Tower"
    Assert-PublishedMarker -Label "interactive weather map" -Marker "portflow-strait"
    # Keep deployment markers ASCII-only because Windows PowerShell 5.1 can
    # reinterpret UTF-8 punctuation when this script has no BOM.
    Assert-PublishedMarker -Label "projection controls" -Marker "Ajuster l"
    Assert-PublishedMarker -Label "projection results" -Marker "Ce qui est pr"
    Assert-PublishedMarker -Label "history and forecast chart" -Marker "Historique et pr"
    Assert-PublishedMarker -Label "dynamic current date" -Marker "Aujourd"
    Assert-PublishedMarker -Label "Tanger timezone" -Marker "Africa/Casablanca"
    Assert-PublishedMarker -Label "capacity watchlist" -Marker "File de vigilance"
    Assert-PublishedMarker -Label "risk timeline" -Marker "VOLUTION DU RISQUE"
    Assert-PublishedMarker -Label "probabilistic ETA" -Marker "Temps restant probabiliste"
    Assert-PublishedMarker -Label "decision trajectory" -Marker "REVUE HUMAINE"
    Assert-PublishedMarker -Label "port-call vigilance" -Marker "Vigilance des escales"
    Assert-PublishedMarker -Label "restored weather dashboard" -Marker "Conditions actuelles"
    Assert-PublishedMarker -Label "PortFlow 2026 cyan palette" -Marker "#36D6CF"
    Assert-PublishedMarker -Label "live systems indicator" -Marker "LIVE"

    docker exec spm-maritime-web grep -Rqs -- "#35E3C0" /usr/share/nginx/html/assets 2>$null
    if ($LASTEXITCODE -eq 0) {
        throw "Legacy green accent is still present in the published bundle"
    }

    $FrontendReady = $false
    for ($Attempt = 1; $Attempt -le 12; $Attempt++) {
        try {
            $Probe = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:8088/health" -TimeoutSec 4
            if ($Probe.StatusCode -eq 200) {
                $FrontendReady = $true
                break
            }
        }
        catch {
            Write-Host "[WAIT] Docker port 8088 attempt $Attempt/12" -ForegroundColor Yellow
            Start-Sleep -Seconds 2
        }
    }

    if (-not $FrontendReady) {
        $PortBinding = @(docker port spm-maritime-web 80/tcp 2>&1) | Out-String
        docker ps --filter "name=spm-maritime-web" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        throw "The restored bundle and Nginx are healthy inside the container, but Docker Desktop is not forwarding port 8088 to Windows. Published binding: $($PortBinding.Trim())"
    }

    foreach ($Route in @("weather", "capacity", "control-tower")) {
        $Response = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:8088/$Route" -TimeoutSec 10
        if ($Response.StatusCode -ne 200) {
            throw "Frontend route /$Route returned HTTP $($Response.StatusCode)"
        }
    }

    Write-Host "[OK] PortFlow control tower deployed" -ForegroundColor Green
    Write-Host "Control Tower: http://127.0.0.1:8088/control-tower"
    Write-Host "Weather: http://127.0.0.1:8088/weather"
    Write-Host "Capacity: http://127.0.0.1:8088/capacity"
}
finally {
    Pop-Location
    if (Test-Path -LiteralPath $Archive) {
        Remove-Item -LiteralPath $Archive -Force
    }
}
