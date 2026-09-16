param(
    [Parameter(Mandatory = $true)]
    [string]$SidecarPath,
    [Parameter(Mandatory = $true)]
    [string]$ResourceRoot
)

$ErrorActionPreference = "Stop"
$resolvedSidecarPath = (Resolve-Path -LiteralPath $SidecarPath).Path
$resolvedResourceRoot = (Resolve-Path -LiteralPath $ResourceRoot).Path
$officialPluginIndex = Join-Path $resolvedResourceRoot "official-plugins\index.json"
$runtimeManifest = Join-Path $resolvedResourceRoot "runtime\runtime-manifest.json"
$defaultAssetIndex = Join-Path $resolvedResourceRoot "default-assets\index.json"
if (-not (Test-Path -LiteralPath $officialPluginIndex -PathType Leaf)) {
    throw "ResourceRoot must contain official-plugins/index.json: $resolvedResourceRoot"
}
if (-not (Test-Path -LiteralPath $runtimeManifest -PathType Leaf)) {
    throw "ResourceRoot must contain runtime/runtime-manifest.json: $resolvedResourceRoot"
}
if (-not (Test-Path -LiteralPath $defaultAssetIndex -PathType Leaf)) {
    throw "ResourceRoot must contain default-assets/index.json: $resolvedResourceRoot"
}
$dataRoot = Join-Path $env:TEMP ("dd-autonomy-sidecar-e2e-" + [Guid]::NewGuid().ToString("N"))
$sessionToken = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
$port = 43123
$process = Start-Process `
    -FilePath $resolvedSidecarPath `
    -ArgumentList @(
        "--host", "127.0.0.1",
        "--port", $port,
        "--token", $sessionToken,
        "--data-root", $dataRoot,
        "--resource-root", $resolvedResourceRoot
    ) `
    -PassThru `
    -WindowStyle Hidden

$ready = $false
for ($attempt = 0; $attempt -lt 80; $attempt += 1) {
    if ($process.HasExited) {
        throw "Packaged sidecar exited before its health endpoint became ready (exit code $($process.ExitCode))."
    }
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 1
        $ready = $true
        break
    } catch {
        Start-Sleep -Milliseconds 150
    }
}
if (-not $ready) {
    throw "Packaged sidecar health check timed out."
}
$headers = @{ Authorization = "Bearer $sessionToken" }
$bootstrap = Invoke-RestMethod `
    -Uri "http://127.0.0.1:$port/v1/bootstrap" `
    -Headers $headers `
    -TimeoutSec 10
if ($bootstrap.models.Count -ne 7 -or $bootstrap.plugins.Count -lt 4) {
    throw "Packaged sidecar bootstrap contract is incomplete."
}
$mapVersions = @($bootstrap.asset_versions | Where-Object { $_.kind -eq "map" })
$vehicleVersions = @($bootstrap.asset_versions | Where-Object { $_.kind -eq "vehicle" })
if ($mapVersions.Count -ne 1 -or $vehicleVersions.Count -ne 1) {
    throw "A fresh profile must contain exactly one qualified default map and aircraft."
}
if ($mapVersions[0].asset_id -ne "dronedream.school-map.v1" -or
    $mapVersions[0].maturity -ne "qualified" -or
    $vehicleVersions[0].asset_id -ne "dronedream.my-drone.v1" -or
    $vehicleVersions[0].maturity -ne "qualified") {
    throw "Packaged default assets are not the exact qualified School Map and My Drone pair."
}
Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:$port/shutdown" `
    -Headers $headers `
    -TimeoutSec 10 | Out-Null
$null = $process.WaitForExit(10000)
Start-Sleep -Seconds 1
if (-not $process.HasExited) {
    throw "Packaged sidecar did not exit after authenticated shutdown."
}
Write-Output (
    "HEALTH=$($health.status) MODELS=$($bootstrap.models.Count) " +
    "PLUGINS=$($bootstrap.plugins.Count) MAP=$($mapVersions[0].asset_id) " +
    "VEHICLE=$($vehicleVersions[0].asset_id) ASSETS=qualified-defaults-and-external-import"
)
