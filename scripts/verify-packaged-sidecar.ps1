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
$defaultAssetCatalog = Get-Content -Raw -LiteralPath $defaultAssetIndex | ConvertFrom-Json
$qualifiedPairs = @($defaultAssetCatalog.qualified_pairs)
$expectedMapIds = @(
    $qualifiedPairs | ForEach-Object {
        @($_.packages | Where-Object { $_.kind -eq "map" })[0].asset_id
    }
)
$actualMapIds = @($mapVersions | ForEach-Object { $_.asset_id } | Sort-Object)
if ($qualifiedPairs.Count -ne 8 -or
    $mapVersions.Count -ne $qualifiedPairs.Count -or
    $vehicleVersions.Count -ne $qualifiedPairs.Count -or
    (Compare-Object ($expectedMapIds | Sort-Object) $actualMapIds) -or
    @($vehicleVersions | Where-Object { $_.asset_id -ne "dronedream.my-drone.v1" }).Count -ne 0 -or
    @($mapVersions + $vehicleVersions | Where-Object { $_.maturity -ne "qualified" }).Count -ne 0) {
    throw "Packaged default qualification catalog is incomplete: expected 8 qualified map/My Drone pairs."
}
$schoolMap = @($mapVersions | Where-Object { $_.asset_id -eq "dronedream.school-map.v1" })
$myDrone = @($vehicleVersions | Where-Object { $_.asset_id -eq "dronedream.my-drone.v1" })
if ($schoolMap.Count -ne 1 -or $schoolMap[0].maturity -ne "qualified" -or
    $myDrone.Count -lt 1 -or @($myDrone | Where-Object { $_.maturity -ne "qualified" }).Count -ne 0) {
    throw "Packaged canonical School Map and My Drone assets are not qualified."
}
$pairCatalog = Invoke-RestMethod `
    -Uri "http://127.0.0.1:$port/v1/asset-pair-catalog" `
    -Headers $headers `
    -TimeoutSec 10
if ($pairCatalog.map_count -ne 8 -or
    $pairCatalog.vehicle_count -ne 13 -or
    $pairCatalog.pair_count -ne 104 -or
    $pairCatalog.counts.qualified_builtin -ne 8 -or
    $pairCatalog.counts.compatible_requires_qualification -ne 78 -or
    $pairCatalog.counts.incompatible -ne 18) {
    throw "Packaged map/aircraft compatibility catalog is incomplete."
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
    "PLUGINS=$($bootstrap.plugins.Count) QUALIFIED_PAIRS=$($qualifiedPairs.Count) " +
    "CATALOG_PAIRS=$($pairCatalog.pair_count) ASSETS=canonical-catalog-and-external-import"
)
