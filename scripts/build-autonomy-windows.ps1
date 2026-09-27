param(
    [switch]$SkipDependencyInstall,
    [switch]$StageOnly,

    [Parameter(Mandatory = $false)]
    [string]$LocalPolicyPackage,

    [Parameter(Mandatory = $false)]
    [string]$LocalPolicySimulationAdmission,

    [Parameter(Mandatory = $false)]
    [string]$LocalPolicyDistributionLicenses,

    [Parameter(Mandatory = $true)]
    [string]$NativeSensorRuntime,

    [Parameter(Mandatory = $true)]
    [string]$PayloadPlacementRuntime,

    [Parameter(Mandatory = $true)]
    [string]$CameraClockRuntime
)

$ErrorActionPreference = "Stop"
$policyInputs = @($LocalPolicyPackage, $LocalPolicySimulationAdmission, $LocalPolicyDistributionLicenses)
$policyInputCount = @($policyInputs | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }).Count
if ($policyInputCount -ne 0 -and $policyInputCount -ne 3) {
    throw "Optional local policy needs package, admission and distribution licenses together"
}
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

# Import narrowly owned build operations; these must never target runtime source
# or the installed DroneDreamRuntime WSL distribution.
. (Join-Path $PSScriptRoot 'build-output-safety.ps1')

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python virtual environment is missing: $python"
}
if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
    throw "npm was not found"
}
if (-not (Get-Command cargo.exe -ErrorAction SilentlyContinue)) {
    throw "Rust/Cargo was not found"
}

$sourceCommit = (& git -C $repoRoot rev-parse --verify HEAD).Trim()
$sourceTree = (& git -C $repoRoot rev-parse "HEAD^{tree}").Trim()
$sourceStatus = (& git -C $repoRoot status --porcelain=v1 --untracked-files=all | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or
    $sourceCommit -notmatch '^[0-9a-f]{40}$' -or
    $sourceTree -notmatch '^[0-9a-f]{40}$' -or
    $sourceStatus) {
    throw "The AGENT Windows build requires one exact clean source commit."
}

$oauthClientId = $null
foreach ($target in @(
    [EnvironmentVariableTarget]::Process,
    [EnvironmentVariableTarget]::User,
    [EnvironmentVariableTarget]::Machine
)) {
    foreach ($name in @(
        "DRONEDREAM_OAUTH_CLIENT_ID_AGENT",
        "DRONEDREAM_OAUTH_CLIENT_ID_AUTONOMY"
    )) {
        $oauthClientId = [Environment]::GetEnvironmentVariable($name, $target)
        if ($oauthClientId) { break }
    }
    if ($oauthClientId) { break }
}
if (-not $oauthClientId -or
    $oauthClientId -notmatch '^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$') {
    throw "Release builds require the registered public DRONEDREAM_OAUTH_CLIENT_ID_AGENT."
}
$env:DRONEDREAM_OAUTH_CLIENT_ID_AGENT = $oauthClientId
# Keep the internal autonomy protocol identifier readable by older build tooling.
$env:DRONEDREAM_OAUTH_CLIENT_ID_AUTONOMY = $oauthClientId

& $python (Join-Path $repoRoot "scripts\verify_shared_runtime_base.py")
if ($LASTEXITCODE -ne 0) { throw "Shared Runtime Base source contract failed" }

$brandSource = Join-Path $repoRoot "app\frontend\public\brand"
& $python (Join-Path $repoRoot "scripts\generate-autonomy-assets.py") `
    --repo $repoRoot `
    --source $brandSource
if ($LASTEXITCODE -ne 0) { throw "Autonomy brand and installer asset generation failed" }

if (-not $SkipDependencyInstall) {
    & $python -m pip install -e "$repoRoot[dev,local-policy]" `
        -c (Join-Path $repoRoot 'runtime\requirements-linux.txt')
    if ($LASTEXITCODE -ne 0) { throw "Python dependency installation failed" }
    & npm.cmd --prefix (Join-Path $repoRoot "app\frontend") ci
    if ($LASTEXITCODE -ne 0) { throw "Frontend dependency installation failed" }
    & npm.cmd --prefix (Join-Path $repoRoot "app\desktop") ci
    if ($LASTEXITCODE -ne 0) { throw "Desktop dependency installation failed" }
}

# 提供可选模型时验证全部 ONNX；独立路线控制不要求模型包。
if ($policyInputCount -eq 3) {
    & $python (Join-Path $repoRoot "scripts\stage_local_policy_runtime.py") `
        --package $LocalPolicyPackage --simulation-admission $LocalPolicySimulationAdmission `
        --distribution-licenses $LocalPolicyDistributionLicenses --check-only --verify-onnx
    if ($LASTEXITCODE -ne 0) { throw "Current local expert package preflight failed; existing output preserved" }
}
& $python (Join-Path $repoRoot "scripts\stage_native_sensor_runtime.py") `
    --source $NativeSensorRuntime --check-only
if ($LASTEXITCODE -ne 0) { throw "Current native sensor preflight failed; existing output preserved" }
& $python (Join-Path $repoRoot "scripts\build_payload_placement_runtime.py") `
    --stage-from $PayloadPlacementRuntime --check-only
if ($LASTEXITCODE -ne 0) { throw "Current payload placement preflight failed; existing output preserved" }
& $python (Join-Path $repoRoot "scripts\stage_native_camera_clock.py") `
    --source $CameraClockRuntime --check-only
if ($LASTEXITCODE -ne 0) { throw "Current camera clock preflight failed; existing output preserved" }

$officialPluginResources = Join-Path $repoRoot "app\desktop\src-tauri\resources\official-plugins"
# 索引存在不代表插件来自当前提交；每次组件构建都重新编译，不提供旧包复用入口。
& (Join-Path $repoRoot "scripts\build-official-plugins.ps1") `
    -OutputRoot $officialPluginResources
if ($LASTEXITCODE -ne 0) { throw "Official plugin bundle build failed" }

$runtimeSource = Join-Path $repoRoot "runtime"
$runtimeResources = Join-Path $repoRoot "app\desktop\src-tauri\resources\runtime"
$runtimeWheels = Join-Path $runtimeResources "wheels"
Reset-GeneratedDirectory $runtimeResources
[System.IO.Directory]::CreateDirectory($runtimeWheels) | Out-Null
# Keep the later first-party asset grant with the distributed Runtime, without
# changing qualification-bound DDPKG bytes. The runtime manifest hashes both files.
$runtimeLicenses = Join-Path $runtimeResources "licenses"
[System.IO.Directory]::CreateDirectory($runtimeLicenses) | Out-Null
Copy-Item -LiteralPath (Join-Path $repoRoot "LICENSE") `
    -Destination (Join-Path $runtimeLicenses "LICENSE")
Copy-Item -LiteralPath (Join-Path $runtimeSource "default-assets-licenses.json") `
    -Destination (Join-Path $runtimeLicenses "default-assets-licenses.json")
Copy-Item -LiteralPath (Join-Path $runtimeSource "requirements-linux.txt") `
    -Destination (Join-Path $runtimeResources "requirements-linux.txt")
Copy-Item -LiteralPath (Join-Path $runtimeSource "control-profile.json") `
    -Destination (Join-Path $runtimeResources "control-profile.json")
Copy-Item -LiteralPath (Join-Path $runtimeSource "px4_map_fusion_experiment_executor.py") `
    -Destination (Join-Path $runtimeResources "px4_map_fusion_experiment_executor.py")
Copy-Item -LiteralPath (Join-Path $runtimeSource "px4_offboard_track_executor.py") `
    -Destination (Join-Path $runtimeResources "px4_offboard_track_executor.py")
Copy-Item -LiteralPath (Join-Path $repoRoot "scripts\px4_checkpoint_executor.py") `
    -Destination (Join-Path $runtimeResources "px4_checkpoint_executor.py")
Copy-Item -LiteralPath (Join-Path $repoRoot "scripts\runtime_depth_safety_worker.py") `
    -Destination (Join-Path $runtimeResources "runtime_depth_safety_worker.py")
& $python (Join-Path $repoRoot "scripts\stage_native_sensor_runtime.py") `
    --source $NativeSensorRuntime --output (Join-Path $runtimeResources "native-sensors")
if ($LASTEXITCODE -ne 0) { throw "Current native sensor Runtime staging failed" }
& $python (Join-Path $repoRoot "scripts\build_payload_placement_runtime.py") `
    --stage-from $PayloadPlacementRuntime --output (Join-Path $runtimeResources "payload-placement")
if ($LASTEXITCODE -ne 0) { throw "Current payload placement Runtime staging failed" }
& $python (Join-Path $repoRoot "scripts\stage_native_camera_clock.py") `
    --source $CameraClockRuntime --output (Join-Path $runtimeResources "camera-clock")
if ($LASTEXITCODE -ne 0) { throw "Current camera clock Runtime staging failed" }
if ($policyInputCount -eq 3) {
& $python (Join-Path $repoRoot "scripts\stage_local_policy_runtime.py") `
    --package $LocalPolicyPackage `
    --simulation-admission $LocalPolicySimulationAdmission `
    --distribution-licenses $LocalPolicyDistributionLicenses `
    --verify-onnx `
    --output (Join-Path $runtimeResources "local-policy")
if ($LASTEXITCODE -ne 0) { throw "Admitted local expert Runtime staging failed" }
}
& $python (Join-Path $repoRoot "scripts\write_runtime_provenance.py") `
    --repository $repoRoot `
    --runtime-root $runtimeResources
if ($LASTEXITCODE -ne 0) { throw "Runtime source provenance generation failed" }
$runtimeRosSource = Join-Path $runtimeResources "ros_ws\src"
# Validate the source tree before Copy-Item can follow an external directory link.
Assert-PlainBuildTree -RepositoryRoot $repoRoot -Path (Join-Path $repoRoot 'ros_ws\src')
[System.IO.Directory]::CreateDirectory($runtimeRosSource) | Out-Null
Copy-Item -Path (Join-Path $repoRoot "ros_ws\src\*") -Destination $runtimeRosSource -Recurse -Force
Assert-PlainBuildTree -RepositoryRoot $repoRoot -Path $runtimeRosSource
Get-ChildItem -LiteralPath $runtimeRosSource -Directory -Filter "__pycache__" -Recurse |
    Sort-Object { $_.FullName.Length } -Descending |
    ForEach-Object {
        if (-not $_.FullName.StartsWith($runtimeResources.TrimEnd('\') + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Generated Python cache escaped runtime resources: $($_.FullName)"
        }
        Assert-PlainBuildTree -RepositoryRoot $repoRoot -Path $_.FullName
        [System.IO.Directory]::Delete($_.FullName, $true)
    }

$previousSourceDateEpoch = [Environment]::GetEnvironmentVariable(
    "SOURCE_DATE_EPOCH",
    [EnvironmentVariableTarget]::Process
)
try {
    # Runtime qualification is bound to the exact manifest hash.  A fixed
    # archive timestamp makes rebuilding the same runtime bytes reproducible,
    # so an installer rebuild cannot invalidate an otherwise identical flight
    # qualification merely because it ran at a different wall-clock time.
    $env:SOURCE_DATE_EPOCH = "946684800"
    & $python -m pip wheel --no-deps --wheel-dir $runtimeWheels $repoRoot
    if ($LASTEXITCODE -ne 0) { throw "Autonomy Linux core wheel build failed" }
} finally {
    if ([string]::IsNullOrWhiteSpace($previousSourceDateEpoch)) {
        Remove-Item Env:\SOURCE_DATE_EPOCH -ErrorAction SilentlyContinue
    } else {
        $env:SOURCE_DATE_EPOCH = $previousSourceDateEpoch
    }
}
& $python -m pip download `
    --dest $runtimeWheels `
    --requirement (Join-Path $runtimeSource "requirements-linux.txt") `
    --platform manylinux2014_x86_64 `
    --platform manylinux_2_28_x86_64 `
    --python-version 3.12 `
    --implementation cp `
    --abi cp312 `
    --only-binary=:all:
if ($LASTEXITCODE -ne 0) { throw "Autonomy Linux dependency bundle failed" }

$runtimeResourcePrefix = $runtimeResources.TrimEnd('\') + '\'
$runtimeFiles = Get-ChildItem -LiteralPath $runtimeResources -Recurse -File |
    Sort-Object FullName |
    ForEach-Object {
        if (-not $_.FullName.StartsWith(
            $runtimeResourcePrefix,
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
            throw "Runtime resource escaped generated root: $($_.FullName)"
        }
        [ordered]@{
            path = $_.FullName.Substring($runtimeResourcePrefix.Length).Replace('\', '/')
            sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            bytes = $_.Length
        }
    }
$runtimeManifest = [ordered]@{
    schema_version = "1.0.0"
    target = "DroneDreamRuntime/Ubuntu-24.04/Python-3.12/x86_64"
    source_date_epoch = 946684800
    files = @($runtimeFiles)
}
[System.IO.File]::WriteAllText(
    (Join-Path $runtimeResources "runtime-manifest.json"),
    (($runtimeManifest | ConvertTo-Json -Depth 5) + "`n"),
    [System.Text.UTF8Encoding]::new($false)
)

$sidecarDist = Join-Path $repoRoot "artifacts\sidecar"
$sidecarWork = Join-Path $repoRoot "artifacts\pyinstaller-work"
$sidecarSpec = Join-Path $repoRoot "artifacts\pyinstaller-spec"
New-Item -ItemType Directory -Force -Path $sidecarDist,$sidecarWork,$sidecarSpec | Out-Null
& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --name dronedream-autonomy-core `
    --collect-submodules dronedream_agent_plugins `
    --collect-data dronedream_agent_app `
    --distpath $sidecarDist `
    --workpath $sidecarWork `
    --specpath $sidecarSpec `
    (Join-Path $repoRoot "app\backend_entry.py")
if ($LASTEXITCODE -ne 0) { throw "AGENT Core sidecar build failed" }

$binaryRoot = Join-Path $repoRoot "app\desktop\src-tauri\binaries"
New-Item -ItemType Directory -Force -Path $binaryRoot | Out-Null
$targetBinary = Join-Path $binaryRoot "dronedream-autonomy-core-x86_64-pc-windows-msvc.exe"
Copy-Item -LiteralPath (Join-Path $sidecarDist "dronedream-autonomy-core.exe") -Destination $targetBinary -Force

$tauriManifest = Join-Path $repoRoot "app\desktop\src-tauri\Cargo.toml"
$previousTauriBundleType = [Environment]::GetEnvironmentVariable(
    "__TAURI_BUNDLE_TYPE",
    [EnvironmentVariableTarget]::Process
)
$previousTauriConfig = [Environment]::GetEnvironmentVariable(
    "TAURI_CONFIG",
    [EnvironmentVariableTarget]::Process
)
try {
    # The standalone isolator must be buildable in a clean checkout before its
    # own packaged external-binary path exists. The subsequent Tauri/NSIS build
    # restores the full config and verifies both real sidecars are present.
    $env:__TAURI_BUNDLE_TYPE = "nsis"
    $env:TAURI_CONFIG = '{"bundle":{"externalBin":[]}}'
    & cargo.exe build --release --manifest-path $tauriManifest --bin dronedream-plugin-isolator
    if ($LASTEXITCODE -ne 0) { throw "AppContainer plugin isolator build failed" }
} finally {
    if ($null -eq $previousTauriBundleType) {
        Remove-Item Env:\__TAURI_BUNDLE_TYPE -ErrorAction SilentlyContinue
    } else {
        $env:__TAURI_BUNDLE_TYPE = $previousTauriBundleType
    }
    if ($null -eq $previousTauriConfig) {
        Remove-Item Env:\TAURI_CONFIG -ErrorAction SilentlyContinue
    } else {
        $env:TAURI_CONFIG = $previousTauriConfig
    }
}
$isolatorTarget = Join-Path $binaryRoot "dronedream-plugin-isolator-x86_64-pc-windows-msvc.exe"
Copy-Item -LiteralPath (
    Join-Path $repoRoot "app\desktop\src-tauri\target\release\dronedream-plugin-isolator.exe"
) -Destination $isolatorTarget -Force

# 同时固定全部组件摘要及构建起点，五款产品只能暂存复验成功的这一份产物。
& $python (Join-Path $repoRoot "scripts\core_build_receipt.py") `
    --repository $repoRoot --expected-commit $sourceCommit
if ($LASTEXITCODE -ne 0) { throw "Core component source/binary binding failed" }
if ($StageOnly) {
    Write-Output "CORE_COMPONENT_RECEIPT=$(Join-Path $repoRoot 'artifacts\desktop\core-components-build.json')"
    return
}

& npm.cmd --prefix (Join-Path $repoRoot "app\desktop") run build
if ($LASTEXITCODE -ne 0) { throw "Tauri/NSIS build failed" }

$bundleRoot = Join-Path $repoRoot "app\desktop\src-tauri\target\release\bundle\nsis"
$installer = Get-ChildItem -LiteralPath $bundleRoot -Filter "*.exe" -File |
    Sort-Object LastWriteTimeUtc -Descending |
    Select-Object -First 1
if (-not $installer) { throw "NSIS installer was not produced" }
$releaseRoot = Join-Path $repoRoot "artifacts\desktop"
New-Item -ItemType Directory -Force -Path $releaseRoot | Out-Null
$releaseInstaller = Join-Path $releaseRoot "DroneDream-AGENT_1.0.0_x64-setup.exe"
Copy-Item -LiteralPath $installer.FullName -Destination $releaseInstaller -Force
$hash = (Get-FileHash -LiteralPath $releaseInstaller -Algorithm SHA256).Hash.ToLowerInvariant()
$checksumPath = "$releaseInstaller.sha256"
[System.IO.File]::WriteAllText(
    $checksumPath,
    "$hash *$([System.IO.Path]::GetFileName($releaseInstaller))`n",
    [System.Text.UTF8Encoding]::new($false)
)
$runtimeSourceCommit = (& git -C (Join-Path $repoRoot "shared\dronedream-runtime-source") rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw "Unable to resolve shared Runtime Base source commit" }
$runtimeBytes = (Get-ChildItem -LiteralPath $runtimeResources -Recurse -File |
    Measure-Object -Property Length -Sum).Sum
$localPolicyCatalogHash = $null
if ($policyInputCount -eq 3) {
    $localPolicyCatalogHash = (Get-FileHash -LiteralPath (
        Join-Path $runtimeResources "local-policy\catalog.json") -Algorithm SHA256).Hash.ToLowerInvariant()
}
$sourceInventory = [ordered]@{
    schema_version = "dronedream.autonomy.windows-build.v1"
    product = "DroneDream AGENT"
    version = "1.0.0"
    source_commit = $sourceCommit
    source_tree = $sourceTree
    shared_runtime_source_commit = $runtimeSourceCommit
    oauth_client_variable = "DRONEDREAM_OAUTH_CLIENT_ID_AGENT"
    installer_file = [System.IO.Path]::GetFileName($releaseInstaller)
    installer_sha256 = $hash
    runtime_files = $runtimeFiles.Count
    runtime_bytes = $runtimeBytes
    local_policy_catalog_sha256 = $localPolicyCatalogHash
}
$inventoryPath = Join-Path $releaseRoot "DroneDream-AGENT_1.0.0_x64-source-inventory.json"
[System.IO.File]::WriteAllText(
    $inventoryPath,
    (($sourceInventory | ConvertTo-Json -Depth 4) + "`n"),
    [System.Text.UTF8Encoding]::new($false)
)
Write-Output "INSTALLER=$releaseInstaller"
Write-Output "SHA256=$hash"
Write-Output "CHECKSUM=$checksumPath"
Write-Output "SOURCE_INVENTORY=$inventoryPath"
Write-Output "RUNTIME_FILES=$($runtimeFiles.Count)"
Write-Output "RUNTIME_BYTES=$runtimeBytes"
