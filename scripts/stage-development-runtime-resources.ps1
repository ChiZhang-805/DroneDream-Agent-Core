param(
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,

    [Parameter(Mandatory = $false)]
    [string]$LocalPolicyPackage,

    [Parameter(Mandatory = $false)]
    [string]$LocalPolicyQualification,

    [Parameter(Mandatory = $false)]
    [string]$LocalPolicySimulationAdmission,

    [Parameter(Mandatory = $true)]
    [string]$NativeSensorRuntime
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$buildRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot "..\..\Build"))
$output = [System.IO.Path]::GetFullPath($OutputRoot)
$allowedPrefix = $buildRoot.TrimEnd('\') + '\'
if (-not $output.StartsWith($allowedPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Development Runtime resources must stay under $buildRoot"
}
if (Test-Path -LiteralPath $output) {
    throw "Development Runtime resource output already exists: $output"
}

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python virtual environment is missing: $python"
}
$runtimeSource = Join-Path $repoRoot "runtime"
$runtimeOutput = Join-Path $output "runtime"
$wheels = Join-Path $runtimeOutput "wheels"
$rosOutput = Join-Path $runtimeOutput "ros_ws\src"
$pythonOutput = Join-Path $output "src"
[System.IO.Directory]::CreateDirectory($wheels) | Out-Null
[System.IO.Directory]::CreateDirectory($rosOutput) | Out-Null
[System.IO.Directory]::CreateDirectory($pythonOutput) | Out-Null

foreach ($name in @(
    "requirements-linux.txt",
    "px4_offboard_track_executor.py"
)) {
    Copy-Item -LiteralPath (Join-Path $runtimeSource $name) `
        -Destination (Join-Path $runtimeOutput $name)
}
Copy-Item -LiteralPath (Join-Path $repoRoot "scripts\px4_checkpoint_executor.py") `
    -Destination (Join-Path $runtimeOutput "px4_checkpoint_executor.py")
Copy-Item -LiteralPath (Join-Path $repoRoot "scripts\runtime_depth_safety_worker.py") `
    -Destination (Join-Path $runtimeOutput "runtime_depth_safety_worker.py")
& $python (Join-Path $repoRoot "scripts\stage_native_sensor_runtime.py") `
    --source $NativeSensorRuntime --output (Join-Path $runtimeOutput "native-sensors")
if ($LASTEXITCODE -ne 0) { throw "Current native sensor Runtime staging failed" }
& $python (Join-Path $repoRoot "scripts\write_runtime_provenance.py") `
    --repository $repoRoot `
    --runtime-root $runtimeOutput
if ($LASTEXITCODE -ne 0) { throw "Runtime source provenance generation failed" }
Copy-Item -Path (Join-Path $repoRoot "ros_ws\src\*") `
    -Destination $rosOutput -Recurse
Copy-Item -Path (Join-Path $repoRoot "src\*") `
    -Destination $pythonOutput -Recurse

# Source worktrees can contain interpreter caches left by local test runs. They
# are neither portable Runtime inputs nor reproducible source artifacts, so
# remove only caches proven to be inside this newly-created staging root.
$outputPrefix = $output.TrimEnd('\') + '\'
$pythonCaches = @(
    Get-ChildItem -LiteralPath $output -Recurse -Force -ErrorAction Stop |
        Where-Object {
            ($_.PSIsContainer -and $_.Name -eq "__pycache__") -or
            (-not $_.PSIsContainer -and $_.Extension -eq ".pyc")
        } |
        Sort-Object { $_.FullName.Length } -Descending
)
foreach ($cache in $pythonCaches) {
    $cachePath = [System.IO.Path]::GetFullPath($cache.FullName)
    if (-not $cachePath.StartsWith(
        $outputPrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Refusing to remove Python cache outside staging root: $cachePath"
    }
    Remove-Item -LiteralPath $cachePath -Recurse -Force
}

$hasLocalPolicy = -not [string]::IsNullOrWhiteSpace($LocalPolicyPackage)
$hasQualification = -not [string]::IsNullOrWhiteSpace($LocalPolicyQualification)
$hasSimulationAdmission = -not [string]::IsNullOrWhiteSpace($LocalPolicySimulationAdmission)
if (($hasQualification -and $hasSimulationAdmission) -or `
    ($hasLocalPolicy -ne ($hasQualification -or $hasSimulationAdmission))) {
    throw "A development local policy needs one package and exactly one receipt"
}
if ($hasLocalPolicy) {
    $packageSource = (Resolve-Path -LiteralPath $LocalPolicyPackage).Path
    $receiptSource = (
        Resolve-Path -LiteralPath $(
            if ($hasQualification) {
                $LocalPolicyQualification
            } else {
                $LocalPolicySimulationAdmission
            }
        )
    ).Path
    if (-not (Test-Path -LiteralPath $packageSource -PathType Container)) {
        throw "Development local policy package is not a directory: $packageSource"
    }
    if (-not (Test-Path -LiteralPath $receiptSource -PathType Leaf)) {
        throw "Development local policy receipt is not a file: $receiptSource"
    }
    $policyRoot = Join-Path $runtimeOutput "local-policy"
    if ($hasQualification) {
        & $python (Join-Path $repoRoot "scripts\stage_local_policy_runtime.py") `
            --package $packageSource `
            --qualification-receipt $receiptSource `
            --output $policyRoot
    } else {
        & $python (Join-Path $repoRoot "scripts\stage_local_policy_runtime.py") `
            --package $packageSource `
            --simulation-admission $receiptSource `
            --output $policyRoot
    }
    if ($LASTEXITCODE -ne 0) { throw "Development local expert Runtime staging failed" }
}

$previousSourceDateEpoch = [Environment]::GetEnvironmentVariable(
    "SOURCE_DATE_EPOCH",
    [EnvironmentVariableTarget]::Process
)
try {
    $env:SOURCE_DATE_EPOCH = "946684800"
    & $python -m pip wheel --no-deps --wheel-dir $wheels $repoRoot
    if ($LASTEXITCODE -ne 0) { throw "Development Runtime core wheel build failed" }
} finally {
    if ([string]::IsNullOrWhiteSpace($previousSourceDateEpoch)) {
        Remove-Item Env:\SOURCE_DATE_EPOCH -ErrorAction SilentlyContinue
    } else {
        $env:SOURCE_DATE_EPOCH = $previousSourceDateEpoch
    }
}
& $python -m pip download `
    --dest $wheels `
    --requirement (Join-Path $runtimeSource "requirements-linux.txt") `
    --platform manylinux2014_x86_64 `
    --platform manylinux_2_28_x86_64 `
    --python-version 3.12 `
    --implementation cp `
    --abi cp312 `
    --only-binary=:all:
if ($LASTEXITCODE -ne 0) { throw "Development Runtime dependency bundle failed" }

$prefix = $runtimeOutput.TrimEnd('\') + '\'
$files = Get-ChildItem -LiteralPath $runtimeOutput -Recurse -File |
    Sort-Object FullName |
    ForEach-Object {
        if (-not $_.FullName.StartsWith(
            $prefix,
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
            throw "Development Runtime resource escaped output root: $($_.FullName)"
        }
        [ordered]@{
            path = $_.FullName.Substring($prefix.Length).Replace('\', '/')
            sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            bytes = $_.Length
        }
    }
$manifest = [ordered]@{
    schema_version = "1.0.0"
    target = "DroneDreamRuntime/Ubuntu-24.04/Python-3.12/x86_64"
    source_date_epoch = 946684800
    development_only = $true
    files = @($files)
}
[System.IO.File]::WriteAllText(
    (Join-Path $runtimeOutput "runtime-manifest.json"),
    (($manifest | ConvertTo-Json -Depth 5) + "`n"),
    [System.Text.UTF8Encoding]::new($false)
)
Write-Output "DEVELOPMENT_RUNTIME_RESOURCES=$output"
