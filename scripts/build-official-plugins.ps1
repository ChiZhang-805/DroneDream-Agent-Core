param(
    [string]$OutputRoot
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
. (Join-Path $PSScriptRoot 'build-output-safety.ps1')
$outputFull = Resolve-OfficialBuildOutput -RepositoryRoot $repoRoot -OutputRoot $OutputRoot
# Keep the published index and unrelated files intact until a new bundle exists.
[System.IO.Directory]::CreateDirectory($outputFull) | Out-Null

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$source = Join-Path $repoRoot "official_plugins\mission_evidence_gate\server.py"
$buildWorkspace = New-OfficialBuildWorkspace -RepositoryRoot $repoRoot
$work = $buildWorkspace.Path
$bundle = Join-Path $work "bundle"
$binary = Join-Path $bundle "bin"
[System.IO.Directory]::CreateDirectory($binary) | Out-Null

& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --name mission-evidence-gate `
    --distpath $binary `
    --workpath (Join-Path $work "pyinstaller-work") `
    --specpath (Join-Path $work "pyinstaller-spec") `
    $source
if ($LASTEXITCODE -ne 0) { throw "Mission Evidence Gate build failed" }

$executable = Join-Path $binary "mission-evidence-gate.exe"
$executableHash = (Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash.ToLowerInvariant()
$pluginVersion = "1.0.0+" + $executableHash.Substring(0, 12)
$manifestPath = Join-Path $bundle "plugin.json"
& $python $source `
    --manifest-sha256 $executableHash `
    --manifest-version $pluginVersion `
    --manifest-output $manifestPath
if ($LASTEXITCODE -ne 0) { throw "Mission Evidence Gate manifest generation failed" }

$archiveName = "mission-evidence-gate-$pluginVersion.zip"
$stagedArchive = Join-Path $work $archiveName
# Hash complete staged bytes before publishing; a failed compiler/compressor
# cannot destroy the currently indexed bundle.
Assert-PlainBuildTree -RepositoryRoot $buildWorkspace.Owner -Path $bundle
Compress-Archive -Path (Join-Path $bundle "*") -DestinationPath $stagedArchive -CompressionLevel Optimal
$archiveHash = (Get-FileHash -LiteralPath $stagedArchive -Algorithm SHA256).Hash.ToLowerInvariant()
# Key the published archive by ZIP bytes, not only the executable hash: ZIP
# metadata may change across rebuilds of an identical executable. Otherwise a
# failed index update could leave the old index pointing at different bytes.
$archiveName = "mission-evidence-gate-$archiveHash.zip"
$archive = Join-Path $outputFull $archiveName
Publish-GeneratedBuildFile -RepositoryRoot $buildWorkspace.Owner -Source $stagedArchive -Destination $archive
$index = [ordered]@{
    schema_version = "dronedream.official-plugin-index.v1"
    plugins = @(
        [ordered]@{
            plugin_id = "dronedream.mission-evidence-gate"
            version = $pluginVersion
            file = $archiveName
            sha256 = $archiveHash
        }
    )
}
[System.IO.File]::WriteAllText(
    (Join-Path $work "index.json"),
    (($index | ConvertTo-Json -Depth 5) + "`n"),
    [System.Text.UTF8Encoding]::new($false)
)
# The index is the commit point; older archives and isolated build diagnostics
# remain recoverable and are not swept by a successful or failed build.
Publish-GeneratedBuildFile -RepositoryRoot $buildWorkspace.Owner `
    -Source (Join-Path $work 'index.json') -Destination (Join-Path $outputFull 'index.json')
Write-Output "BUILD_EVIDENCE=$work"
Write-Output "OFFICIAL_PLUGIN=$archive"
Write-Output "SHA256=$archiveHash"
