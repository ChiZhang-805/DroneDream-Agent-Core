# Build-time ownership checks only. Dot-sourcing never starts a build or touches
# output. Callers must serialize builds; path checks are not an OS sandbox against
# another process replacing a checked directory between validation and I/O.

function Assert-PlainBuildPath {
    <# Validate an owned descendant and every ancestor before writing or deleting. #>
    param([string]$RepositoryRoot, [string]$Path, [switch]$File)
    $root = [System.IO.Path]::GetFullPath($RepositoryRoot).TrimEnd('\')
    $full = [System.IO.Path]::GetFullPath($Path).TrimEnd('\')
    if (-not $full.StartsWith($root + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw 'Build path escaped the repository.'
    }
    $cursor = [System.IO.Path]::GetPathRoot($full)
    foreach ($part in $full.Substring($cursor.Length).Split('\')) {
        # Reject Win32 aliases and alternate streams instead of relying on each
        # filesystem API to normalize an ambiguous spelling in the same way.
        if (-not $part -or $part -match '[<>:"|?*\x00-\x1f]' -or
            $part.EndsWith('.') -or $part.EndsWith(' ')) {
            throw 'Ambiguous Windows build path component.'
        }
        $cursor = Join-Path $cursor $part
        try { $item = Get-Item -LiteralPath $cursor -Force -ErrorAction Stop }
        catch [System.Management.Automation.ItemNotFoundException] { continue }
        if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw 'Build paths cannot cross links or reparse points.'
        }
        $isLeaf = [string]::Equals($cursor, $full, [System.StringComparison]::OrdinalIgnoreCase)
        if (($File -and $isLeaf) -eq [bool]$item.PSIsContainer) {
            throw 'Build path has the wrong file/directory kind.'
        }
    }
}

function Assert-PlainBuildTree {
    <# Inspect the entire tree before mutation; never descend through a junction. #>
    param([string]$RepositoryRoot, [string]$Path)
    Assert-PlainBuildPath -RepositoryRoot $RepositoryRoot -Path $Path
    if (-not [System.IO.Directory]::Exists($Path)) { return }
    $pending = [System.Collections.Generic.Stack[string]]::new()
    $pending.Push([System.IO.Path]::GetFullPath($Path))
    while ($pending.Count -gt 0) {
        foreach ($item in @(Get-ChildItem -LiteralPath $pending.Pop() -Force -ErrorAction Stop)) {
            if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw 'Generated tree contains a link or reparse point.'
            }
            if ($item.PSIsContainer) { $pending.Push($item.FullName) }
        }
    }
}

function Reset-GeneratedDirectory {
    <# Clear only staged installer Runtime resources, never source/runtime or a WSL installation. #>
    param([string]$Path, [string]$RepositoryRoot = $repoRoot)
    $full = [System.IO.Path]::GetFullPath($Path).TrimEnd('\')
    $owned = [System.IO.Path]::GetFullPath(
        (Join-Path $RepositoryRoot 'app\desktop\src-tauri\resources\runtime')
    )
    if (-not [string]::Equals($full, $owned, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw 'Only the named generated Runtime resource directory can be reset.'
    }
    # Complete the preflight before removing even the first ordinary file. A
    # nested junction must fail with the pre-existing generated contents intact.
    Assert-PlainBuildTree -RepositoryRoot $RepositoryRoot -Path $full
    if ([System.IO.Directory]::Exists($full)) {
        [System.IO.Directory]::Delete($full, $true)
    }
    [System.IO.Directory]::CreateDirectory($full) | Out-Null
}

function Resolve-OfficialBuildOutput {
    <# Restrict plugin publication to its two documented roots; preserve unrelated outputs. #>
    param([string]$RepositoryRoot, [string]$OutputRoot)
    $artifact = [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot 'artifacts\official-plugins'))
    $resource = [System.IO.Path]::GetFullPath(
        (Join-Path $RepositoryRoot 'app\desktop\src-tauri\resources\official-plugins')
    )
    if (-not $OutputRoot) { $OutputRoot = $artifact }
    $full = [System.IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
    if (-not [string]::Equals($full, $artifact, [System.StringComparison]::OrdinalIgnoreCase) -and
        -not [string]::Equals($full, $resource, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw 'Official plugin output is not an owned generated directory.'
    }
    Assert-PlainBuildPath -RepositoryRoot $RepositoryRoot -Path $full
    return $full
}

function Publish-GeneratedBuildFile {
    <# Atomically publish one completed same-volume file; leave the old destination on failure. #>
    param([string]$RepositoryRoot, [string]$Source, [string]$Destination)
    Assert-PlainBuildPath -RepositoryRoot $RepositoryRoot -Path $Source -File
    Assert-PlainBuildPath -RepositoryRoot $RepositoryRoot -Path $Destination -File
    if ([System.IO.File]::Exists($Destination)) {
        # PowerShell binds $null to an empty string for this overload; explicitly
        # pass a null string so .NET does not try to create an empty backup path.
        [System.IO.File]::Replace($Source, $Destination, [System.Management.Automation.Language.NullString]::Value)
    } else {
        [System.IO.File]::Move($Source, $Destination)
    }
}

function New-OfficialBuildWorkspace {
    <# Isolate each build under the owning workspace's Build root, outside packaged resources. #>
    param([string]$RepositoryRoot)
    $repository = [System.IO.DirectoryInfo]::new([System.IO.Path]::GetFullPath($RepositoryRoot))
    $owner = $repository.FullName
    for ($ancestor = $repository; $null -ne $ancestor; $ancestor = $ancestor.Parent) {
        if ($ancestor.Name -eq 'DroneDream-Workspace') {
            $owner = $ancestor.FullName
            break
        }
    }
    # Portable checkouts without the canonical workspace use their own Build
    # directory; never invent a new top-level drive directory or reuse .build.
    $work = Join-Path $owner ('Build\Official-Plugins\' + $repository.Name + '\' + [guid]::NewGuid().ToString('N'))
    Assert-PlainBuildPath -RepositoryRoot $owner -Path $work
    if (Test-Path -LiteralPath $work) { throw 'Isolated build workspace already exists.' }
    [System.IO.Directory]::CreateDirectory($work) | Out-Null
    return [pscustomobject]@{ Owner = $owner; Path = $work }
}
