param(
    [Parameter(Mandatory = $true)]
    [string] $CredentialPath,
    [Parameter(Mandatory = $true)]
    [string] $CliExecutable,
    [string] $ResourceRoot,
    [string] $SessionRoot,
    [switch] $CheckOnly
)

$ErrorActionPreference = 'Stop'

$sourceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$python = Join-Path $sourceRoot '.venv\Scripts\python.exe'
if (-not $ResourceRoot) {
    $ResourceRoot = Join-Path $sourceRoot 'app\desktop\src-tauri\resources'
}
$pluginCandidates = @(
    (Join-Path $sourceRoot 'app\desktop\src-tauri\binaries\dronedream-plugin-isolator-x86_64-pc-windows-msvc.exe')
)
$cli = $CliExecutable
if (-not $SessionRoot) {
    $SessionRoot = Join-Path $sourceRoot 'artifacts\live-cli'
}
$project = 'yggabfynndpzymlqvnim'

$requiredPaths = @($sourceRoot, $python, $resourceRoot, $cli, $credentialPath)
foreach ($path in $requiredPaths) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "DroneDream live-session dependency is unavailable: $path"
    }
}

$pluginIsolator = $pluginCandidates |
    Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
    Select-Object -First 1
if (-not $pluginIsolator) {
    throw "DroneDream live-session dependency is unavailable. Checked: $($pluginCandidates -join ', ')"
}

New-Item -ItemType Directory -Path $sessionRoot -Force | Out-Null
$stdoutLog = Join-Path $sessionRoot 'core.stdout.log'
$stderrLog = Join-Path $sessionRoot 'core.stderr.log'
Set-Content -LiteralPath $stdoutLog -Value '' -Encoding UTF8
Set-Content -LiteralPath $stderrLog -Value '' -Encoding UTF8

$listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
$listener.Start()
$corePort = ([Net.IPEndPoint] $listener.LocalEndpoint).Port
$listener.Stop()
$coreUrl = "http://127.0.0.1:$corePort"

$environmentNames = @(
    'DRONEDREAM_CONSOLE_CORE_URL',
    'DRONEDREAM_CONSOLE_SUPABASE_URL',
    'DRONEDREAM_CONSOLE_LOCAL_TOKEN',
    'DRONEDREAM_CONSOLE_IDENTITY_TOKEN',
    'DRONEDREAM_CONSOLE_PUBLISHABLE_KEY',
    'DRONEDREAM_CONSOLE_EDITION',
    'PYTHONPATH',
    'NO_COLOR',
    'FORCE_COLOR'
)
$previousEnvironment = @{}
foreach ($name in $environmentNames) {
    $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}

$coreProcess = $null
$password = $null
$auth = $null
$localToken = $null

try {
    $keys = (& npx.cmd --yes supabase projects api-keys --project-ref $project --output json 2>$null | Out-String | ConvertFrom-Json)
    if ($LASTEXITCODE -ne 0) {
        throw 'Unable to read the authorized DroneDream model-gateway key.'
    }
    $publishableKey = ($keys | Where-Object { $_.name -in @('anon', 'publishable') } | Select-Object -First 1).api_key
    if (-not $publishableKey) {
        throw 'The DroneDream model-gateway key is unavailable.'
    }

    $credential = Import-Clixml -LiteralPath $credentialPath
    $password = $credential.GetNetworkCredential().Password
    $authBody = @{ email = $credential.UserName; password = $password } | ConvertTo-Json -Compress
    $auth = Invoke-RestMethod -Method Post -Uri "https://$project.supabase.co/auth/v1/token?grant_type=password" -Headers @{ apikey = $publishableKey } -ContentType 'application/json' -Body $authBody
    if (-not $auth.access_token) {
        throw 'DroneDream test-account login did not return an access token.'
    }

    $tokenBytes = New-Object byte[] 32
    $tokenGenerator = [Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $tokenGenerator.GetBytes($tokenBytes)
    }
    finally {
        $tokenGenerator.Dispose()
    }
    $localToken = -join ($tokenBytes | ForEach-Object { $_.ToString('x2') })

    $env:PYTHONPATH = Join-Path $sourceRoot 'src'
    $serverArguments = @(
        '-m', 'dronedream_agent_app.server',
        '--port', "$corePort",
        '--token', $localToken,
        '--data-root', (Join-Path $sessionRoot 'core-data'),
        '--resource-root', $resourceRoot,
        '--plugin-isolator', $pluginIsolator,
        '--account-memory-db', (Join-Path $sessionRoot 'account-memory.db')
    )
    $coreProcess = Start-Process -FilePath $python -ArgumentList $serverArguments -WorkingDirectory $sourceRoot -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -WindowStyle Hidden -PassThru

    $deadline = (Get-Date).AddSeconds(90)
    while ((Get-Date) -lt $deadline) {
        if ($coreProcess.HasExited) {
            throw "DroneDream Core exited during startup with code $($coreProcess.ExitCode)."
        }
        if (Get-NetTCPConnection -State Listen -LocalPort $corePort -ErrorAction SilentlyContinue) {
            break
        }
        Start-Sleep -Milliseconds 400
    }
    if (-not (Get-NetTCPConnection -State Listen -LocalPort $corePort -ErrorAction SilentlyContinue)) {
        throw 'DroneDream Core did not become ready within 90 seconds.'
    }

    $env:DRONEDREAM_CONSOLE_CORE_URL = $coreUrl
    $env:DRONEDREAM_CONSOLE_SUPABASE_URL = "https://$project.supabase.co"
    $env:DRONEDREAM_CONSOLE_LOCAL_TOKEN = $localToken
    $env:DRONEDREAM_CONSOLE_IDENTITY_TOKEN = $auth.access_token
    $env:DRONEDREAM_CONSOLE_PUBLISHABLE_KEY = $publishableKey
    $env:DRONEDREAM_CONSOLE_EDITION = 'autonomy'

    $doctorText = (& $cli --json doctor 2>&1 | Out-String)
    if ($LASTEXITCODE -notin @(0, 2)) {
        throw "DroneDream readiness check failed: $doctorText"
    }
    $doctor = $doctorText | ConvertFrom-Json
    if (-not $doctor.ready) {
        $null = (& $cli --json runtime provision 2>&1 | Out-String | ConvertFrom-Json)
        $doctor = (& $cli --json doctor 2>&1 | Out-String | ConvertFrom-Json)
    }
    if (-not $doctor.ready) {
        $blocked = ($doctor.checks | Where-Object { $_.status -ne 'ready' } | ForEach-Object { "$($_.name): $($_.detail)" }) -join '; '
        throw "DroneDream is not ready: $blocked"
    }

    if ($CheckOnly) {
        Write-Output 'READY'
    }
    else {
        Remove-Item Env:NO_COLOR -ErrorAction SilentlyContinue
        $env:FORCE_COLOR = '1'
        Clear-Host
        & $cli
    }
}
finally {
    $password = $null
    $auth = $null
    $localToken = $null
    if ($coreProcess -and -not $coreProcess.HasExited) {
        Stop-Process -Id $coreProcess.Id -Force -ErrorAction SilentlyContinue
        Wait-Process -Id $coreProcess.Id -Timeout 10 -ErrorAction SilentlyContinue
    }
    foreach ($name in $environmentNames) {
        [Environment]::SetEnvironmentVariable($name, $previousEnvironment[$name], 'Process')
    }
}
