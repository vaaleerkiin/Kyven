# External PowerShell bootstrap: no existing Python or administrator rights needed.
function Get-KyvenFileSHA256([string]$Path) {
    $Stream = [IO.File]::OpenRead($Path)
    $Hasher = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($Hasher.ComputeHash($Stream)).Replace("-", "").ToLowerInvariant() }
    finally { $Hasher.Dispose(); $Stream.Dispose() }
}

function Test-KyvenManagedPython([string]$Executable) {
    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { return $false }
    try {
        & $Executable -I -c "import sys, struct, venv, ensurepip; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and struct.calcsize('P') == 8 else 1)" 2>$null
        return $LASTEXITCODE -eq 0
    }
    catch { return $false }
}

function Get-KyvenManagedPythonPath([string]$Root) {
    return Join-Path $Root ".runtime\python\cpython-3.12.16-20261110\python.exe"
}

function Install-KyvenManagedPython([string]$Root) {
    $Manifest = Get-Content -LiteralPath (Join-Path $Root "requirements\python-windows-x64.json") -Raw | ConvertFrom-Json
    $Python = Get-KyvenManagedPythonPath $Root
    if (Test-KyvenManagedPython $Python) { return $Python }
    if (-not [Environment]::Is64BitOperatingSystem) { throw "Kyven requires 64-bit Windows." }
    $Tar = Join-Path $env:SystemRoot "System32\tar.exe"
    if (-not (Test-Path -LiteralPath $Tar -PathType Leaf)) { throw "Windows tar.exe was not found. Update Windows 10/11 before installing Kyven." }
    $Parent = Join-Path $Root ".runtime\python"
    New-Item -ItemType Directory -Force -Path $Parent | Out-Null
    $Archive = Join-Path $Parent "cpython-$($Manifest.version)-$($Manifest.build).tar.gz"
    $Valid = $false
    if (Test-Path -LiteralPath $Archive -PathType Leaf) {
        $Valid = ((Get-Item -LiteralPath $Archive).Length -eq $Manifest.size_bytes) -and
            ((Get-KyvenFileSHA256 $Archive) -eq $Manifest.sha256)
    }
    if (-not $Valid) {
        Write-Host "Downloading private Python $($Manifest.version) (22 MB)..."
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        $Partial = "$Archive.partial"
        for ($Attempt = 1; $Attempt -le 3; $Attempt++) {
            try {
                Invoke-WebRequest -UseBasicParsing -Uri $Manifest.url -OutFile $Partial
                if ((Get-Item -LiteralPath $Partial).Length -ne $Manifest.size_bytes -or
                    (Get-KyvenFileSHA256 $Partial) -ne $Manifest.sha256) {
                    throw "Python archive size or SHA-256 verification failed."
                }
                Move-Item -LiteralPath $Partial -Destination $Archive -Force
                break
            }
            catch {
                if ($Attempt -eq 3) { throw "Could not download verified Python: $_" }
                Write-Host "Python download failed; retrying ($Attempt/3)..."
            }
        }
    }
    # Extract to a unique staging directory. Publish only a verified interpreter;
    # cancellation cannot make a partial installation look ready on the next run.
    $Stage = Join-Path $Parent ("staging-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $Stage | Out-Null
    Write-Host "Extracting and verifying private Python..."
    # Windows tar may encode absolute argv paths using the active ANSI code page.
    # Inherit the Unicode working directory and pass only an ASCII relative path.
    Push-Location -LiteralPath $Stage
    try {
        & $Tar -xzf ("..\" + (Split-Path -Leaf $Archive))
        if ($LASTEXITCODE -ne 0) { throw "Could not extract the Python archive." }
    }
    finally { Pop-Location }
    $Extracted = Join-Path $Stage "python"
    if (-not (Test-KyvenManagedPython (Join-Path $Extracted "python.exe"))) {
        throw "Downloaded Python failed its interpreter/venv verification."
    }
    $Destination = Split-Path -Parent $Python
    $Backup = "$Destination.incomplete-" + [guid]::NewGuid().ToString("N")
    $AllowedPrefix = (Resolve-Path -LiteralPath $Parent).Path.TrimEnd('\') + '\'
    foreach ($MovePath in @($Extracted, $Destination, $Backup)) {
        if (-not [IO.Path]::GetFullPath($MovePath).StartsWith($AllowedPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Python installation path is outside the managed runtime directory."
        }
    }
    if (Test-Path -LiteralPath $Destination) {
        Move-Item -LiteralPath $Destination -Destination $Backup
    }
    Move-Item -LiteralPath $Extracted -Destination $Destination
    if (-not (Test-KyvenManagedPython $Python)) { throw "Private Python failed verification after installation." }
    Write-Host "Private Python is ready."
    return $Python
}
