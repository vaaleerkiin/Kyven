param([string]$CMake = "cmake", [string]$Generator = "Visual Studio 17 2022")

$ErrorActionPreference = "Stop"
$KyvenRoot = Split-Path -Parent $PSScriptRoot
$SourceRoot = Join-Path $KyvenRoot "native\dlss"
$BuildRoot = Join-Path $KyvenRoot ".runtime\kyven-dlss-build"
$PackageRoot = Join-Path $KyvenRoot "src\kyven\resources\dlss\windows_x64"

function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE." }
}

if ($CMake -eq "cmake" -and -not (Get-Command $CMake -ErrorAction SilentlyContinue)) {
    $VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $VsWhere) {
        $VsInstall = & $VsWhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($VsInstall) {
            $Candidate = Join-Path $VsInstall "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
            if (Test-Path -LiteralPath $Candidate) { $CMake = $Candidate }
        }
    }
}
if (-not (Get-Command $CMake -ErrorAction SilentlyContinue)) {
    throw "Build machine requires CMake and VS 2022 C++. End users install the bundled runtime from Model Manager."
}
Invoke-Checked $CMake @("-S", $SourceRoot, "-B", $BuildRoot, "-G", $Generator, "-A", "x64")
Invoke-Checked $CMake @("--build", $BuildRoot, "--config", "Release")
Invoke-Checked $CMake @("--install", $BuildRoot, "--config", "Release", "--prefix", $PackageRoot)
$Hashes = [ordered]@{}
foreach ($Name in @("KyvenDLSSWorker.exe", "nvngx.dll", "LICENSE.upstream")) {
    $Hashes[$Name] = (Get-FileHash -LiteralPath (Join-Path $PackageRoot $Name) -Algorithm SHA256).Hash.ToLowerInvariant()
}
$VendorPath = Join-Path $PackageRoot "nvngx_dlssnr.dll"
if (Test-Path -LiteralPath $VendorPath) {
    $VendorSpec = Get-Content -LiteralPath (Join-Path $PackageRoot "..\vendor-source.json") -Raw | ConvertFrom-Json
    $VendorHash = (Get-FileHash -LiteralPath $VendorPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($VendorHash -ne $VendorSpec.member_sha256 -or (Get-Item -LiteralPath $VendorPath).Length -ne $VendorSpec.member_size_bytes) {
        throw "Bundled NVIDIA component does not match the pinned source."
    }
    $Hashes["nvngx_dlssnr.dll"] = $VendorHash
}
$Manifest = [ordered]@{ version = 1; product = "KyvenDLSSWorker"; sha256 = $Hashes }
$Json = $Manifest | ConvertTo-Json -Depth 4
[System.IO.File]::WriteAllText((Join-Path $PackageRoot "manifest.json"), $Json, (New-Object System.Text.UTF8Encoding($false)))
Write-Host "Portable Kyven worker packaged at $PackageRoot"
Write-Host "Static MSVC runtime; no compiler is needed on the end-user machine. Bundled NR is included when present."
