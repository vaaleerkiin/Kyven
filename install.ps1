[CmdletBinding()]
param(
    [string[]]$Model = @(),

    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$KyvenRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$VenvRoot = Join-Path $KyvenRoot ".venv"
$VenvPython = Join-Path $VenvRoot "Scripts\python.exe"
$Requirements = Join-Path $KyvenRoot "requirements\runtime-cu128.txt"
$ModelsRoot = Join-Path $KyvenRoot "models"
$RuntimeRoot = Join-Path $KyvenRoot ".runtime"
$PipCache = Join-Path $RuntimeRoot "pip-cache"

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-Native([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $Executable $($Arguments -join ' ')"
    }
}

function Test-Python([string]$Executable, [string[]]$PrefixArguments) {
    try {
        & $Executable @PrefixArguments -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)" 2>$null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Find-BootstrapPython {
    if ($PythonExe) {
        $ResolvedPython = (Resolve-Path -LiteralPath $PythonExe).Path
        if (-not (Test-Python $ResolvedPython @())) {
            throw "PythonExe must point to Python 3.10-3.13: $ResolvedPython"
        }
        return [pscustomobject]@{
            Executable = $ResolvedPython
            Prefix = @()
        }
    }

    $PyLauncher = Get-Command "py.exe" -ErrorAction SilentlyContinue
    if ($PyLauncher -and (Test-Python $PyLauncher.Source @("-3.12"))) {
        return [pscustomobject]@{
            Executable = $PyLauncher.Source
            Prefix = @("-3.12")
        }
    }

    $PythonCommand = Get-Command "python.exe" -ErrorAction SilentlyContinue
    if ($PythonCommand -and (Test-Python $PythonCommand.Source @())) {
        return [pscustomobject]@{
            Executable = $PythonCommand.Source
            Prefix = @()
        }
    }

    throw @"
Python 3.10-3.13 was not found.
Install Python 3.12 from python.org, enable the Python launcher, and run this script again.
You can also pass an explicit interpreter:
  .\install.ps1 -PythonExe "C:\Path\To\python.exe"
"@
}

function Select-Models {
    $Choices = @(
        [pscustomobject]@{ Number = "1"; Id = "sam2.1-tiny"; Label = "SAM 2.1 Tiny"; Guidance = "4 GB VRAM minimum" },
        [pscustomobject]@{ Number = "2"; Id = "sam2.1-small"; Label = "SAM 2.1 Small"; Guidance = "6 GB, recommended for 8 GB" },
        [pscustomobject]@{ Number = "3"; Id = "sam2.1-base-plus"; Label = "SAM 2.1 Base+"; Guidance = "8-12 GB VRAM" },
        [pscustomobject]@{ Number = "4"; Id = "sam2.1-large"; Label = "SAM 2.1 Large"; Guidance = "12+ GB VRAM" },
        [pscustomobject]@{ Number = "5"; Id = "vitmatte-small-composition-1k"; Label = "ViTMatte Small"; Guidance = "4 GB+, refinement" },
        [pscustomobject]@{ Number = "6"; Id = "lama-2025jan-onnx"; Label = "LaMa ONNX Fast"; Guidance = "CPU / Live, fixed 512 input" },
        [pscustomobject]@{ Number = "7"; Id = "big-lama-native"; Label = "Big-LaMa Native"; Guidance = "best detail, native ROI, 4 GB+" },
        [pscustomobject]@{ Number = "0"; Id = "none"; Label = "No model"; Guidance = "install runtime only" }
    )

    Write-Host ""
    Write-Host "Select one or more models to install" -ForegroundColor Yellow
    foreach ($Choice in $Choices) {
        Write-Host "  $($Choice.Number)) $($Choice.Label) - $($Choice.Guidance)"
    }
    Write-Host ""
    $Answer = Read-Host "Enter numbers separated by commas [2,5]"
    if ([string]::IsNullOrWhiteSpace($Answer)) {
        $Answer = "2,5"
    }

    $Selected = @()
    foreach ($Part in ($Answer -split "[,; ]+")) {
        if ([string]::IsNullOrWhiteSpace($Part)) {
            continue
        }
        $Match = $Choices | Where-Object Number -eq $Part
        if (-not $Match) {
            throw "Unknown model choice '$Part'. Run the installer again and enter 0-7."
        }
        if ($Match.Id -eq "none") {
            $AnswerParts = @($Answer -split "[,; ]+" | Where-Object { $_ })
            if ($AnswerParts.Count -gt 1) {
                throw "Choice 0 (no model) cannot be combined with another model."
            }
            return @()
        }
        if ($Selected -notcontains $Match.Id) {
            $Selected += $Match.Id
        }
    }
    return $Selected
}

function Resolve-RequestedModels([string[]]$RequestedModels) {
    $Allowed = @(
        "sam2.1-tiny",
        "sam2.1-small",
        "sam2.1-base-plus",
        "sam2.1-large",
        "vitmatte-small-composition-1k",
        "lama-2025jan-onnx",
        "big-lama-native"
    )
    if (-not $RequestedModels -or $RequestedModels.Count -eq 0) {
        return @(Select-Models)
    }

    $Resolved = @()
    foreach ($Requested in $RequestedModels) {
        foreach ($Item in ($Requested -split "[,; ]+")) {
            if ([string]::IsNullOrWhiteSpace($Item)) {
                continue
            }
            if ($Item -eq "none") {
                if ($RequestedModels.Count -gt 1 -or $Resolved.Count -gt 0) {
                    throw "Model 'none' cannot be combined with another model."
                }
                return @()
            }
            if ($Allowed -notcontains $Item) {
                throw "Unknown model '$Item'. Allowed values: $($Allowed -join ', '), none."
            }
            if ($Resolved -notcontains $Item) {
                $Resolved += $Item
            }
        }
    }
    return $Resolved
}

function Stop-LocalKyvenLaunchers {
    $LauncherPath = (Join-Path $VenvRoot "Scripts\kyven.exe")
    $Launchers = @(Get-Process -Name "kyven" -ErrorAction SilentlyContinue | Where-Object {
        try {
            $_.Path -and ([string]::Equals($_.Path, $LauncherPath, [System.StringComparison]::OrdinalIgnoreCase))
        }
        catch {
            $false
        }
    })
    if ($Launchers.Count -eq 0) {
        return
    }

    Write-Step "Stopping the Kyven server from this repository before updating"
    $Launchers | Stop-Process -Force
    $Launchers | Wait-Process -Timeout 10 -ErrorAction SilentlyContinue
}

Write-Host "Kyven portable installer" -ForegroundColor White
Write-Host "Repository: $KyvenRoot"
Write-Host "Nothing will be installed outside this repository."
Write-Host "Nuke init.py will not be modified."

$SelectedModels = @(Resolve-RequestedModels $Model)
if ($SelectedModels.Count -gt 0) {
    Write-Host "Models: $($SelectedModels -join ', ')"
}
else {
    Write-Host "Models: none"
}

if (-not (Test-Path -LiteralPath $Requirements -PathType Leaf)) {
    throw "Runtime requirements file was not found: $Requirements"
}

New-Item -ItemType Directory -Force -Path $RuntimeRoot, $PipCache, $ModelsRoot | Out-Null
$env:PIP_CACHE_DIR = $PipCache
$env:PYTHONNOUSERSITE = "1"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    Write-Step "Creating the private Python environment"
    $Bootstrap = Find-BootstrapPython
    Invoke-Native $Bootstrap.Executable ($Bootstrap.Prefix + @("-m", "venv", $VenvRoot))
}
else {
    Write-Step "Reusing the existing private Python environment"
}

if (-not (Test-Python $VenvPython @())) {
    throw "The repository .venv is not a supported Python environment: $VenvPython"
}

Push-Location $KyvenRoot
try {
    Stop-LocalKyvenLaunchers

    Write-Step "Updating packaging tools inside .venv"
    Invoke-Native $VenvPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools")

    Write-Step "Installing PyTorch and SAM 2 inside .venv"
    Invoke-Native $VenvPython @("-m", "pip", "install", "-r", $Requirements)

    Write-Step "Installing Kyven inside .venv"
    Invoke-Native $VenvPython @("-m", "pip", "install", "--upgrade", ".")

    foreach ($SelectedModel in $SelectedModels) {
        Write-Step "Downloading and verifying $SelectedModel"
        Invoke-Native $VenvPython @(
            "-m", "kyven.cli", "models", "download", $SelectedModel,
            "--models-dir", $ModelsRoot
        )
    }
    if ($SelectedModels.Count -eq 0) {
        Write-Step "Skipping model download"
    }

    Write-Step "Checking the installed runtime"
    Invoke-Native $VenvPython @(
        "-c",
        "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
    )
    Invoke-Native $VenvPython @(
        "-m", "kyven.cli", "models", "list", "--models-dir", $ModelsRoot
    )
}
finally {
    Pop-Location
}

$PluginPath = (Join-Path $KyvenRoot "hosts\nuke").Replace("\", "/")
$SuggestedInit = Join-Path $env:USERPROFILE ".nuke\init.py"

Write-Host ""
Write-Host "Kyven portable installation completed." -ForegroundColor Green
Write-Host ""
Write-Host "Manually add these lines to:" -ForegroundColor Yellow
Write-Host "  $SuggestedInit"
Write-Host ""
Write-Host "import nuke" -ForegroundColor White
Write-Host "nuke.pluginAddPath(`"$PluginPath`")" -ForegroundColor White
Write-Host ""
Write-Host "Then restart Nuke and choose Kyven > Segment."
Write-Host "To update later: pull the repository and run install.ps1 again."
Write-Host "To install more models, rerun the script and select them from the menu."
