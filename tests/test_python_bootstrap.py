"""Windows bootstrap must work without a Python command and verify before extraction."""
import hashlib
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows Python bootstrap")
ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("mode", ["download", "cached", "corrupt", "interrupted", "store-aliases"])
def test_private_python_bootstrap_without_system_python(tmp_path, mode):
    root = tmp_path / "clean checkout with spaces \u0416"
    (root / "requirements").mkdir(parents=True)
    archive = tmp_path / "fixture.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        content = b"test interpreter fixture"
        info = tarfile.TarInfo("python/python.exe")
        info.size = len(content)
        output.addfile(info, io.BytesIO(content))
    manifest = json.loads((ROOT / "requirements/python-windows-x64.json").read_text())
    manifest.update(size_bytes=archive.stat().st_size,
                    sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (root / "requirements/python-windows-x64.json").write_text(json.dumps(manifest))
    if mode in {"cached", "corrupt", "interrupted"}:
        cache = root / ".runtime/python/cpython-3.12.16-20261110.tar.gz"
        cache.parent.mkdir(parents=True)
        cache.write_bytes(b"broken download" if mode == "corrupt" else archive.read_bytes())
    if mode == "interrupted":
        partial = root / ".runtime/python/cpython-3.12.16-20261110"
        partial.mkdir()
        (partial / "incomplete.txt").write_text("Incomplete previous installation")
    harness = tmp_path / "bootstrap.ps1"
    harness.write_text('''param($Root, $Bootstrap, $Installer, $FixtureArchive, $Mode)
$ErrorActionPreference = "Stop"
. $Bootstrap
# Test the real find/fallback function without running the heavyweight pip setup.
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Installer, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw $errors[0] }
$definition = $ast.FindAll({param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $node.Name -eq "Find-BootstrapPython"
}, $true)[0]
Invoke-Expression $definition.Extent.Text
$KyvenRoot = $Root; $PythonExe = ""; $script:downloads = 0
$env:PATH = "$env:SystemRoot\\System32;$env:SystemRoot\\System32\\WindowsPowerShell\\v1.0"
if ($Mode -eq "store-aliases") {
    function Get-Command($Name, $ErrorAction) {
        return [pscustomobject]@{Source="$env:LOCALAPPDATA\\Microsoft\\WindowsApps\\$Name"}
    }
    function Test-Python { throw "Must not execute Windows Store aliases" }
}
function Test-KyvenManagedPython($Executable) { return Test-Path -LiteralPath $Executable -PathType Leaf }
function Invoke-WebRequest($Uri, $OutFile, [switch]$UseBasicParsing) {
    $script:downloads++
    if ($Mode -eq "corrupt") { [IO.File]::WriteAllText($OutFile, "untrusted") }
    else { Copy-Item -LiteralPath $FixtureArchive -Destination $OutFile }
}
if ($Mode -eq "corrupt") {
    try { $result = Find-BootstrapPython; throw "Accepted corrupt archive" }
    catch {
        if ($_ -notmatch "SHA-256") { throw }
        if (Test-Path -LiteralPath (Get-KyvenManagedPythonPath $Root)) { throw "Published unverified Python" }
        if (Get-ChildItem -LiteralPath (Join-Path $Root ".runtime/python") -Filter "staging-*") { throw "Extracted unverified archive" }
    }
} else {
    $result = Find-BootstrapPython
    if (-not (Test-Path -LiteralPath $result.Executable)) { throw "Python not published" }
    $again = Find-BootstrapPython
    if ($again.Executable -ne $result.Executable) { throw "Did not reuse managed Python" }
}
ConvertTo-Json @{downloads=$script:downloads; ok=$true} -Compress
''', encoding="utf-8-sig")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-File", str(harness),
        "-Root", str(root), "-Bootstrap", str(ROOT / "scripts/bootstrap_python.ps1"),
        "-Installer", str(ROOT / "install.ps1"), "-FixtureArchive", str(archive), "-Mode", mode],
        capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report == {"ok": True, "downloads": {
        "cached": 0, "download": 1, "corrupt": 3, "interrupted": 0, "store-aliases": 1}[mode]}
    if mode == "interrupted":
        assert len(list((root / ".runtime/python").glob("*.incomplete-*/incomplete.txt"))) == 1
