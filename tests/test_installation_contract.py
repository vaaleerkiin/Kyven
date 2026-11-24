"""The documented ZIP install must not introduce an unadvertised VCS requirement."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _fill_setup():
    path = Path(__file__).parents[1] / "scripts/setup_generative_fill.py"
    spec = importlib.util.spec_from_file_location("fill_setup_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fill_setup_resolves_dependencies_but_never_resolves_upstream_git(tmp_path, monkeypatch):
    module = _fill_setup()
    root = tmp_path / "clean checkout with spaces \u0416"
    script = root / "scripts/setup_generative_fill.py"
    script.parent.mkdir(parents=True)
    script.touch()
    core_site = root / (".venv/Lib/site-packages" if os.name == "nt" else
                       f".venv/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
    (core_site / "torch").mkdir(parents=True)
    core_python = root / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    core_python.parent.mkdir(parents=True, exist_ok=True)
    core_python.touch()
    monkeypatch.setattr(module, "__file__", str(script))
    monkeypatch.setattr(sys, "prefix", str(root / ".venv"))
    monkeypatch.setattr(sys, "argv", [str(script)])
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "2.10.0" if name == "torch" else "0.25.0")

    class Builder:
        def __init__(self, **kwargs):
            pass

        def create(self, target):
            relative = core_site.relative_to(root / ".venv")
            (target / relative).mkdir(parents=True)

    calls = []
    monkeypatch.setattr(module.venv, "EnvBuilder", Builder)
    monkeypatch.setattr(module.subprocess, "run", lambda command, **kwargs: calls.append(command))
    module.main()
    normal, archives, probe = calls
    assert "--no-deps" not in normal and "--ignore-installed" not in normal
    assert "--constraint" in normal
    assert archives[-1].endswith("generative-fill-lanpaint.txt") and "--no-deps" in archives
    assert "-I" in probe and "splu" in probe[-1] and "FluxKleinAdapter" in probe[-1]
    pth = next((root / ".runtime").rglob("kyven_core_torch.pth"))
    code = pth.read_text(encoding="ascii")
    monkeypatch.setattr(sys, "path", list(sys.path))
    exec(code)  # noqa: S102 - exercise the installer's own generated .pth with a Unicode fixture
    assert sys.path[-1] == str(core_site)


def test_fill_check_only_verifies_without_installing(monkeypatch):
    module = _fill_setup()
    monkeypatch.setattr(sys, "argv", ["setup_generative_fill.py", "--check"])
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda command, **kwargs: calls.append(command))
    module.main()
    assert len(calls) == 1 and calls[0][1:3] == ["-I", "-c"]


@pytest.mark.skipif(os.name != "nt", reason="Windows installer argument parsing")
@pytest.mark.parametrize("selection,expected", [
    ("none", []), ("flux2-klein-4b-q4-lanpaint", ["flux2-klein-4b-q4-lanpaint"]),
    ("none,sam2.1-small", None), ("sam2.1-small,none", None),
])
def test_windows_installer_model_selection_is_noninteractive(tmp_path, selection, expected):
    script = tmp_path / "parse.ps1"
    script.write_text('''param([string]$Installer, [string]$Selection)
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Installer, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw $errors[0] }
$definition = $ast.FindAll({param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $node.Name -eq "Resolve-RequestedModels"
}, $true)[0]
Invoke-Expression $definition.Extent.Text
try { $result = @(Resolve-RequestedModels @($Selection)); ConvertTo-Json -InputObject $result -Compress }
catch { exit 2 }
''', encoding="utf-8")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-File", str(script),
        "-Installer", str(Path(__file__).parents[1] / "install.ps1"), "-Selection", selection],
        capture_output=True, text=True, timeout=20, check=False)
    if expected is None:
        assert result.returncode == 2
    else:
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == expected


def test_runtime_installer_and_requirements_have_no_git_only_dependencies():
    root = Path(__file__).parents[1]
    paths = [root/'install.ps1', *list((root/'requirements').glob('*.txt'))]
    for path in paths:
        text = path.read_text(encoding='utf-8').lower()
        assert 'git+https' not in text, f'{path.name} requires Git for ZIP users'
        assert 'tapnet' not in text, f'{path.name} installs an unused development package'
