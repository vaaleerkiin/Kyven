"""Manager setup works before a backend exists and never downloads unwanted weights."""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "hosts/nuke"))
from kyven_nuke import model_manager as manager
from kyven_nuke import runtime


class ImmediateThread:
    def __init__(self, target, **kwargs):
        self.target = target

    def start(self):
        self.target()


class FakeNuke:
    def __init__(self):
        self.messages = []

    def message(self, text):
        self.messages.append(text)

    def ProgressTask(self, title):
        return SimpleNamespace(setMessage=lambda text: None, setProgress=lambda value: None,
                               isCancelled=lambda: False)

    def executeInMainThread(self, function, args=()):
        return function(*args)

    executeInMainThreadWithResult = executeInMainThread

    def allNodes(self, kind):
        return []


@pytest.fixture
def offline(tmp_path, monkeypatch):
    root = Path(__file__).parents[1]
    catalog = tmp_path / "src/kyven/resources/models/catalog.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_bytes((root / "src/kyven/resources/models/catalog.json").read_bytes())
    monkeypatch.setattr(manager.config, "root", lambda: tmp_path)
    monkeypatch.setattr(manager.config, "python_executable", lambda: tmp_path / ".venv/Scripts/python.exe")
    monkeypatch.setattr(manager, "ensure_server", lambda: pytest.fail("Server cannot be required before setup"))
    return tmp_path


def test_manager_has_trusted_models_and_core_setup_without_python(offline):
    models, ready = manager._manager_models()
    assert not ready
    assert next(m for m in models if m["model_id"] == "kyven-core-runtime")["core_runtime"]
    fill = next(m for m in models if m["model_id"] == "flux2-klein-4b-q4-lanpaint")
    assert fill["runtime"] == "generative-fill" and fill["needs_core"]


def test_torch_models_declare_components_but_onnx_keeps_cpu_baseline():
    data = json.loads((Path(__file__).parents[1] / "src/kyven/resources/models/catalog.json").read_text())
    for model in data["models"]:
        if model["provider"] != "lama-onnx":
            assert model.get("runtime"), model["model_id"]
        else:
            assert not model.get("runtime")


def test_manager_uses_current_runtime_requirements_with_an_older_server(offline, monkeypatch):
    executable = manager.config.python_executable()
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setattr(manager, "ensure_server", lambda: SimpleNamespace(models=lambda: [
        {"model_id": "sam2.1-small", "runtime": "", "installed": True}]))
    models, ready = manager._manager_models()
    assert ready and models[0]["runtime"] == "core"


@pytest.mark.parametrize("needs_core", [False, True])
def test_components_only_installs_dependencies_without_model_download(monkeypatch, needs_core):
    nuke = FakeNuke()
    monkeypatch.setattr(manager, "_nuke", lambda: nuke)
    monkeypatch.setattr(manager, "threading", SimpleNamespace(Thread=ImmediateThread))
    monkeypatch.setattr(manager, "_refresh_all_nodes", lambda: None)
    calls = []
    monkeypatch.setattr(manager, "ensure_model_runtime", lambda name, **kwargs: calls.append(name) or False)
    monkeypatch.setattr(manager, "ensure_server", lambda: pytest.fail("Runtime-only must not download weights"))
    model = {"model_id": "fill", "display_name": "Generative Fill", "runtime": "generative-fill",
             "needs_core": needs_core}
    manager._start_operation("runtime", model)
    assert calls == (["core", "generative-fill"] if needs_core else ["generative-fill"])
    assert nuke.messages == ["Required components for Generative Fill are ready."]
    assert "fill" not in manager._tasks


def test_core_install_reopens_the_manager(monkeypatch):
    opened = []
    monkeypatch.setattr(manager, "_nuke", FakeNuke)
    monkeypatch.setattr(manager, "_refresh_all_nodes", lambda: None)
    monkeypatch.setattr(manager, "show_model_manager", lambda: opened.append(True))
    manager._finish_runtime("core", {"core_runtime": True, "display_name": "Base Runtime"})
    assert opened == [True]


def test_missing_base_is_installed_and_then_verified(offline, monkeypatch):
    checks = iter([False, True])
    monkeypatch.setattr(runtime, "_probe_runtime", lambda *args: next(checks))
    monkeypatch.setattr(runtime, "_wait_for_install_idle", lambda *args: False)
    calls = []
    monkeypatch.setattr(runtime, "_run_runtime_installer", lambda command, *a: calls.append(command))
    if os.name != "nt":
        pytest.skip("Windows bootstrap")
    assert runtime.ensure_core_runtime()
    assert calls[0][-2:] == ["-Model", "none"]
    assert str(offline / "install.ps1") in calls[0]


def test_installer_streams_progress_and_cancels_its_child(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.config, "root", lambda: tmp_path)
    monkeypatch.setattr(runtime.config, "python_executable", lambda: Path(sys.executable))
    messages = []
    started = time.monotonic()
    with pytest.raises(runtime.RuntimeInstallCancelled):
        runtime._run_runtime_installer([sys.executable, "-u", "-c",
            ("import subprocess,sys,time; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],stdout=sys.stdout); "
            "print('Installer child '+str(p.pid),flush=True); time.sleep(60)")],
            messages.append, lambda: bool(messages))
    assert messages and time.monotonic() - started < 10


def test_installer_waits_for_processing_to_finish(monkeypatch):
    states = iter([{"jobs": {"active_job_id": "work"}}, {"jobs": {"active_job_id": None}}])
    monkeypatch.setattr(runtime, "client", lambda **kwargs: SimpleNamespace(health=lambda: next(states)))
    monkeypatch.setattr(runtime.time, "sleep", lambda delay: None)
    messages = []
    assert runtime._wait_for_install_idle(messages.append, None)
    assert messages


def test_bootstrap_retains_external_python_but_excludes_nuke_paths(tmp_path, monkeypatch):
    host = tmp_path / "Nuke/Nuke16.exe"
    python = tmp_path / "Python312"
    monkeypatch.setattr(sys, "executable", str(host))
    monkeypatch.setenv("PATH", os.pathsep.join([str(host.parent), str(host.parent / "plugins"), str(python)]))
    monkeypatch.setenv("PYTHONHOME", "Nuke")
    monkeypatch.setattr(runtime.config, "root", lambda: tmp_path)
    monkeypatch.setattr(runtime.config, "python_executable", lambda: tmp_path / ".venv/python.exe")
    captured = {}

    def popen(command, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(stdout=io.StringIO("Ready\n"), poll=lambda: 0,
                               wait=lambda **kw: 0, returncode=0)

    monkeypatch.setattr(runtime.subprocess, "Popen", popen)
    runtime._run_runtime_installer(["powershell.exe", "-File", "install.ps1"])
    path = captured["env"]["PATH"].split(os.pathsep)
    assert str(python) in path and str(host.parent) not in path
    assert str(host.parent / "plugins") not in path
    assert "PYTHONHOME" not in captured["env"]


@pytest.mark.skipif(os.name != "nt", reason="Windows bootstrap process integration")
def test_manager_bootstraps_components_through_powershell_with_no_backend(tmp_path, monkeypatch):
    root = tmp_path / "checkout with spaces \u0416"
    root.mkdir()
    (root / "install.ps1").write_text('''param([string]$Model)
if ($Model -ne "none") { throw "Bootstrap must not download model weights" }
Write-Output "Installing test runtime"
New-Item -ItemType File -Path (Join-Path $PSScriptRoot "verified-runtime") -Force | Out-Null
''', encoding="utf-8")
    monkeypatch.setattr(manager.config, "root", lambda: root)
    monkeypatch.setattr(manager.config, "python_executable", lambda: root / ".venv/Scripts/python.exe")
    monkeypatch.setattr(runtime, "_probe_runtime", lambda *a: (root / "verified-runtime").is_file())
    monkeypatch.setattr(runtime, "_wait_for_install_idle", lambda *a: False)
    monkeypatch.setattr(manager, "stop_server", lambda: None)
    monkeypatch.setattr(manager, "threading", SimpleNamespace(Thread=ImmediateThread))
    nuke = FakeNuke()
    monkeypatch.setattr(manager, "_nuke", lambda: nuke)
    monkeypatch.setattr(manager, "_refresh_all_nodes", lambda: None)
    reopened = []
    monkeypatch.setattr(manager, "show_model_manager", lambda: reopened.append(True))
    manager._start_operation("runtime", {"model_id": "kyven-core-runtime", "runtime": "core",
        "core_runtime": True, "display_name": "Kyven Base Runtime"})
    assert (root / "verified-runtime").is_file()
    assert reopened == [True]
    assert nuke.messages == ["Required components for Kyven Base Runtime are ready."]


def test_refresh_repairs_fill_without_replacing_its_model_with_lama(monkeypatch):
    from kyven_nuke import generative_fill_node

    class Group(dict):
        def knobs(self):
            return self

    overwritten = []
    node = Group(qwen_system_prompt=object(), kyven_kind=SimpleNamespace(value=lambda: "inpaint"),
        model=SimpleNamespace(getValue=lambda: 0, setValue=lambda value: None,
                              setValues=overwritten.append))
    monkeypatch.setattr(manager, "_nuke", lambda: SimpleNamespace(allNodes=lambda kind: [node]))
    monkeypatch.setattr(manager, "ensure_server", lambda: SimpleNamespace(models=lambda: [{
        "model_id": "big-lama-native", "display_name": "LaMa", "task": "inpaint", "size_bytes": 1}]))
    repaired = []
    monkeypatch.setattr(generative_fill_node, "_ensure_fill_graph", repaired.append)
    manager._refresh_all_nodes()
    assert repaired == [node] and not overwritten
