from __future__ import annotations

import hashlib
import io
import json
import os
import struct
import subprocess
import threading
import zipfile
from pathlib import Path

import pytest

from kyven.dlss import runtime
from kyven.errors import ErrorCode, KyvenError


@pytest.fixture
def package(tmp_path, monkeypatch):
    folder = tmp_path / "package"
    folder.mkdir()
    for name in runtime.OWN_FILES:
        (folder / name).write_bytes(b"synthetic redistributable worker fixture: " + name.encode())
    hashes = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
              for name in runtime.OWN_FILES}
    (folder / "manifest.json").write_text(json.dumps({"sha256": hashes}))
    monkeypatch.setattr(runtime, "package_root", lambda: folder)
    return folder


def test_clean_machine_installs_owned_worker_and_reports_missing_vendor(tmp_path, package):
    root = tmp_path / "clean" / ".runtime"
    with pytest.raises(KyvenError) as error:
        runtime.bind_runtime(root)
    assert error.value.code == ErrorCode.PROVIDER_UNAVAILABLE
    assert (root / "dlss/bin/KyvenDLSSWorker.exe").is_file()
    assert runtime.resolve_worker(root) is None
    status = runtime.runtime_snapshot(root)
    assert not status["installed"]
    assert status["status_message"] == "NVIDIA NR component required"


def test_runtime_does_not_search_home_or_another_plugin(tmp_path, monkeypatch):
    def forbidden():
        raise AssertionError("A managed installation must never inspect another plugin")
    monkeypatch.setattr(Path, "home", forbidden)
    external = tmp_path / "foreign" / "DLSS_Nuke_Worker.exe"
    external.parent.mkdir()
    external.write_bytes(b"foreign executable")
    assert runtime.resolve_worker(tmp_path / "clean", preferred=str(external)) is None


def test_install_uses_kyven_worker_even_when_vendor_component_is_imported(tmp_path, package):
    vendor = tmp_path / runtime.VENDOR_FILE
    vendor.write_bytes(b"synthetic vendor fixture; never executed")
    root = tmp_path / "clean"
    worker = runtime.bind_runtime(root, source=vendor)
    assert worker.name == runtime.WORKER_NAME
    assert worker.parent == (root / "dlss/bin").resolve()
    assert worker.read_bytes() == (package / runtime.WORKER_NAME).read_bytes()
    assert (worker.parent / runtime.VENDOR_FILE).read_bytes() == vendor.read_bytes()
    assert vendor.is_file()


def test_model_manager_reports_old_worker_as_update_required(tmp_path, package):
    vendor = tmp_path / runtime.VENDOR_FILE
    vendor.write_bytes(b"synthetic vendor fixture")
    root = tmp_path / "managed"
    worker = runtime.bind_runtime(root, source=vendor)
    worker.write_bytes(b"older Kyven worker without RGBA guide support")
    status = runtime.runtime_snapshot(root)
    assert status["update_available"]
    assert not status["installed"]
    assert not status["supports_control_mask"]
    assert status["status_message"] == "Update Kyven runtime"


def test_altered_package_is_rejected_before_install(tmp_path, package):
    (package / runtime.WORKER_NAME).write_bytes(b"tampered")
    with pytest.raises(KyvenError) as error:
        runtime.bind_runtime(tmp_path / "clean")
    assert error.value.code == ErrorCode.DEPENDENCY_MISSING
    assert not (tmp_path / "clean/dlss/bin/KyvenDLSSWorker.exe").exists()


def test_runtime_installation_is_cancellable_before_publication(tmp_path, package):
    with pytest.raises(KyvenError) as error:
        runtime.bind_runtime(tmp_path / "clean", cancelled=lambda: True)
    assert error.value.code == ErrorCode.CANCELLED
    assert not (tmp_path / "clean/dlss/bin/KyvenDLSSWorker.exe").exists()


def test_cannot_import_another_apps_worker(tmp_path, package):
    other = tmp_path / "DLSS_Nuke_Worker.exe"
    other.write_bytes(b"other app")
    with pytest.raises(KyvenError) as error:
        runtime.bind_runtime(tmp_path / "clean", source=other)
    assert error.value.code == ErrorCode.INVALID_REQUEST


def test_uninstall_preserves_supplied_component(tmp_path, package):
    vendor = tmp_path / runtime.VENDOR_FILE
    vendor.write_bytes(b"synthetic vendor")
    root = tmp_path / "clean"
    runtime.bind_runtime(root, vendor)
    runtime.unbind_runtime(root)
    assert runtime.resolve_worker(root) is None
    assert vendor.is_file()
    assert (root / "dlss/bin" / runtime.VENDOR_FILE).is_file()


def test_nuke_dependency_installer_supports_exr_only_in_private_python():
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1] / "hosts/nuke"))
    from kyven_nuke.runtime import _runtime_install_command, _runtime_probe

    executable = Path("D:/Kyven/.venv/Scripts/python.exe")
    command = _runtime_install_command("dlss", executable)
    assert command[0] == str(executable)
    assert command[-1] == "OpenEXR==3.4.14"
    assert "OpenEXR" in _runtime_probe("dlss")


@pytest.fixture
def vendor_archive(monkeypatch):
    data = b"synthetic NVIDIA NR fixture, never executed"
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("runtime/nvngx_dlssnr.dll", data)
        archive.writestr("runtime/DLSS_Nuke_Worker.exe", b"foreign worker: must never be installed")
        archive.writestr("../escape.dll", b"must not be extracted")
    content = stream.getvalue()
    spec = {"url": "https://example.invalid/pinned.zip", "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(), "member": "runtime/nvngx_dlssnr.dll",
            "member_size_bytes": len(data), "member_sha256": hashlib.sha256(data).hexdigest()}
    monkeypatch.setattr(runtime, "vendor_source", lambda: spec)
    monkeypatch.setattr(runtime.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(content))
    return content, data, spec


def test_automatic_install_extracts_only_verified_vendor_member(tmp_path, package, vendor_archive):
    root = tmp_path / "clean"
    progress = []
    worker = runtime.bind_runtime(root, download_vendor=True,
                                 progress=lambda done, total: progress.append((done, total)))
    assert worker.name == runtime.WORKER_NAME
    assert (worker.parent / runtime.VENDOR_FILE).read_bytes() == vendor_archive[1]
    assert not (worker.parent / "DLSS_Nuke_Worker.exe").exists()
    assert not (root / "escape.dll").exists()
    assert progress[-1][0] == progress[-1][1]
    assert not list(worker.parent.glob(".nr-package-*"))


@pytest.mark.parametrize("field", ["sha256", "member_sha256", "size_bytes", "member_size_bytes"])
def test_archive_and_component_tampering_are_rejected(tmp_path, package, vendor_archive, field):
    spec = vendor_archive[2]
    spec[field] = "0" * 64 if "sha256" in field else spec[field] + 1
    with pytest.raises(KyvenError):
        runtime.bind_runtime(tmp_path / "clean", download_vendor=True)
    assert runtime.resolve_worker(tmp_path / "clean") is None
    assert not (tmp_path / "clean/dlss/bin" / runtime.VENDOR_FILE).exists()


def test_cancellation_interrupts_vendor_download_without_publishing(tmp_path, vendor_archive, monkeypatch):
    cancelled = threading.Event()
    class Response(io.BytesIO):
        def read(self, count=-1):
            data = super().read(count)
            cancelled.set()
            return data
    monkeypatch.setattr(runtime.urllib.request, "urlopen",
                        lambda *args, **kwargs: Response(vendor_archive[0]))
    with pytest.raises(KyvenError) as error:
        runtime._download_vendor(tmp_path, cancelled=cancelled.is_set)
    assert error.value.code == ErrorCode.CANCELLED
    assert not (tmp_path / runtime.VENDOR_FILE).exists()
    assert not list(tmp_path.glob(".nr-package-*"))


def test_bundled_component_installs_offline_and_is_verified(tmp_path, package, monkeypatch):
    data = b"synthetic bundled NVIDIA component"
    digest = hashlib.sha256(data).hexdigest()
    (package / runtime.VENDOR_FILE).write_bytes(data)
    manifest = json.loads((package / "manifest.json").read_text())
    manifest["sha256"][runtime.VENDOR_FILE] = digest
    (package / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(runtime, "vendor_source", lambda: {
        "member_size_bytes": len(data), "member_sha256": digest})
    def forbidden(*args, **kwargs):
        raise AssertionError("A bundled installation must not access the network")
    monkeypatch.setattr(runtime.urllib.request, "urlopen", forbidden)
    assert runtime.bundled_vendor_available()
    worker = runtime.bind_runtime(tmp_path / "offline", download_vendor=True)
    assert (worker.parent / runtime.VENDOR_FILE).read_bytes() == data


def test_corrupt_bundled_component_is_not_published_or_downloaded(tmp_path, package, monkeypatch):
    data = b"expected component"
    digest = hashlib.sha256(data).hexdigest()
    (package / runtime.VENDOR_FILE).write_bytes(b"incorrect contents")
    manifest = json.loads((package / "manifest.json").read_text())
    manifest["sha256"][runtime.VENDOR_FILE] = digest
    (package / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(runtime, "vendor_source", lambda: {
        "member_size_bytes": len(data), "member_sha256": digest})
    with pytest.raises(KyvenError) as error:
        runtime.bind_runtime(tmp_path / "offline", download_vendor=True)
    assert error.value.code == ErrorCode.DEPENDENCY_MISSING
    assert not (tmp_path / "offline/dlss/bin" / runtime.VENDOR_FILE).exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows native worker")
def test_native_worker_starts_without_cpp_redist_and_rejects_internal_flow():
    package = Path(runtime.__file__).parents[1] / "resources/dlss/windows_x64"
    executable = package / runtime.WORKER_NAME
    if not executable.is_file():
        pytest.skip("Worker has not been built on this development checkout")
    result = subprocess.run([str(executable), "--version"], capture_output=True, check=True)
    assert b"KyvenDLSSWorker" in result.stdout
    values = [0x35563544, 64, 64, 64, 64, 0, 100000, 0, 0, 0, 0, 0, 0, 0,
              1.0, 1.0, 1.0, -1.0, 2, 0, 0, 0, 0, 0.0]
    process = subprocess.run([str(executable)], input=struct.pack("<14I4f5If", *values),
                             capture_output=True, timeout=5, check=False)
    assert process.returncode == 1
    setup = struct.unpack("<12I", process.stdout)
    assert setup[0] == 0x34505553 and setup[1] == 0 and setup[2] == 0xBAD00001
