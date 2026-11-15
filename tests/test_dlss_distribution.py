from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import zipfile
from pathlib import Path

import pytest


def test_distribution_contains_managed_worker_and_installer_but_no_workstation_state(tmp_path):
    root = Path(__file__).parents[1]
    worker = root / "src/kyven/resources/dlss/windows_x64/KyvenDLSSWorker.exe"
    if not worker.is_file():
        pytest.skip("Build the native worker package before the distribution check")
    waft_models = root / "plugins/Cattery/WAFT"
    if any(not (waft_models / name).is_file() or (waft_models / name).stat().st_size < 1024
           for name in ("WAFT.cat", "WAFT_FP32.cat")):
        pytest.skip("Run git lfs pull before the complete distribution check")
    spec = importlib.util.spec_from_file_location("kyven_release", root / "scripts/package_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = module.package_release(root, tmp_path / "Kyven.zip")
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert "Kyven/install.cmd" in names
        assert "Kyven/hosts/nuke/kyven_nuke/dlss_node.py" in names
        assert "Kyven/src/kyven/resources/dlss/vendor-source.json" in names
        waft = "Kyven/plugins/Cattery/WAFT/"
        assert waft + "WAFT.gizmo" in names
        assert waft + "README.md" in names
        models = json.loads(archive.read(waft + "model-manifest.json"))["models"]
        for name, expected in models.items():
            assert archive.getinfo(waft + name).file_size == expected["size_bytes"]
            with archive.open(waft + name) as stream:
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            assert digest.hexdigest() == expected["sha256"]
        assert not any(name.startswith("Kyven/development/") for name in names)
        binary = "Kyven/src/kyven/resources/dlss/windows_x64/"
        manifest = json.loads(archive.read(binary + "manifest.json"))
        for name, expected in manifest["sha256"].items():
            assert hashlib.sha256(archive.read(binary + name)).hexdigest() == expected
        assert not any("/.runtime/" in name or "/.venv/" in name or "/models/" == name
                       for name in names)
        vendor = root / "src/kyven/resources/dlss/windows_x64/nvngx_dlssnr.dll"
        if vendor.is_file():
            source = json.loads(archive.read("Kyven/src/kyven/resources/dlss/vendor-source.json"))
            info = archive.getinfo(binary + "nvngx_dlssnr.dll")
            assert info.file_size == source["member_size_bytes"]
            assert manifest["sha256"]["nvngx_dlssnr.dll"] == source["member_sha256"]


def test_source_only_distribution_omits_unavailable_vendor_from_archive_manifest(tmp_path):
    original = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location("kyven_release", original / "scripts/package_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "source"
    package = root / "src/kyven/resources/dlss/windows_x64"
    package.mkdir(parents=True)
    for name in module.ROOT_FILES:
        (root / name).write_text("source checkout fixture", encoding="utf-8")
    for name in ("KyvenDLSSWorker.exe", "nvngx.dll", "LICENSE.upstream", "manifest.json"):
        source = original / "src/kyven/resources/dlss/windows_x64" / name
        if not source.is_file():
            pytest.skip("Build the native worker package before the distribution check")
        shutil.copyfile(source, package / name)
    # A source-only archive must work even if the last local build included NR.
    manifest = json.loads((package / "manifest.json").read_text())
    manifest["sha256"]["nvngx_dlssnr.dll"] = "0" * 64
    (package / "manifest.json").write_text(json.dumps(manifest))
    output = module.package_release(root, tmp_path / "source-only.zip")
    with zipfile.ZipFile(output) as archive:
        archived = json.loads(archive.read("Kyven/src/kyven/resources/dlss/windows_x64/manifest.json"))
        assert "nvngx_dlssnr.dll" not in archived["sha256"]
        for name, expected in archived["sha256"].items():
            content = archive.read("Kyven/src/kyven/resources/dlss/windows_x64/" + name)
            assert hashlib.sha256(content).hexdigest() == expected
    assert "nvngx_dlssnr.dll" in json.loads((package / "manifest.json").read_text())["sha256"]


@pytest.mark.parametrize("pointer", [False, True])
def test_distribution_rejects_missing_waft_models_or_lfs_pointers(tmp_path, pointer):
    original = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location("kyven_release", original / "scripts/package_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "source"
    waft = root / "plugins/Cattery/WAFT"
    waft.mkdir(parents=True)
    shutil.copyfile(original / "plugins/Cattery/WAFT/model-manifest.json", waft / "model-manifest.json")
    if pointer:
        (waft / "WAFT.cat").write_text("version https://git-lfs.github.com/spec/v1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="run git lfs pull"):
        module.package_release(root, tmp_path / "incomplete.zip")
