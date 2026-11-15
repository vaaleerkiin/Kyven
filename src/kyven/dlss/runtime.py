"""Install Kyven's portable worker; never depend on another Nuke plugin."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from kyven.errors import ErrorCode, KyvenError

MODEL_ID = "dlss-external"
WORKER_NAME = "KyvenDLSSWorker.exe"
OWN_FILES = (WORKER_NAME, "nvngx.dll", "LICENSE.upstream")
VENDOR_FILE = "nvngx_dlssnr.dll"


def runtime_root(models_dir: Path) -> Path:
    return Path(os.environ.get("KYVEN_RUNTIME_DIR", models_dir.parent / ".runtime")).resolve()


def package_root() -> Path:
    return Path(__file__).parents[1] / "resources" / "dlss" / "windows_x64"


def vendor_source() -> dict[str, Any]:
    path = Path(__file__).parents[1] / "resources" / "dlss" / "vendor-source.json"
    return json.loads(path.read_text(encoding="utf-8"))


def bundled_vendor_available() -> bool:
    """Cheap package metadata probe; installation verifies the complete file hash."""
    package = package_root()
    try:
        manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
        spec = vendor_source()
        vendor = package / VENDOR_FILE
        return (vendor.is_file() and vendor.stat().st_size == spec["member_size_bytes"]
                and manifest.get("sha256", {}).get(VENDOR_FILE) == spec["member_sha256"])
    except (OSError, ValueError, KeyError):
        return False


def _download_vendor(destination: Path, cancelled=None, progress=None) -> None:
    """Download one pinned archive; extract only the audited NR member, never its worker."""
    spec = vendor_source()
    descriptor, name = tempfile.mkstemp(prefix=".nr-package-", suffix=".zip", dir=destination)
    os.close(descriptor)
    archive_path = Path(name)
    temporary = destination / ".nvngx_dlssnr.installing"
    try:
        received = 0
        digest = hashlib.sha256()
        with urllib.request.urlopen(spec["url"], timeout=30) as response, archive_path.open("wb") as output:
            while True:
                if cancelled is not None and cancelled():
                    raise KyvenError(ErrorCode.CANCELLED, "DLSS component download cancelled.")
                block = response.read(1024 * 1024)
                if not block:
                    break
                received += len(block)
                if received > spec["size_bytes"]:
                    raise KyvenError(ErrorCode.DEPENDENCY_MISSING, "DLSS archive exceeded its pinned size.")
                digest.update(block)
                output.write(block)
                if progress is not None:
                    progress(received, spec["size_bytes"])
        if received != spec["size_bytes"] or digest.hexdigest() != spec["sha256"]:
            raise KyvenError(ErrorCode.DEPENDENCY_MISSING, "DLSS archive size or SHA-256 verification failed.")
        with zipfile.ZipFile(archive_path) as archive:
            info = archive.getinfo(spec["member"])
            if info.file_size != spec["member_size_bytes"]:
                raise KyvenError(ErrorCode.DEPENDENCY_MISSING, "DLSS component has an unexpected size.")
            with archive.open(info) as source, temporary.open("wb") as output:
                while True:
                    if cancelled is not None and cancelled():
                        raise KyvenError(ErrorCode.CANCELLED, "DLSS extraction cancelled.")
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
        if _hash(temporary, cancelled) != spec["member_sha256"]:
            raise KyvenError(ErrorCode.DEPENDENCY_MISSING, "DLSS component SHA-256 verification failed.")
        os.replace(temporary, destination / VENDOR_FILE)
    except (OSError, ValueError, zipfile.BadZipFile, KeyError) as exc:
        raise KyvenError(ErrorCode.DEPENDENCY_MISSING, "Could not install the pinned DLSS component.",
                         technical_detail=str(exc), recoverable=True,
                         suggested_action="Retry Install Selected in Model Manager.") from exc
    finally:
        archive_path.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


def resolve_worker(root: Path, preferred: str = "") -> Path | None:
    """Only Kyven's managed directory participates; legacy node paths are ignored."""
    worker = root / "dlss" / "bin" / WORKER_NAME
    if worker.is_file() and (worker.parent / "nvngx.dll").is_file() and (
            worker.parent / VENDOR_FILE).is_file():
        return worker.resolve()
    return None


def _hash(path: Path, cancelled=None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            if cancelled is not None and cancelled():
                raise KyvenError(ErrorCode.CANCELLED, "DLSS runtime setup cancelled.")
            digest.update(block)
    return digest.hexdigest()


def bind_runtime(root: Path, source: Path | None = None, cancelled=None,
                 download_vendor: bool = False, progress=None) -> Path:
    """Install checksum-verified Kyven files and an optional NVIDIA component.

    Source is the vendor DLL, never another application's worker/runtime.
    No binaries are discovered in .nuke, game folders or other plugins.
    """
    package = package_root()
    try:
        manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise KyvenError(ErrorCode.DEPENDENCY_MISSING, "The Kyven DLSS worker package is missing.",
                         suggested_action="Install the Kyven distribution containing its built worker.") from exc
    for name in OWN_FILES:
        if _hash(package / name, cancelled) != manifest.get("sha256", {}).get(name):
            raise KyvenError(ErrorCode.DEPENDENCY_MISSING, f"Kyven runtime checksum failed: {name}")
    if source is not None and (not source.is_file() or source.name.lower() != VENDOR_FILE):
        raise KyvenError(ErrorCode.INVALID_REQUEST, f"Select the NVIDIA component {VENDOR_FILE}.")
    destination = root / "dlss" / "bin"
    destination.mkdir(parents=True, exist_ok=True)
    for name in OWN_FILES:
        if cancelled is not None and cancelled():
            raise KyvenError(ErrorCode.CANCELLED, "DLSS runtime setup cancelled.")
        shutil.copy2(package / name, destination / name)
    vendor = source or (package / VENDOR_FILE)
    if vendor.is_file():
        descriptor, name = tempfile.mkstemp(prefix=".vendor-", suffix=".dll", dir=destination)
        os.close(descriptor)
        temporary = Path(name)
        try:
            shutil.copyfile(vendor, temporary)
            actual = _hash(temporary, cancelled)
            if source is None:
                spec = vendor_source()
                if (actual != manifest.get("sha256", {}).get(VENDOR_FILE)
                        or actual != spec["member_sha256"]
                        or temporary.stat().st_size != spec["member_size_bytes"]):
                    raise KyvenError(ErrorCode.DEPENDENCY_MISSING,
                                     "Bundled NVIDIA component size or SHA-256 verification failed.")
            os.replace(temporary, destination / VENDOR_FILE)
        finally:
            temporary.unlink(missing_ok=True)
    elif download_vendor and not (destination / VENDOR_FILE).is_file():
        _download_vendor(destination, cancelled, progress)
    worker = resolve_worker(root)
    if worker is None:
        raise KyvenError(
            ErrorCode.PROVIDER_UNAVAILABLE,
            "Kyven's worker is installed. The NVIDIA DLSS 5 NR component is unavailable.",
            suggested_action="Use Install Selected in Model Manager to review and install the pinned "
                             "community-supplied NVIDIA component, or import your licensed DLL.",
        )
    return worker


def unbind_runtime(root: Path) -> None:
    """Remove only Kyven worker files, preserving any supplied NVIDIA component."""
    folder = root / "dlss" / "bin"
    for name in OWN_FILES:
        (folder / name).unlink(missing_ok=True)


def runtime_snapshot(root: Path) -> dict[str, Any]:
    worker = resolve_worker(root)
    exr_ready = importlib.util.find_spec("OpenEXR") is not None
    own_ready = (root / "dlss" / "bin" / WORKER_NAME).is_file()
    current = False
    if worker is not None:
        try:
            manifest = json.loads((package_root() / "manifest.json").read_text(encoding="utf-8"))
            current = all(_hash(worker.parent / name) == manifest["sha256"][name]
                          for name in (WORKER_NAME, "nvngx.dll"))
        except (OSError, ValueError, KeyError):
            pass
    ready = worker is not None and exr_ready and current
    source = vendor_source()
    bundled = bundled_vendor_available()
    status = ("Ready" if ready else "Update Kyven runtime" if worker and not current else
              "Install EXR support" if worker else
              "NVIDIA NR component required" if own_ready else "Install Kyven runtime")
    return {
        "model_id": MODEL_ID, "task": "dlss", "display_name": "Kyven DLSS Runtime",
        "provider": MODEL_ID,
        "size_bytes": source["member_size_bytes"] if bundled else source["size_bytes"],
        "parameters_millions": 0,
        "recommended_vram_mb": None, "supports_cpu": False, "compatible": None,
        "license": source["license"], "license_url": source["license_url"],
        "commercial_use": None, "redistribution": False,
        "license_acceptance_required": not bundled,
        "source_url": source["release_url"],
        "runtime": "dlss", "download_type": "bundled_runtime", "installed": ready,
        "worker": str(worker) if worker else "", "status_message": status,
        "local_runtime": True,
        "vendor_bundled": bundled,
        "supports_control_mask": ready,
        "update_available": worker is not None and not current,
    }
