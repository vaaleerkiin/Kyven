"""Start and communicate with the external Kyven process."""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

from kyven_nuke import config
from kyven_nuke.client import NukeKyvenClient, NukeKyvenClientError

PORT = 8768
REQUIRED_API_VERSION = 4
LEGACY_PORTS = (8765, 8766, 8767)


def _check_health(current: NukeKyvenClient) -> None:
    health = current.health()
    if int(health.get("api_version", 0)) != REQUIRED_API_VERSION:
        raise NukeKyvenClientError(
            f"Kyven server API {health.get('api_version')} is incompatible; "
            f"expected {REQUIRED_API_VERSION}."
        )


def _server_environment(
    source: Mapping[str, str] | None = None,
    executable_dir: Path | None = None,
) -> dict[str, str]:
    """Build a clean process environment isolated from Nuke's DLL and Python runtime."""
    environment = dict(os.environ if source is None else source)
    for key in (
        "PYTHONHOME",
        "PYTHONPATH",
        "QT_PLUGIN_PATH",
        "QT_QPA_PLATFORM_PLUGIN_PATH",
        "QML2_IMPORT_PATH",
        "TCL_LIBRARY",
        "TK_LIBRARY",
    ):
        environment.pop(key, None)
    system_root = environment.get("SystemRoot") or environment.get("WINDIR") or r"C:\Windows"
    scripts_dir = executable_dir or config.executable().parent
    environment["PATH"] = os.pathsep.join(
        (str(scripts_dir), str(Path(system_root) / "System32"), system_root)
    )
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _startup_failure(log_path: Path, return_code: int) -> RuntimeError:
    try:
        detail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:].strip()
    except OSError:
        detail = "Server log could not be read."
    return RuntimeError(f"Kyven server exited with code {return_code}. {detail}")


def client() -> NukeKyvenClient:
    return NukeKyvenClient.from_token_file(config.token_file(), port=PORT)


def _unload_legacy_servers() -> None:
    """Release VRAM held by authenticated Kyven servers from older API revisions."""
    for port in LEGACY_PORTS:
        try:
            legacy = NukeKyvenClient.from_token_file(
                config.token_file(),
                port=port,
                timeout_seconds=2.0,
            )
            health = legacy.health()
            if health.get("service") == "kyven":
                legacy.unload_all()
        except (OSError, NukeKyvenClientError):
            continue


def ensure_server(timeout_seconds: float = 30.0) -> NukeKyvenClient:
    try:
        existing = client()
        _check_health(existing)
        return existing
    except (OSError, NukeKyvenClientError):
        pass

    _unload_legacy_servers()
    executable = config.python_executable()
    if not executable.is_file():
        raise RuntimeError(f"Kyven Python executable was not found: {executable}")
    config.runtime_dir().mkdir(parents=True, exist_ok=True)
    log_path = config.runtime_dir() / "server.log"
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                str(executable),
                "-I",
                "-m",
                "kyven.server.bootstrap",
                "serve",
                "--models-dir",
                str(config.models_dir()),
                "--device",
                "auto",
                "--port",
                str(PORT),
                "--token-file",
                str(config.token_file()),
            ],
            cwd=str(config.root()),
            env=_server_environment(executable_dir=executable.parent),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
        )
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise _startup_failure(log_path, return_code)
        try:
            current = client()
            _check_health(current)
            return current
        except (OSError, NukeKyvenClientError):
            time.sleep(0.2)
    raise RuntimeError(f"Kyven server did not become ready within 30 seconds. See {log_path}")
