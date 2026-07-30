"""Start and communicate with the external Kyven process."""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

from kyven_nuke import config
from kyven_nuke.client import NukeKyvenClient, NukeKyvenClientError

PORT = 8765


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


def ensure_server(timeout_seconds: float = 30.0) -> NukeKyvenClient:
    try:
        existing = client()
        existing.health()
        return existing
    except (OSError, NukeKyvenClientError):
        pass

    executable = config.executable()
    if not executable.is_file():
        raise RuntimeError(f"Kyven executable was not found: {executable}")
    config.runtime_dir().mkdir(parents=True, exist_ok=True)
    log_path = config.runtime_dir() / "server.log"
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                str(executable),
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
            current.health()
            return current
        except (OSError, NukeKyvenClientError):
            time.sleep(0.2)
    raise RuntimeError(f"Kyven server did not become ready within 30 seconds. See {log_path}")
