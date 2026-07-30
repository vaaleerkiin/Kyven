"""Start and communicate with the external Kyven process."""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Mapping

from kyven_nuke import config
from kyven_nuke.client import NukeKyvenClient, NukeKyvenClientError

PORT = 8765


def _server_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Prevent Nuke's embedded Python settings from leaking into Kyven's venv."""
    environment = dict(os.environ if source is None else source)
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


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
        subprocess.Popen(
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
            env=_server_environment(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
        )
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            current = client()
            current.health()
            return current
        except (OSError, NukeKyvenClientError):
            time.sleep(0.2)
    raise RuntimeError(f"Kyven server did not become ready within 30 seconds. See {log_path}")
