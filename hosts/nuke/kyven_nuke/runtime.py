"""Start and communicate with the external Kyven process."""

from __future__ import annotations

import subprocess
import time

from kyven_nuke import config
from kyven_nuke.client import NukeKyvenClient, NukeKyvenClientError

PORT = 8765


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
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
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
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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
    raise RuntimeError("Kyven server did not become ready within 30 seconds.")
