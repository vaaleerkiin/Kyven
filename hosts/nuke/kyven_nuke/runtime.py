"""Start and communicate with the external Kyven process."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections.abc import Mapping
from pathlib import Path

from kyven_nuke import config
from kyven_nuke.client import NukeKyvenClient, NukeKyvenClientError

PORT = 18788
REQUIRED_API_VERSION = 26
LEGACY_PORTS = (8765, 8766, 8767, 8768, 8769, 18768, 18769, 18770, 18771, 18772, 18773, 18774, 18775, 18776, 18777, 18778, 18779, 18780, 18781, 18782, 18783, 18784, 18785, 18786, 18787)
_server_start_lock = threading.Lock()


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
    environment = {
        str(key).upper(): value for key, value in (os.environ if source is None else source).items()
    }
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
    system_root = environment.get("SYSTEMROOT") or environment.get("WINDIR") or r"C:\Windows"
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


def _listener_pids(netstat_output: str, port: int = PORT) -> tuple[int, ...]:
    """Extract Windows listener PIDs for one exact TCP port."""

    found: set[int] = set()
    suffix = f":{port}"
    for line in netstat_output.splitlines():
        fields = line.split()
        if len(fields) < 5 or fields[0].upper() != "TCP" or fields[3].upper() != "LISTENING":
            continue
        if fields[1].endswith(suffix) and fields[4].isdigit():
            found.add(int(fields[4]))
    return tuple(sorted(found))


def _terminate_authenticated_listener() -> None:
    """Windows fallback for an older authenticated Kyven server without shutdown API."""

    if os.name != "nt":
        raise RuntimeError("The running Kyven server does not support remote shutdown.")
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    output = subprocess.check_output(
        ["netstat", "-ano"],
        text=True,
        errors="replace",
        creationflags=creation_flags,
    )
    pids = _listener_pids(output)
    if len(pids) != 1:
        raise RuntimeError(f"Could not identify one Kyven listener on port {PORT}: {pids}")
    os.kill(pids[0], signal.SIGTERM)


def stop_server(timeout_seconds: float = 10.0) -> bool:
    """Stop only the authenticated Kyven service on the configured port."""

    try:
        current = client()
        health = current.health()
    except (OSError, NukeKyvenClientError):
        return False
    if health.get("service") != "kyven":
        raise RuntimeError(f"Port {PORT} is not owned by a Kyven service.")
    try:
        current.shutdown_server()
    except (OSError, NukeKyvenClientError):
        _terminate_authenticated_listener()
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            client().health()
        except (OSError, NukeKyvenClientError):
            return True
        time.sleep(0.1)
    raise RuntimeError("Kyven Server did not stop within 10 seconds.")


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

    with _server_start_lock:
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
        raise RuntimeError(
            f"Kyven server did not become ready within {timeout_seconds:g} seconds. See {log_path}"
        )
