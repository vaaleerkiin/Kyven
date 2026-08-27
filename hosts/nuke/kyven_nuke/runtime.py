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
REQUIRED_API_VERSION = 30
LEGACY_PORTS = (8765, 8766, 8767, 8768, 8769, 18768, 18769, 18770, 18771, 18772, 18773, 18774, 18775, 18776, 18777, 18778, 18779, 18780, 18781, 18782, 18783, 18784, 18785, 18786, 18787)
_server_start_lock = threading.Lock()
_server_state_lock = threading.RLock()
_server_state = {
    "status": "unknown",
    "message": "Server state has not been checked.",
    "started_at": None,
    "elapsed_seconds": 0.0,
}

HQ_SAM2_RUNTIME_URL = (
    "HQ-SAM-2 @ https://github.com/SysCV/sam-hq/archive/"
    "e696978d60352dc9a26b12631cd91781502c6546.zip#subdirectory=sam-hq2"
)


class RuntimeInstallCancelled(RuntimeError):
    """Raised when the artist cancels an optional model runtime installation."""


def _runtime_probe(runtime_id: str) -> str:
    if runtime_id == "hq-sam2":
        return (
            "from sam2.build_sam import build_sam2_hq_video_predictor; "
            "from sam2.modeling.sam2_hq_base import SAM2HQBase"
        )
    raise RuntimeError(f"Unknown Kyven model runtime: {runtime_id}")


def _runtime_install_command(runtime_id: str, executable: Path) -> list[str]:
    if runtime_id != "hq-sam2":
        raise RuntimeError(f"Unknown Kyven model runtime: {runtime_id}")
    return [
        str(executable),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-build-isolation",
        "--no-deps",
        "--force-reinstall",
        HQ_SAM2_RUNTIME_URL,
    ]


def ensure_model_runtime(runtime_id: str, progress_callback=None, cancel_requested=None) -> bool:
    """Install a catalog-declared runtime into Kyven's private environment when missing."""
    if not runtime_id:
        return False
    executable = config.python_executable()
    if not executable.is_file():
        raise RuntimeError(f"Kyven Python executable was not found: {executable}")
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    probe = subprocess.run(
        [str(executable), "-I", "-c", _runtime_probe(runtime_id)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation_flags,
        check=False,
    )
    if probe.returncode == 0:
        return False
    _report_progress(progress_callback, f"Installing required runtime: {runtime_id}...")
    process = subprocess.Popen(
        _runtime_install_command(runtime_id, executable),
        cwd=str(config.root()),
        env=_server_environment(executable_dir=executable.parent),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        creationflags=creation_flags,
    )
    while True:
        try:
            output, _ = process.communicate(timeout=0.2)
            break
        except subprocess.TimeoutExpired:
            if cancel_requested is not None and cancel_requested():
                process.terminate()
                output, _ = process.communicate()
                raise RuntimeInstallCancelled("Model runtime installation was cancelled.")
    if process.returncode != 0:
        detail = (output or "pip returned no diagnostic output")[-4000:].strip()
        raise RuntimeError(f"Could not install required runtime {runtime_id}: {detail}")
    verification = subprocess.run(
        [str(executable), "-I", "-c", _runtime_probe(runtime_id)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creation_flags=creation_flags,
        check=False,
    )
    if verification.returncode != 0:
        raise RuntimeError(f"Runtime {runtime_id} was installed but failed its import check.")
    return True


def server_state() -> dict[str, object]:
    with _server_state_lock:
        return dict(_server_state)


def _update_server_state(status: str, message: str, started_at: float | None = None) -> None:
    with _server_state_lock:
        if started_at is not None:
            _server_state["started_at"] = started_at
        origin = _server_state.get("started_at")
        _server_state.update(
            status=status,
            message=message,
            elapsed_seconds=(time.monotonic() - float(origin)) if origin is not None else 0.0,
        )


def _report_progress(callback, message: str) -> None:
    if callback is None:
        return
    try:
        callback(message)
    except Exception:  # noqa: BLE001 - a host UI callback must not abort server startup
        return


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


def _listening_legacy_ports(netstat_output: str) -> tuple[int, ...]:
    candidates = set(LEGACY_PORTS)
    found: set[int] = set()
    for line in netstat_output.splitlines():
        fields = line.split()
        if len(fields) < 5 or fields[0].upper() != "TCP" or fields[3].upper() != "LISTENING":
            continue
        token = fields[1].rsplit(":", 1)[-1]
        if token.isdigit() and int(token) in candidates:
            found.add(int(token))
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

    _update_server_state("stopping", "Stopping Kyven Server...")
    try:
        current = client()
        health = current.health()
    except (OSError, NukeKyvenClientError):
        _update_server_state("stopped", "Kyven Server is stopped.")
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
            _update_server_state("stopped", "Kyven Server stopped.")
            return True
        time.sleep(0.1)
    message = f"Kyven Server did not stop within {timeout_seconds:g} seconds."
    _update_server_state("failed", message)
    raise RuntimeError(message)


def _unload_legacy_servers() -> None:
    """Release VRAM held by authenticated Kyven servers from older API revisions."""
    ports = LEGACY_PORTS
    if os.name == "nt":
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            output = subprocess.check_output(
                ["netstat", "-ano"],
                text=True,
                errors="replace",
                creationflags=creation_flags,
            )
            ports = _listening_legacy_ports(output)
        except (OSError, subprocess.SubprocessError):
            ports = ()
    for port in ports:
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


def ensure_server(timeout_seconds: float = 30.0, progress_callback=None) -> NukeKyvenClient:
    try:
        existing = client()
        _check_health(existing)
        _update_server_state("ready", "Kyven Server is ready.")
        _report_progress(progress_callback, "Kyven Server is ready.")
        return existing
    except NukeKyvenClientError as exc:
        if "incompatible" in str(exc).lower():
            _update_server_state("restarting", "Restarting outdated Kyven Server...")
            _report_progress(progress_callback, "Restarting outdated Kyven Server...")
            stop_server()
    except OSError:
        pass

    with _server_start_lock:
        try:
            existing = client()
            _check_health(existing)
            _update_server_state("ready", "Kyven Server is ready.")
            _report_progress(progress_callback, "Kyven Server became ready while waiting.")
            return existing
        except NukeKyvenClientError as exc:
            if "incompatible" in str(exc).lower():
                _update_server_state("restarting", "Restarting outdated Kyven Server...")
                _report_progress(progress_callback, "Restarting outdated Kyven Server...")
                stop_server()
        except OSError:
            pass

        started_at = time.monotonic()
        _update_server_state("starting", "Checking older Kyven server processes...", started_at)
        _report_progress(progress_callback, "Starting server: checking older processes...")
        _unload_legacy_servers()
        executable = config.python_executable()
        if not executable.is_file():
            message = f"Kyven Python executable was not found: {executable}"
            _update_server_state("failed", message)
            raise RuntimeError(message)
        log_path = config.runtime_dir() / "server.log"
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            config.runtime_dir().mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as log:
                _update_server_state("starting", "Launching isolated Python runtime...")
                _report_progress(progress_callback, "Starting server: launching runtime...")
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
        except OSError as exc:
            message = f"Could not launch Kyven Server: {exc}. Log: {log_path}"
            _update_server_state("failed", message)
            raise RuntimeError(message) from exc
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            return_code = process.poll()
            if return_code is not None:
                failure = _startup_failure(log_path, return_code)
                _update_server_state("failed", str(failure))
                raise failure
            try:
                current = client()
                _check_health(current)
                elapsed = time.monotonic() - started_at
                message = f"Kyven Server is ready in {elapsed:.1f}s."
                _update_server_state("ready", message)
                _report_progress(progress_callback, message)
                return current
            except (OSError, NukeKyvenClientError):
                elapsed = time.monotonic() - started_at
                message = f"Starting Kyven Server... {elapsed:.1f}s / {timeout_seconds:g}s"
                _update_server_state("starting", message)
                _report_progress(progress_callback, message)
                time.sleep(0.2)
        message = (
            f"Kyven server did not become ready within {timeout_seconds:g} seconds. See {log_path}"
        )
        _update_server_state("failed", message)
        raise RuntimeError(message)
