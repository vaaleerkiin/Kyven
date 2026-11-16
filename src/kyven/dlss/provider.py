"""Bounded, cancelable client for the upstream D5V5 worker pipe protocol.

Protocol reference: KJzzzKJ/DLSS5-for-Nuke, commit
d61307ef4543c8991668730a4cab8019b3cb0096, worker/Protocol.h.
The Nuke adapter selects Kyven's managed worker; runtime installation is handled separately
through Model Manager. CLI requests can explicitly select a local compatible worker.
"""

from __future__ import annotations

import os
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from kyven.cancellation import CancellationToken
from kyven.dlss.models import DlssRequest
from kyven.errors import ErrorCode, KyvenError

VIDEO = struct.Struct("<14I4f5If")
SETUP = struct.Struct("<12I")
FRAME = struct.Struct("<4Iq")
RESPONSE = struct.Struct("<5Iq")


@dataclass(frozen=True)
class DlssCapabilities:
    provider_id: str = "dlss-external"
    supports_cpu: bool = False
    supports_tiling: bool = False
    supports_reduced_resolution: bool = True
    supports_external_motion: bool = True
    # Kyven's worker expands the scalar artist guide into an RGBA control texture
    # and initializes the native backbuffer. A scalar GPU texture is not equivalent.
    supports_control_mask: bool = True
    minimum_vram_mb: int | None = None  # Undocumented runtime: no invented guarantee.
    license_name: str = "Kyven adapter Apache-2.0; worker MIT; NVIDIA NR separately licensed"
    license_url: str = "https://github.com/KJzzzKJ/DLSS5-for-Nuke#runtime-policy"


class DlssProvider(Protocol):
    capabilities: DlssCapabilities

    def predict(self, request: DlssRequest, rgba: np.ndarray, motion: np.ndarray | None,
                mask: np.ndarray | None, depth: np.ndarray | None, reset: bool,
                cancellation: CancellationToken) -> np.ndarray: ...

    def unload(self) -> None: ...


class ExternalDlssProvider:
    """One resident worker. All calls are serialized by the shared GPU scheduler."""

    capabilities = DlssCapabilities()

    def __init__(self, device: str = "auto") -> None:
        self.device = device
        self._process: subprocess.Popen | None = None
        self._configuration: bytes | None = None
        self._identity: tuple | None = None
        self._index = 0
        self.generation = 0
        self._guide_worker: tuple | None = None

    def unload(self) -> None:
        process, self._process = self._process, None
        self._configuration = None
        self._identity = None
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()

    @staticmethod
    def _read(stream, count: int) -> bytes:
        data = bytearray()
        while len(data) < count:
            chunk = stream.read(count - len(data))
            if not chunk:
                raise OSError("DLSS worker closed its output pipe")
            data.extend(chunk)
        return bytes(data)

    def predict(self, request: DlssRequest, rgba: np.ndarray, motion: np.ndarray | None,
                mask: np.ndarray | None, depth: np.ndarray | None, reset: bool,
                cancellation: CancellationToken) -> np.ndarray:
        if os.name != "nt" or self.device == "cpu":
            raise KyvenError(
                ErrorCode.PROVIDER_UNAVAILABLE, "DLSS requires Windows and an NVIDIA RTX GPU.",
                suggested_action="Use a Windows RTX workstation; other Kyven CPU tools remain available.",
            )
        cancellation.raise_if_cancelled()
        if mask is not None and not self.capabilities.supports_control_mask:
            raise KyvenError(
                ErrorCode.PROVIDER_UNAVAILABLE,
                "This DLSS component has not passed control-mask image validation.",
                recoverable=True,
                suggested_action="Select Control Mask None for full-frame reconstruction. "
                                 "Mask-guided NR requires a qualified runtime; invalid masked RGB "
                                 "is not published as a successful result.",
            )
        height, width = rgba.shape[:2]
        header = VIDEO.pack(
            0x35563544, width, height, width, height, 0, 100000, 0, 0,
            0, request.preset, request.style, 0, 0,
            # Native Intensity fades RGB toward black. Artist intensity is applied
            # to the scene-linear residual by the service instead.
            1.0, request.local_tone, request.local_structure, request.skin_structure,
            int(motion is not None), 0, 0, 0, 0, 0.0,
        )
        if not request.worker.is_file():
            raise KyvenError(ErrorCode.PROVIDER_UNAVAILABLE, "Select a local DLSS worker executable.")
        self._check_runtime(request)
        stat = request.worker.stat()
        identity = (str(request.worker.resolve()), stat.st_size, stat.st_mtime_ns, request.session)
        if mask is not None and self._guide_worker != identity[:3]:
            self._check_mask_worker(request)
            self._guide_worker = identity[:3]
        # Do blocking pipe I/O on a disposable thread. The scheduler watches cancellation
        # and deadlines, kills the child to interrupt both reads and writes, then joins.
        done = threading.Event()
        results: list[np.ndarray | Exception] = []

        def exchange() -> None:
            try:
                restarted = (self._process is None or self._process.poll() is not None
                             or self._configuration != header or self._identity != identity
                             or reset)
                if restarted:
                    self.unload()
                    cancellation.raise_if_cancelled()
                    self._process = subprocess.Popen(
                        [str(request.worker), "--video"], stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
                        cwd=str(request.worker.parent), creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                    self._configuration, self._identity = header, identity
                    self._index = 0
                    self.generation += 1
                    self._write(header)
                    setup = SETUP.unpack(self._read(self._process.stdout, SETUP.size))
                    if setup[0] != 0x34505553 or setup[1] != 1:
                        raise OSError(f"DLSS setup rejected (NGX 0x{setup[2]:08x})")
                    if setup[3:7] != (width, height, width, height):
                        raise OSError("DLSS worker negotiated unexpected dimensions")
                flags = int(depth is not None) | (int(mask is not None) << 1)
                index = self._index
                self._write(FRAME.pack(0x314D5246, index, int(reset or restarted), flags, request.frame))
                self._write(np.ascontiguousarray(rgba, dtype="<f2").tobytes())
                if motion is not None:
                    self._write(np.ascontiguousarray(motion, dtype="<f2").tobytes())
                # White enables all NR controls; black preserves the initialized base.
                control = None if mask is None else np.clip(mask, 0.0, 1.0)
                for guide in (depth, control):
                    if guide is not None:
                        self._write(np.ascontiguousarray(guide, dtype="<f4").tobytes())
                response = RESPONSE.unpack(self._read(self._process.stdout, RESPONSE.size))
                if response != (0x3154554F, index, 1, width * height * 8,
                                response[4], request.frame):
                    raise OSError(f"Invalid DLSS frame response: {response}")
                pixels = self._read(self._process.stdout, width * height * 8)
                self._index += 1
                results.append(np.frombuffer(pixels, dtype="<f2").reshape(height, width, 4)
                               .astype(np.float32))
            except Exception as exc:  # noqa: BLE001 - pipe/child boundary
                results.append(exc)
            finally:
                done.set()

        thread = threading.Thread(target=exchange, name="kyven-dlss-pipe", daemon=True)
        thread.start()
        deadline = time.monotonic() + request.timeout
        try:
            while not done.wait(0.025):
                cancellation.raise_if_cancelled()
                if time.monotonic() > deadline:
                    raise OSError("DLSS worker timed out")
            cancellation.raise_if_cancelled()
            if isinstance(results[0], Exception):
                raise results[0]
            return results[0]
        except Exception as exc:
            # Kill without closing a pipe while the I/O thread may still be using it.
            process = self._process
            if process is not None and process.poll() is None:
                process.kill()
            thread.join(timeout=5)
            self.unload()
            if isinstance(exc, KyvenError):
                raise
            raise KyvenError(
                ErrorCode.INFERENCE_FAILED, "DLSS worker initialization or inference failed.",
                technical_detail=str(exc), recoverable=True,
                suggested_action="Check runtime compatibility; for GPU memory errors lower Processing Limit.",
            ) from exc

    @staticmethod
    def _check_mask_worker(request: DlssRequest) -> None:
        try:
            version = subprocess.run([str(request.worker), "--version"], capture_output=True,
                                     text=True, timeout=5,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                                     check=True)
            if "control-mask=rgba-v1" in version.stdout:
                return
        except (OSError, subprocess.SubprocessError):
            pass
        raise KyvenError(ErrorCode.PROVIDER_UNAVAILABLE,
                         "The installed DLSS worker needs the RGBA control-guide update.",
                         suggested_action="Open Model Manager > Kyven DLSS Runtime > Install Selected.")

    @staticmethod
    def _check_runtime(request: DlssRequest) -> None:
        missing = [name for name in ("nvngx_dlssnr.dll", "nvngx.dll")
                   if not (request.worker.parent / name).is_file()]
        if missing:
            raise KyvenError(
                ErrorCode.PROVIDER_UNAVAILABLE, "DLSS runtime files are missing: " + ", ".join(missing),
                recoverable=True,
                suggested_action="Place your compatible local runtime and the project-built caller shim "
                                 "beside DLSS_Nuke_Worker.exe; Kyven does not download these files.",
            )

    def _write(self, data: bytes) -> None:
        view = memoryview(data)
        while view:
            written = self._process.stdin.write(view)
            if not written:
                raise OSError("DLSS worker closed its input pipe")
            view = view[written:]
