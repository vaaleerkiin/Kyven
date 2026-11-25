"""Apache-2.0 transport to the separately licensed, standalone LanPaint worker."""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from kyven.cancellation import CancellationToken
from kyven.errors import ErrorCode, KyvenError
from kyven.generative_fill.assets import bundle_ready, manifest_checksum
from kyven.generative_fill.models import (
    PROVIDER_ID,
    EnhancePromptRequest,
    GenerativeFillRequest,
)
from kyven.inpaint.models import InpaintCapabilities, InpaintPrediction
from kyven.inpaint.providers.base import InpaintProvider


class KleinLanPaintProvider(InpaintProvider):
    """Resident private process; the shared Kyven scheduler serializes requests."""

    def __init__(self, models_dir: Path, device: str = "auto") -> None:
        self.models = models_dir.resolve() / PROVIDER_ID
        self.device = device
        root = Path(os.environ.get("KYVEN_ROOT", str(models_dir.resolve().parent)))
        self.runtime = Path(os.environ.get("KYVEN_RUNTIME_DIR", str(root / ".runtime"))) / "generative-fill"
        self.python = Path(os.environ.get("KYVEN_GENERATIVE_PYTHON", str(
            self.runtime / ("venv/Scripts/python.exe" if os.name == "nt" else "venv/bin/python"))))
        self.worker = Path(__file__).parents[1] / "resources/generative_fill/worker.py"
        self._process: subprocess.Popen | None = None
        self._messages: queue.Queue = queue.Queue()
        self._reader: threading.Thread | None = None
        self._log: Any = None
        self._verified = False

    @property
    def capabilities(self) -> InpaintCapabilities:
        return InpaintCapabilities(
            PROVIDER_ID, "FLUX.2 Klein 4B Q4 + LanPaint", "2.4.0",
            manifest_checksum(), "Apache-2.0 weights; GPL-3.0 optional worker",
            "https://github.com/scraed/LanPaint/blob/2d7912f9a5efe5ece8de334c7ca18317b8288c39/LICENSE",
            True, 8192,
        )

    def _start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self.unload()
        if not self.python.is_file():
            raise KyvenError(ErrorCode.DEPENDENCY_MISSING,
                             "Install the Generative Fill runtime first.", recoverable=True,
                             suggested_action="Click Install Runtime on the Generative Fill node.")
        if not self._verified:
            if not self._bundle_ready():
                raise KyvenError(ErrorCode.MODEL_NOT_FOUND,
                                 "Generative Fill model bundle is missing or failed verification.",
                                 recoverable=True, suggested_action="Install the selected Generative Fill model in Model Manager.")
            self._verified = True
        self.runtime.mkdir(parents=True, exist_ok=True)
        self._log = (self.runtime / "worker.log").open("a", encoding="utf-8")
        environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                           HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
        self._messages = queue.Queue()
        messages = self._messages
        try:
            self._process = subprocess.Popen(
                [str(self.python), "-I", "-u", str(self.worker), "--models", str(self.models),
                 "--device", self.device],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                text=True, encoding="utf-8", env=environment,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except OSError as exc:
            self.unload()
            raise KyvenError(ErrorCode.PROVIDER_UNAVAILABLE,
                             "Could not start the Generative Fill worker.",
                             technical_detail=str(exc), recoverable=True) from exc
        process = self._process

        def read() -> None:
            try:
                for line in process.stdout:
                    try:
                        messages.put(json.loads(line))
                    except ValueError:
                        continue
            finally:
                messages.put({"event": "closed"})

        self._reader = threading.Thread(target=read, name="kyven-fill-worker", daemon=True)
        self._reader.start()

    def _bundle_ready(self) -> bool:
        return bundle_ready(self.models, verify=True)

    def _exchange(self, payload: dict, token: CancellationToken, timeout: int) -> dict:
        token.raise_if_cancelled()
        self._start()
        try:
            token.raise_if_cancelled()
            self._process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._process.stdin.flush()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                token.raise_if_cancelled()
                try:
                    response = self._messages.get(timeout=0.1)
                except queue.Empty:
                    continue
                event = response.get("event")
                if event == "progress":
                    token.report_progress(float(response["progress"]), str(response["message"]))
                elif event == "result":
                    return response
                elif event == "error":
                    raise KyvenError(ErrorCode.INFERENCE_FAILED,
                                     str(response.get("message", "Generation failed.")),
                                     technical_detail=str(response.get("detail", "")),
                                     recoverable=True,
                                     suggested_action="Use a smaller Working Size or check worker.log.")
                elif event == "closed":
                    raise KyvenError(ErrorCode.INFERENCE_FAILED,
                                     "The Generative Fill worker stopped unexpectedly.",
                                     recoverable=True, suggested_action=f"Check {self.runtime / 'worker.log'}.")
            raise KyvenError(ErrorCode.INFERENCE_FAILED, "Generative Fill exceeded its time limit.",
                             recoverable=True, suggested_action="Reduce Working Size or increase Advanced timeout.")
        except (BrokenPipeError, OSError) as exc:
            raise KyvenError(ErrorCode.INFERENCE_FAILED, "Generative Fill worker communication failed.",
                             technical_detail=str(exc), recoverable=True) from exc
        finally:
            # A cancelled, failed or timed-out exchange must never keep computing.
            # A successful resident worker remains available for the next frame.
            if 'response' not in locals() or response.get("event") != "result":
                self.unload()

    def predict(self, request: GenerativeFillRequest,
                cancellation: CancellationToken) -> InpaintPrediction:
        from tempfile import TemporaryDirectory

        request.fill.validate()
        with TemporaryDirectory(prefix="kyven-fill-output-") as directory:
            output = Path(directory) / "patch.png"
            response = self._exchange({
                "action": "fill", "source": str(request.source), "mask": str(request.mask),
                "output": str(output), "settings": asdict(request.fill),
                "low_memory": request.profile.value == "low_memory",
            }, cancellation, request.fill.timeout_seconds)
            if not output.is_file():
                raise KyvenError(ErrorCode.OUTPUT_FAILED, "Generative Fill did not produce a patch.")
            with Image.open(output) as image:
                pixels = np.asarray(image.convert("RGB")).copy()
        return InpaintPrediction(pixels, dict(response.get("metadata") or {}))

    def stitch(self, generated: np.ndarray, source: np.ndarray, mask: np.ndarray,
               cancellation: CancellationToken) -> tuple[np.ndarray, dict]:
        from tempfile import TemporaryDirectory
        with TemporaryDirectory(prefix="kyven-fill-stitch-") as directory:
            root = Path(directory)
            for name, pixels in (("generated", generated), ("source", source), ("mask", mask)):
                Image.fromarray(pixels).save(root / f"{name}.png")
            response = self._exchange({"action": "stitch", "generated": str(root / "generated.png"),
                "source": str(root / "source.png"), "mask": str(root / "mask.png"),
                "output": str(root / "output.png")}, cancellation, 300)
            cancellation.raise_if_cancelled()
            with Image.open(root / "output.png") as image:
                result = np.array(image.convert("RGB"))
        return result, dict(response.get("metadata") or {})

    def enhance(self, request: EnhancePromptRequest, cancellation: CancellationToken) -> dict:
        request.validate()
        response = self._exchange({"action": "enhance", "settings": asdict(request)},
                                  cancellation, request.timeout_seconds)
        prompt = str(response.get("prompt", "")).strip()
        if not prompt:
            raise KyvenError(ErrorCode.INFERENCE_FAILED, "Qwen returned an empty prompt.")
        return {"prompt": prompt, "original_prompt": request.prompt,
                "metadata": dict(response.get("metadata") or {})}

    def unload(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
            if self._reader:
                self._reader.join(timeout=2)
            for stream in (process.stdin, process.stdout):
                if stream:
                    stream.close()
        if self._log:
            self._log.close()
            self._log = None
        self._reader = None
