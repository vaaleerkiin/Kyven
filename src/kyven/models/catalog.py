"""Trusted, task-neutral model catalog."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kyven.errors import ErrorCode, KyvenError
from kyven.segment.providers.registry import ProviderRegistry


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    task: str
    display_name: str
    provider: str
    config: str
    filename: str
    source: str
    sha256: str
    size_bytes: int
    parameters_millions: float
    recommended_vram_mb: int
    supports_cpu: bool
    license: str
    license_url: str
    commercial_use: bool
    redistribution: bool
    download_type: str = "file"
    revision: str = ""
    allow_patterns: tuple[str, ...] = ()
    license_acceptance_required: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ModelSpec:
        normalized = dict(value)
        normalized["allow_patterns"] = tuple(normalized.get("allow_patterns", ()))
        return cls(**normalized)

    def path(self, models_dir: Path) -> Path:
        return models_dir / self.filename

    def snapshot(self, models_dir: Path, available_vram_mb: int | None = None) -> dict[str, Any]:
        path = self.path(models_dir)
        installed = (
            (path / "model_index.json").is_file()
            if self.download_type == "huggingface_snapshot"
            else path.is_file()
        )
        return {
            "model_id": self.model_id,
            "task": self.task,
            "display_name": self.display_name,
            "provider": self.provider,
            "size_bytes": self.size_bytes,
            "parameters_millions": self.parameters_millions,
            "recommended_vram_mb": self.recommended_vram_mb,
            "supports_cpu": self.supports_cpu,
            "license": self.license,
            "license_url": self.license_url,
            "commercial_use": self.commercial_use,
            "redistribution": self.redistribution,
            "license_acceptance_required": self.license_acceptance_required,
            "installed": installed,
            "compatible": (
                None
                if available_vram_mb is None
                else available_vram_mb + 128 >= self.recommended_vram_mb
            ),
        }


class ModelCatalog:
    """Read model manifests and create lazy provider factories."""

    def __init__(self, specs: tuple[ModelSpec, ...]) -> None:
        self._specs = {spec.model_id: spec for spec in specs}

    @classmethod
    def builtin(cls) -> ModelCatalog:
        path = Path(__file__).parents[1] / "resources" / "models" / "catalog.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(tuple(ModelSpec.from_dict(item) for item in payload["models"]))

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._specs[model_id]
        except KeyError as exc:
            raise KyvenError(
                code=ErrorCode.MODEL_NOT_FOUND,
                message=f"Model is not present in the trusted catalog: {model_id}",
                suggested_action="List available Kyven models and choose one of them.",
            ) from exc

    def list(self, task: str | None = None) -> tuple[ModelSpec, ...]:
        return tuple(
            spec for spec in self._specs.values() if task is None or spec.task == task
        )

    def registry(self, models_dir: Path, device: str) -> ProviderRegistry:
        from kyven.inpaint.providers.big_lama import BigLamaProvider
        from kyven.inpaint.providers.lama import LamaOnnxProvider
        from kyven.refine.providers.vitmatte import VitMatteProvider
        from kyven.segment.providers.sam2 import Sam2Provider

        registry = ProviderRegistry()
        for spec in self.list("segment"):
            registry.register(
                spec.model_id,
                lambda spec=spec: Sam2Provider(
                    checkpoint=str(spec.path(models_dir)),
                    model_config=spec.config,
                    device=device,
                    expected_checksum=spec.sha256,
                    provider_id=spec.model_id,
                    display_name=spec.display_name,
                ),
            )
        resources = Path(__file__).parents[1] / "resources" / "models"
        for spec in self.list("refine"):
            registry.register(
                spec.model_id,
                lambda spec=spec: VitMatteProvider(
                    checkpoint=str(spec.path(models_dir)),
                    config=str(resources / spec.config),
                    preprocessor_config=str(resources / "vitmatte-preprocessor.json"),
                    device=device,
                    expected_checksum=spec.sha256,
                    provider_id=spec.model_id,
                    display_name=spec.display_name,
                    license_url=spec.license_url,
                    minimum_vram_mb=spec.recommended_vram_mb,
                ),
            )
        for spec in self.list("inpaint"):
            if spec.provider == "big-lama-torchscript":
                registry.register(
                    spec.model_id,
                    lambda spec=spec: BigLamaProvider(
                        checkpoint=str(spec.path(models_dir)),
                        expected_checksum=spec.sha256,
                        device=device,
                    ),
                )
            else:
                registry.register(
                    spec.model_id,
                    lambda spec=spec: LamaOnnxProvider(
                        checkpoint=str(spec.path(models_dir)),
                        expected_checksum=spec.sha256,
                        device=device,
                    ),
                )
        from kyven.inpaint.providers.sdxl import SdxlInpaintProvider

        for spec in self.list("generative_inpaint"):
            registry.register(
                spec.model_id,
                lambda spec=spec: SdxlInpaintProvider(
                    checkpoint=str(spec.path(models_dir)),
                    device=device,
                    provider_id=spec.model_id,
                    display_name=spec.display_name,
                    license_url=spec.license_url,
                    minimum_vram_mb=spec.recommended_vram_mb,
                ),
            )
        return registry

    def download(
        self,
        model_id: str,
        models_dir: Path,
        progress: Callable[[int, int], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Path:
        spec = self.get(model_id)
        models_dir.mkdir(parents=True, exist_ok=True)
        target = spec.path(models_dir)
        if spec.download_type == "huggingface_snapshot":
            return self._download_snapshot(spec, models_dir, target, progress, cancelled)
        if target.is_file() and self._sha256(target) == spec.sha256:
            return target
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{spec.filename}-", suffix=".download", dir=models_dir
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            downloaded = 0
            with urllib.request.urlopen(spec.source, timeout=30) as response, temporary.open("wb") as stream:
                while True:
                    if cancelled is not None and cancelled():
                        raise KyvenError(ErrorCode.CANCELLED, f"Download cancelled: {model_id}")
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        progress(downloaded, spec.size_bytes)
            if temporary.stat().st_size != spec.size_bytes:
                raise KyvenError(
                    code=ErrorCode.MODEL_NOT_FOUND,
                    message=f"Downloaded model size is invalid for {model_id}.",
                    suggested_action="Retry the download from the official source.",
                )
            checksum = self._sha256(temporary)
            if checksum != spec.sha256:
                raise KyvenError(
                    code=ErrorCode.MODEL_NOT_FOUND,
                    message=f"Downloaded model checksum is invalid for {model_id}.",
                    technical_detail=f"Expected {spec.sha256}, received {checksum}.",
                    suggested_action="Delete the download and retry from the official source.",
                )
            os.replace(temporary, target)
            return target
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _download_snapshot(
        spec: ModelSpec,
        models_dir: Path,
        target: Path,
        progress: Callable[[int, int], None] | None,
        cancelled: Callable[[], bool] | None,
    ) -> Path:
        if (target / "model_index.json").is_file():
            return target
        if cancelled is not None and cancelled():
            raise KyvenError(ErrorCode.CANCELLED, f"Download cancelled: {spec.model_id}")
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise KyvenError(
                ErrorCode.PROVIDER_UNAVAILABLE,
                "Hugging Face Hub support is not installed.",
                suggested_action="Run install.ps1 again before installing SDXL.",
            ) from exc
        temporary = Path(tempfile.mkdtemp(prefix=f".{spec.filename}-", dir=models_dir))
        monitor_stop = threading.Event()

        def monitor_download() -> None:
            while not monitor_stop.wait(0.5):
                if progress is None:
                    continue
                try:
                    downloaded = sum(
                        path.stat().st_size for path in temporary.rglob("*") if path.is_file()
                    )
                    progress(min(downloaded, max(1, spec.size_bytes - 1)), spec.size_bytes)
                except OSError:
                    time.sleep(0.1)

        monitor = threading.Thread(
            target=monitor_download,
            name=f"kyven-download-progress-{spec.model_id}",
            daemon=True,
        )
        try:
            if progress is not None:
                progress(1, max(1, spec.size_bytes))
            monitor.start()
            snapshot_download(
                repo_id=spec.source.removeprefix("hf://"),
                revision=spec.revision or None,
                local_dir=temporary,
                allow_patterns=list(spec.allow_patterns) or None,
            )
            if not (temporary / "model_index.json").is_file():
                raise KyvenError(
                    ErrorCode.MODEL_NOT_FOUND,
                    f"The pinned model snapshot is incomplete: {spec.model_id}",
                )
            if cancelled is not None and cancelled():
                raise KyvenError(ErrorCode.CANCELLED, f"Download cancelled: {spec.model_id}")
            if target.exists():
                shutil.rmtree(target)
            os.replace(temporary, target)
            if progress is not None:
                progress(spec.size_bytes, spec.size_bytes)
            return target
        finally:
            monitor_stop.set()
            if monitor.is_alive():
                monitor.join(timeout=1.0)
            if temporary.exists():
                shutil.rmtree(temporary)

    def remove(self, model_id: str, models_dir: Path) -> bool:
        spec = self.get(model_id)
        models_root = models_dir.resolve()
        target = spec.path(models_root).resolve()
        if target.parent != models_root:
            raise KyvenError(ErrorCode.INVALID_REQUEST, "Model path is outside the models directory.")
        if not target.exists():
            return False
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        return True

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
