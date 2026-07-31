"""Lazy Hugging Face ViTMatte refinement provider."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from kyven.cancellation import CancellationToken
from kyven.errors import ErrorCode, KyvenError
from kyven.refine.models import RefinementCapabilities, RefinePrediction, RefineRequest
from kyven.refine.providers.base import RefinementProvider


class VitMatteProvider(RefinementProvider):
    """ViTMatte provider that stays unloaded until first use."""

    def __init__(
        self,
        checkpoint: str,
        config: str,
        preprocessor_config: str,
        device: str = "auto",
        expected_checksum: str | None = None,
        provider_id: str = "vitmatte-small-composition-1k",
        display_name: str = "ViTMatte Small (Composition-1k)",
        license_url: str = "https://huggingface.co/hustvl/vitmatte-small-composition-1k",
        minimum_vram_mb: int = 4096,
    ) -> None:
        self._checkpoint = Path(checkpoint)
        self._config = Path(config)
        self._preprocessor_config = Path(preprocessor_config)
        self._requested_device = device
        self._expected_checksum = expected_checksum
        self._provider_id = provider_id
        self._display_name = display_name
        self._license_url = license_url
        self._minimum_vram_mb = minimum_vram_mb
        self._checksum: str | None = None
        self._model: Any | None = None
        self._processor: Any | None = None
        self._resolved_device = "unresolved"

    def _checksum_if_present(self) -> str:
        if self._checksum is not None:
            return self._checksum
        if not self._checkpoint.is_file():
            return "missing"
        digest = hashlib.sha256()
        with self._checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        self._checksum = digest.hexdigest()
        return self._checksum

    @property
    def capabilities(self) -> RefinementCapabilities:
        return RefinementCapabilities(
            provider_id=self._provider_id,
            display_name=self._display_name,
            provider_version="1",
            model_checksum=self._checksum_if_present(),
            license_name="Apache-2.0",
            license_url=self._license_url,
            supports_cpu=True,
            supports_tiling=True,
            minimum_vram_mb=self._minimum_vram_mb,
        )

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        if not self._checkpoint.is_file():
            raise KyvenError(
                code=ErrorCode.MODEL_NOT_FOUND,
                message=f"ViTMatte checkpoint was not found: {self._checkpoint}",
                suggested_action="Open Kyven > Model Manager in Nuke or run install.cmd.",
            )
        checksum = self._checksum_if_present()
        if self._expected_checksum and checksum.lower() != self._expected_checksum.lower():
            raise KyvenError(
                code=ErrorCode.MODEL_NOT_FOUND,
                message="The ViTMatte checkpoint checksum does not match the trusted manifest.",
            )
        try:
            import torch
            from safetensors.torch import load_model
            from transformers import VitMatteConfig, VitMatteForImageMatting, VitMatteImageProcessor

            config = VitMatteConfig.from_dict(json.loads(self._config.read_text(encoding="utf-8")))
            processor_settings = json.loads(self._preprocessor_config.read_text(encoding="utf-8"))
            model = VitMatteForImageMatting(config)
            load_model(model, str(self._checkpoint), strict=True)
            self._resolved_device = (
                "cuda" if self._requested_device == "auto" and torch.cuda.is_available()
                else self._requested_device
            )
            if self._resolved_device == "auto":
                self._resolved_device = "cpu"
            model.to(self._resolved_device).eval()
            self._model = model
            self._processor = VitMatteImageProcessor(**processor_settings)
            return model, self._processor
        except ImportError as exc:
            raise KyvenError(
                code=ErrorCode.DEPENDENCY_MISSING,
                message="The ViTMatte runtime is not installed.",
                technical_detail=str(exc),
                suggested_action="Run install.cmd again to install Transformers and Safetensors.",
            ) from exc
        except KyvenError:
            raise
        except Exception as exc:
            raise KyvenError(
                code=ErrorCode.INFERENCE_FAILED,
                message="ViTMatte could not be loaded.",
                technical_detail=str(exc),
                suggested_action="Verify the model and try the Low Memory profile.",
            ) from exc

    def _infer(self, image: Image.Image, trimap: Image.Image, cancellation: CancellationToken):
        import torch

        model, processor = self._load()
        inputs = processor(images=image, trimaps=trimap, return_tensors="pt")
        inputs = {name: value.to(self._resolved_device) for name, value in inputs.items()}
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self._resolved_device == "cuda"
            else nullcontext()
        )
        cancellation.raise_if_cancelled()
        with torch.inference_mode(), autocast:
            alpha = model(**inputs).alphas[0, 0]
        return alpha[: image.height, : image.width].float().cpu().numpy()

    @staticmethod
    def _starts(length: int, tile: int, overlap: int) -> list[int]:
        if length <= tile:
            return [0]
        step = tile - overlap
        starts = list(range(0, max(1, length - tile + 1), step))
        final = length - tile
        if starts[-1] != final:
            starts.append(final)
        return starts

    def predict(self, request: RefineRequest, cancellation: CancellationToken) -> RefinePrediction:
        request.validate()
        cancellation.report_progress(0.18, "Loading ViTMatte model")
        image = Image.open(request.source).convert("RGB")
        trimap = Image.open(request.mask).convert("L")
        if image.size != trimap.size:
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="Source and trimap dimensions must match.",
            )
        tile = request.tile_size
        if not tile or (image.width <= tile and image.height <= tile):
            cancellation.report_progress(0.25, "Running ViTMatte inference")
            alpha = self._infer(image, trimap, cancellation)
            tiles = 1
            cancellation.report_progress(0.90, "ViTMatte inference complete")
        else:
            alpha_sum = np.zeros((image.height, image.width), dtype=np.float32)
            weights = np.zeros_like(alpha_sum)
            tiles = 0
            y_starts = self._starts(image.height, tile, request.tile_overlap)
            x_starts = self._starts(image.width, tile, request.tile_overlap)
            total_tiles = len(y_starts) * len(x_starts)
            for y in y_starts:
                for x in x_starts:
                    cancellation.raise_if_cancelled()
                    box = (x, y, min(x + tile, image.width), min(y + tile, image.height))
                    prediction = self._infer(image.crop(box), trimap.crop(box), cancellation)
                    height, width = prediction.shape
                    weight = np.ones((height, width), dtype=np.float32)
                    edge = min(request.tile_overlap, height // 2, width // 2)
                    if edge:
                        ramp = np.linspace(0.05, 1.0, edge, dtype=np.float32)
                        if x > 0:
                            weight[:, :edge] *= ramp
                        if x + width < image.width:
                            weight[:, -edge:] *= ramp[::-1]
                        if y > 0:
                            weight[:edge, :] *= ramp[:, None]
                        if y + height < image.height:
                            weight[-edge:, :] *= ramp[::-1, None]
                    alpha_sum[y : y + height, x : x + width] += prediction * weight
                    weights[y : y + height, x : x + width] += weight
                    tiles += 1
                    cancellation.report_progress(
                        0.25 + 0.65 * tiles / total_tiles,
                        f"Refining tile {tiles}/{total_tiles}",
                    )
            alpha = alpha_sum / np.maximum(weights, 1e-6)
        trimap_pixels = np.asarray(trimap)
        alpha = np.where(trimap_pixels >= 254, 1.0, np.where(trimap_pixels <= 1, 0.0, alpha))
        return RefinePrediction(
            alpha=np.asarray(np.clip(alpha, 0.0, 1.0), dtype=np.float32),
            metadata={
                "provider": self._provider_id,
                "device": self._resolved_device,
                "profile": request.profile.value,
                "tiles": tiles,
                "tile_size": request.tile_size,
                "tile_overlap": request.tile_overlap,
            },
        )

    def unload(self) -> None:
        self._model = None
        self._processor = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
