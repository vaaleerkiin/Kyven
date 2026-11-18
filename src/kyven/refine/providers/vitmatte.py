"""Lazy Hugging Face ViTMatte refinement provider."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from PIL import Image

from kyven.cancellation import CancellationToken
from kyven.errors import ErrorCode, KyvenError
from kyven.hashing import cached_file_sha256
from kyven.refine.models import RefinementCapabilities, RefinePrediction, RefineRequest
from kyven.refine.providers.base import RefinementProvider
from kyven.refine.tiling import infer_tiles, plan_tiles, tile_starts


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
        self._checksum = cached_file_sha256(self._checkpoint)
        return self._checksum

    @property
    def capabilities(self) -> RefinementCapabilities:
        return RefinementCapabilities(
            provider_id=self._provider_id,
            display_name=self._display_name,
            provider_version="3",
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
        return tile_starts(length, tile, overlap)

    def predict(self, request: RefineRequest, cancellation: CancellationToken) -> RefinePrediction:
        request.validate()
        with Image.open(request.source) as source_file, Image.open(request.mask) as mask_file:
            image = source_file.convert("RGB")
            trimap = mask_file.convert("L")
        if image.size != trimap.size:
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="Source and trimap dimensions must match.",
            )
        plan = plan_tiles(trimap, request.tile_size, request.tile_overlap,
                          request.skip_known_tiles, cancellation)
        if plan.boxes:
            loading = self._model is None or self._processor is None
            cancellation.report_progress(0.18, "Loading ViTMatte model" if loading
                                         else "Reusing loaded ViTMatte model")
            self._load()
        alpha = infer_tiles(image, trimap, plan, request.tile_overlap,
                            lambda rgb, guide: self._infer(rgb, guide, cancellation), cancellation)
        return RefinePrediction(
            alpha=alpha,
            metadata={
                "provider": self._provider_id,
                "device": self._resolved_device if plan.boxes else "not_used",
                "profile": request.profile.value,
                **plan.metadata,
                "tile_size": request.tile_size,
                "tile_overlap": request.tile_overlap,
                "skip_known_tiles": request.skip_known_tiles,
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
