"""Lazy MEMatte provider for high-resolution trimap refinement."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from kyven.cancellation import CancellationToken
from kyven.errors import ErrorCode, KyvenError
from kyven.hashing import cached_file_sha256
from kyven.refine.models import RefinementCapabilities, RefinePrediction, RefineRequest
from kyven.refine.providers.base import RefinementProvider
from kyven.refine.providers.vitmatte import VitMatteProvider


class MEMatteProvider(RefinementProvider):
    """Official MEMatte ViT-S AIM-500 checkpoint with adaptive token routing."""

    def __init__(
        self,
        checkpoint: str,
        device: str = "auto",
        expected_checksum: str | None = None,
        provider_id: str = "mematte-vits-aim500",
        display_name: str = "MEMatte ViT-S (AIM-500)",
        license_url: str = "https://github.com/linyiheng123/MEMatte",
        minimum_vram_mb: int = 4096,
        variant: str = "small",
    ) -> None:
        self._checkpoint = Path(checkpoint)
        self._requested_device = device
        self._expected_checksum = expected_checksum
        self._provider_id = provider_id
        self._display_name = display_name
        self._license_url = license_url
        self._minimum_vram_mb = minimum_vram_mb
        self._variant = variant
        self._checksum: str | None = None
        self._model: Any | None = None
        self._resolved_device = "unresolved"

    def _checksum_if_present(self) -> str:
        if self._checksum is None:
            self._checksum = (
                cached_file_sha256(self._checkpoint) if self._checkpoint.is_file() else "missing"
            )
        return self._checksum

    @property
    def capabilities(self) -> RefinementCapabilities:
        return RefinementCapabilities(
            provider_id=self._provider_id,
            display_name=self._display_name,
            provider_version="2",
            model_checksum=self._checksum_if_present(),
            license_name="MIT",
            license_url=self._license_url,
            supports_cpu=True,
            supports_tiling=True,
            minimum_vram_mb=self._minimum_vram_mb,
        )

    @staticmethod
    def _token_limit(request: RefineRequest) -> int:
        return {
            "low_memory": 8_000,
            "balanced": 18_500,
            "quality": 32_000,
        }[request.profile.value]

    def _load(self, request: RefineRequest):
        if self._model is not None:
            self._model.backbone.max_number_token = self._token_limit(request)
            return self._model
        if not self._checkpoint.is_file():
            raise KyvenError(
                ErrorCode.MODEL_NOT_FOUND,
                f"MEMatte checkpoint was not found: {self._checkpoint}",
                suggested_action="Install MEMatte from Kyven Model Manager.",
            )
        checksum = self._checksum_if_present()
        if self._expected_checksum and checksum.lower() != self._expected_checksum.lower():
            raise KyvenError(
                ErrorCode.MODEL_NOT_FOUND,
                "The MEMatte checkpoint checksum does not match the trusted manifest.",
            )
        try:
            import torch

            from kyven.refine.mematte_model import MEMatteInference

            self._resolved_device = (
                "cuda"
                if self._requested_device == "auto" and torch.cuda.is_available()
                else self._requested_device
            )
            if self._resolved_device == "auto":
                self._resolved_device = "cpu"
            model = MEMatteInference(
                max_number_token=self._token_limit(request), variant=self._variant
            )
            payload = torch.load(self._checkpoint, map_location="cpu", weights_only=True)
            state = payload.get("model", payload)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"Checkpoint architecture mismatch; missing={missing}, unexpected={unexpected}"
                )
            model.to(self._resolved_device).eval()
            self._model = model
            return model
        except ImportError as exc:
            raise KyvenError(
                ErrorCode.DEPENDENCY_MISSING,
                "The MEMatte runtime is not installed.",
                technical_detail=str(exc),
                suggested_action="Run install.cmd again.",
            ) from exc
        except KyvenError:
            raise
        except Exception as exc:
            raise KyvenError(
                ErrorCode.INFERENCE_FAILED,
                "MEMatte could not be loaded.",
                technical_detail=str(exc),
            ) from exc

    def _infer(
        self,
        image: Image.Image,
        trimap: Image.Image,
        request: RefineRequest,
        cancellation: CancellationToken,
    ) -> np.ndarray:
        import torch

        model = self._load(request)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        trimap_array = np.asarray(trimap, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        trimap_tensor = torch.from_numpy(trimap_array).unsqueeze(0).unsqueeze(0)
        image_tensor = image_tensor.to(self._resolved_device)
        trimap_tensor = trimap_tensor.to(self._resolved_device)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self._resolved_device == "cuda"
            else nullcontext()
        )
        cancellation.raise_if_cancelled()
        with torch.inference_mode(), autocast:
            alpha = model(image_tensor, trimap_tensor)[0, 0]
        return alpha.float().cpu().numpy()

    def predict(self, request: RefineRequest, cancellation: CancellationToken) -> RefinePrediction:
        request.validate()
        if self._model is None:
            cancellation.report_progress(0.18, "Loading MEMatte model")
        else:
            cancellation.report_progress(0.18, "Reusing loaded MEMatte model")
        self._load(request)
        image = Image.open(request.source).convert("RGB")
        trimap = Image.open(request.mask).convert("L")
        if image.size != trimap.size:
            raise KyvenError(ErrorCode.INVALID_REQUEST, "Source and trimap dimensions must match.")
        tile = request.tile_size
        if not tile or (image.width <= tile and image.height <= tile):
            cancellation.report_progress(0.25, "Running MEMatte inference")
            alpha = self._infer(image, trimap, request, cancellation)
            tiles = 1
        else:
            alpha_sum = np.zeros((image.height, image.width), dtype=np.float32)
            weights = np.zeros_like(alpha_sum)
            x_starts = VitMatteProvider._starts(image.width, tile, request.tile_overlap)
            y_starts = VitMatteProvider._starts(image.height, tile, request.tile_overlap)
            total = len(x_starts) * len(y_starts)
            tiles = 0
            for y in y_starts:
                for x in x_starts:
                    cancellation.raise_if_cancelled()
                    box = (x, y, min(x + tile, image.width), min(y + tile, image.height))
                    prediction = self._infer(
                        image.crop(box), trimap.crop(box), request, cancellation
                    )
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
                        0.25 + 0.65 * tiles / total, f"Refining tile {tiles}/{total}"
                    )
            alpha = alpha_sum / np.maximum(weights, 1e-6)
        trimap_pixels = np.asarray(trimap)
        alpha = np.where(trimap_pixels >= 254, 1.0, np.where(trimap_pixels <= 1, 0.0, alpha))
        cancellation.report_progress(0.90, "MEMatte inference complete")
        return RefinePrediction(
            alpha=np.asarray(np.clip(alpha, 0.0, 1.0), dtype=np.float32),
            metadata={
                "provider": self._provider_id,
                "device": self._resolved_device,
                "profile": request.profile.value,
                "tiles": tiles,
                "tile_size": request.tile_size,
                "tile_overlap": request.tile_overlap,
                "max_number_token": self._token_limit(request),
                "variant": self._variant,
            },
        )

    def unload(self) -> None:
        self._model = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
