"""Resolution-robust Big-LaMa TorchScript inpainting provider."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from kyven.cancellation import CancellationToken
from kyven.errors import ErrorCode, KyvenError
from kyven.inpaint.models import InpaintCapabilities, InpaintPrediction, InpaintRequest
from kyven.inpaint.providers.base import InpaintProvider
from kyven.inpaint.refinement import refine_big_lama


class BigLamaProvider(InpaintProvider):
    """Run the full resolution-robust Big-LaMa generator through TorchScript."""

    def __init__(self, checkpoint: str, expected_checksum: str, device: str = "auto") -> None:
        self._checkpoint = Path(checkpoint)
        self._expected_checksum = expected_checksum
        self._requested_device = device
        self._device = "cpu"
        self._model = None

    @property
    def capabilities(self) -> InpaintCapabilities:
        return InpaintCapabilities(
            provider_id="big-lama-native",
            display_name="Big-LaMa Native",
            provider_version="2",
            model_checksum=self._expected_checksum,
            license_name="Apache-2.0",
            license_url="https://github.com/advimman/lama/blob/main/LICENSE",
            supports_cpu=True,
            minimum_vram_mb=4096,
        )

    def _load(self):
        if self._model is not None:
            return self._model
        if not self._checkpoint.is_file():
            raise KyvenError(
                ErrorCode.MODEL_NOT_FOUND,
                f"Big-LaMa model is not installed: {self._checkpoint}",
                suggested_action="Run install.ps1 again and select Big-LaMa Native.",
            )
        digest = hashlib.sha256(self._checkpoint.read_bytes()).hexdigest()
        if digest != self._expected_checksum:
            raise KyvenError(
                ErrorCode.MODEL_NOT_FOUND,
                "The installed Big-LaMa model checksum is invalid.",
                suggested_action="Delete big-lama.pt and reinstall the model.",
            )
        try:
            import torch
        except ImportError as exc:
            raise KyvenError(
                ErrorCode.DEPENDENCY_MISSING,
                "PyTorch is required for Big-LaMa Native.",
                suggested_action="Run install.ps1 again to install the Kyven runtime.",
            ) from exc
        use_cuda = self._requested_device != "cpu" and torch.cuda.is_available()
        self._device = "cuda" if use_cuda else "cpu"
        try:
            self._model = torch.jit.load(str(self._checkpoint), map_location=self._device).eval()
            for parameter in self._model.parameters():
                parameter.requires_grad_(False)
        except Exception as exc:
            raise KyvenError(
                ErrorCode.MODEL_NOT_FOUND,
                "Big-LaMa could not be loaded.",
                technical_detail=str(exc),
            ) from exc
        return self._model

    def predict(self, request: InpaintRequest, cancellation: CancellationToken) -> InpaintPrediction:
        cancellation.raise_if_cancelled()
        model = self._load()
        import torch

        with Image.open(request.source) as source_file, Image.open(request.mask) as mask_file:
            image = np.asarray(source_file.convert("RGB"), dtype=np.float32) / 255.0
            mask = (np.asarray(mask_file.convert("L"), dtype=np.uint8) >= 128).astype(np.float32)
        height, width = mask.shape
        padded_height = (height + 7) // 8 * 8
        padded_width = (width + 7) // 8 * 8
        image = np.pad(image, ((0, padded_height - height), (0, padded_width - width), (0, 0)), mode="reflect")
        mask = np.pad(mask, ((0, padded_height - height), (0, padded_width - width)), mode="constant")
        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))[None]).to(self._device)
        mask_tensor = torch.from_numpy(mask[None, None]).to(self._device)
        refined = request.quality_mode == "refined"
        if refined and height * width > {
            "low_memory": 700_000,
            "balanced": 1_500_000,
            "quality": 2_500_000,
        }[request.profile.value]:
            raise KyvenError(
                ErrorCode.INFERENCE_FAILED,
                "The ROI is too large for the selected Big-LaMa Refined memory profile.",
                suggested_action="Use Auto ROI, reduce the ROI, or select Standard quality.",
            )
        cancellation.report_progress(
            0.45,
            "Big-LaMa refinement" if refined else "Big-LaMa native inpainting",
        )
        try:
            if refined:
                result = refine_big_lama(
                    model,
                    image_tensor,
                    mask_tensor,
                    steps=request.refinement_steps,
                    strength=request.refinement_strength,
                    max_scales=request.refinement_scales,
                    cancellation=cancellation,
                )
            else:
                with torch.inference_mode():
                    result = model(image_tensor, mask_tensor)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if self._device == "cuda":
                    torch.cuda.empty_cache()
                raise KyvenError(
                    ErrorCode.INFERENCE_FAILED,
                    "Big-LaMa ran out of GPU memory for this ROI.",
                    suggested_action="Use Auto ROI, reduce the ROI, or choose the low_memory profile.",
                ) from exc
            raise KyvenError(
                ErrorCode.INFERENCE_FAILED,
                "Big-LaMa inference failed.",
                technical_detail=str(exc),
                suggested_action="Retry with Standard quality or a smaller Auto ROI.",
            ) from exc
        rgb = result[0, :, :height, :width].permute(1, 2, 0).detach().float().cpu().numpy()
        return InpaintPrediction(
            rgb=np.clip(rgb * 255.0, 0, 255).astype(np.uint8),
            metadata={
                "device": self._device,
                "model_input": [padded_width, padded_height],
                "native_resolution": True,
                "quality_mode": request.quality_mode,
                "refinement_steps": request.refinement_steps if refined else 0,
                "refinement_scales": request.refinement_scales if refined else 0,
            },
        )

    def unload(self) -> None:
        self._model = None
        if self._device == "cuda":
            try:
                import torch

                torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass
