"""OpenCV's Apache-2.0 LaMa ONNX inpainting provider."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from kyven.cancellation import CancellationToken
from kyven.errors import ErrorCode, KyvenError
from kyven.inpaint.models import InpaintCapabilities, InpaintPrediction, InpaintRequest
from kyven.inpaint.providers.base import InpaintProvider


class LamaOnnxProvider(InpaintProvider):
    def __init__(self, checkpoint: str, expected_checksum: str, device: str = "auto") -> None:
        self._checkpoint = Path(checkpoint)
        self._expected_checksum = expected_checksum
        self._device = device
        self._session = None

    @property
    def capabilities(self) -> InpaintCapabilities:
        return InpaintCapabilities(
            provider_id="lama-2025jan-onnx",
            display_name="LaMa 2025-01 ONNX",
            provider_version="1",
            model_checksum=self._expected_checksum,
            license_name="Apache-2.0",
            license_url="https://huggingface.co/opencv/inpainting_lama",
            supports_cpu=True,
            minimum_vram_mb=None,
        )

    def _load(self):
        if self._session is not None:
            return self._session
        if not self._checkpoint.is_file():
            raise KyvenError(
                ErrorCode.MODEL_NOT_FOUND,
                f"LaMa model is not installed: {self._checkpoint}",
                suggested_action="Run the Kyven installer and select LaMa Inpaint.",
            )
        digest = hashlib.sha256(self._checkpoint.read_bytes()).hexdigest()
        if digest != self._expected_checksum:
            raise KyvenError(ErrorCode.MODEL_NOT_FOUND, "The installed LaMa model checksum is invalid.")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise KyvenError(
                ErrorCode.DEPENDENCY_MISSING,
                "ONNX Runtime is required for LaMa Inpaint.",
                suggested_action="Run install.ps1 again to install the Inpaint runtime.",
            ) from exc
        available = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if self._device != "cpu" and "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(self._checkpoint), sess_options=options, providers=providers
        )
        return self._session

    def predict(self, request: InpaintRequest, cancellation: CancellationToken) -> InpaintPrediction:
        cancellation.raise_if_cancelled()
        session = self._load()
        with Image.open(request.source) as source_file, Image.open(request.mask) as mask_file:
            image = np.asarray(source_file.convert("RGB"), dtype=np.float32) / 255.0
            mask = (np.asarray(mask_file.convert("L"), dtype=np.float32) >= 128).astype(np.float32)
        original_size = (image.shape[1], image.shape[0])
        inputs = session.get_inputs()
        model_height = inputs[0].shape[2]
        model_width = inputs[0].shape[3]
        restore_box: tuple[int, int, int, int] | None = None
        if isinstance(model_width, int) and isinstance(model_height, int):
            scale = min(model_width / original_size[0], model_height / original_size[1])
            resized_width = max(1, round(original_size[0] * scale))
            resized_height = max(1, round(original_size[1] * scale))
            resized_image = np.asarray(
                Image.fromarray((image * 255).astype(np.uint8)).resize(
                    (resized_width, resized_height), Image.Resampling.LANCZOS
                ),
                dtype=np.float32,
            ) / 255.0
            resized_mask = np.asarray(
                Image.fromarray((mask * 255).astype(np.uint8)).resize(
                    (resized_width, resized_height), Image.Resampling.NEAREST
                ),
                dtype=np.float32,
            ) / 255.0
            left = (model_width - resized_width) // 2
            top = (model_height - resized_height) // 2
            right = model_width - resized_width - left
            bottom = model_height - resized_height - top
            image = np.pad(
                resized_image,
                ((top, bottom), (left, right), (0, 0)),
                mode="edge",
            )
            mask = np.pad(
                resized_mask,
                ((top, bottom), (left, right)),
                mode="constant",
            )
            restore_box = (left, top, left + resized_width, top + resized_height)
        elif request.processing_size:
            scale = min(1.0, float(request.processing_size) / max(original_size))
            if scale < 1.0:
                size = (max(8, round(original_size[0] * scale)), max(8, round(original_size[1] * scale)))
                image = np.asarray(Image.fromarray((image * 255).astype(np.uint8)).resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
                mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize(size, Image.Resampling.NEAREST), dtype=np.float32) / 255.0
        feed = {
            inputs[0].name: np.transpose(image, (2, 0, 1))[None].astype(np.float32),
            inputs[1].name: mask[None, None].astype(np.float32),
        }
        cancellation.report_progress(0.45, "LaMa inpainting")
        result = np.asarray(session.run(None, feed)[0])
        if result.ndim == 4:
            result = result[0]
        if result.shape[0] == 3:
            result = np.transpose(result, (1, 2, 0))
        if result.max(initial=0) <= 1.5:
            result = result * 255.0
        rgb = np.clip(result, 0, 255).astype(np.uint8)
        if restore_box is not None:
            left, top, right, bottom = restore_box
            rgb = rgb[top:bottom, left:right]
        if (rgb.shape[1], rgb.shape[0]) != original_size:
            rgb = np.asarray(Image.fromarray(rgb).resize(original_size, Image.Resampling.LANCZOS))
        return InpaintPrediction(
            rgb=rgb,
            metadata={
                "execution_providers": session.get_providers(),
                "model_input": [int(model_width), int(model_height)],
                "aspect_preserved": restore_box is not None,
            },
        )

    def unload(self) -> None:
        self._session = None
