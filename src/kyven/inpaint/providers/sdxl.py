"""Optional, locally executed SDXL inpainting provider."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from kyven.cancellation import CancellationToken
from kyven.errors import ErrorCode, KyvenError
from kyven.inpaint.models import InpaintCapabilities, InpaintPrediction, InpaintRequest
from kyven.inpaint.providers.base import InpaintProvider


def _model_size(size: tuple[int, int], maximum: int) -> tuple[int, int]:
    """Fit an image into the quality limit while keeping SDXL dimensions valid."""

    width, height = size
    scale = maximum / max(width, height)
    width = max(64, round(width * scale / 8.0) * 8)
    height = max(64, round(height * scale / 8.0) * 8)
    return width, height


class SdxlInpaintProvider(InpaintProvider):
    """Diffusers SDXL Inpaint provider loaded only when its node is rendered."""

    def __init__(
        self,
        checkpoint: str,
        device: str = "auto",
        provider_id: str = "sdxl-inpainting-1.0",
        display_name: str = "SDXL Inpainting 1.0",
        license_url: str = "",
        minimum_vram_mb: int = 8192,
    ) -> None:
        self._checkpoint = Path(checkpoint)
        self._device = device
        self._provider_id = provider_id
        self._display_name = display_name
        self._license_url = license_url
        self._minimum_vram_mb = minimum_vram_mb
        self._pipeline: Any | None = None
        self._execution_device = "cpu"
        self._offloaded = False

    @property
    def capabilities(self) -> InpaintCapabilities:
        revision = "115134f363124c53c7d878647567d04daf26e41e"
        return InpaintCapabilities(
            provider_id=self._provider_id,
            display_name=self._display_name,
            provider_version="1",
            model_checksum=f"huggingface-revision:{revision}",
            license_name="CreativeML Open RAIL++-M",
            license_url=self._license_url,
            supports_cpu=True,
            minimum_vram_mb=self._minimum_vram_mb,
        )

    def _load(self, low_memory: bool) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        if not (self._checkpoint / "model_index.json").is_file():
            raise KyvenError(
                ErrorCode.MODEL_NOT_FOUND,
                f"SDXL Inpainting is not installed: {self._checkpoint}",
                suggested_action="Open Kyven Model Manager and install SDXL Inpainting 1.0.",
            )
        try:
            import torch
            from diffusers import StableDiffusionXLInpaintPipeline
        except ImportError as exc:
            raise KyvenError(
                ErrorCode.DEPENDENCY_MISSING,
                "The optional SDXL runtime is not installed.",
                suggested_action="Run install.ps1 again, then install SDXL in Model Manager.",
            ) from exc

        use_cuda = self._device != "cpu" and torch.cuda.is_available()
        dtype = torch.float16 if use_cuda else torch.float32
        try:
            pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                str(self._checkpoint),
                torch_dtype=dtype,
                local_files_only=True,
                use_safetensors=True,
                variant="fp16",
            )
            pipeline.enable_vae_tiling()
            pipeline.enable_attention_slicing()
            if use_cuda and low_memory:
                pipeline.enable_model_cpu_offload()
                self._execution_device = "cpu"
                self._offloaded = True
            elif use_cuda:
                pipeline.to("cuda")
                self._execution_device = "cuda"
            else:
                pipeline.to("cpu")
                self._execution_device = "cpu"
        except Exception as exc:
            raise KyvenError(
                ErrorCode.PROVIDER_UNAVAILABLE,
                "Could not load the installed SDXL Inpainting model.",
                technical_detail=str(exc),
                suggested_action="Use Low Memory on an 8 GB GPU, or reinstall the model.",
            ) from exc
        self._pipeline = pipeline
        return pipeline

    def predict(self, request: InpaintRequest, cancellation: CancellationToken) -> InpaintPrediction:
        cancellation.raise_if_cancelled()
        pipeline = self._load(request.low_memory)
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - guarded by _load
            raise KyvenError(ErrorCode.DEPENDENCY_MISSING, "PyTorch is required for SDXL.") from exc

        with Image.open(request.source) as source_file, Image.open(request.mask) as mask_file:
            source = source_file.convert("RGB")
            mask = mask_file.convert("L")
        original_size = source.size
        maximum = 768 if request.render_quality == "preview" else 1024
        size = _model_size(original_size, maximum)
        source_input = source.resize(size, Image.Resampling.LANCZOS)
        mask_input = mask.resize(size, Image.Resampling.NEAREST)
        steps = min(request.steps, 12) if request.render_quality == "preview" else request.steps
        seed = int(request.seed) & 0x7FFFFFFF
        generator = torch.Generator(device=self._execution_device).manual_seed(seed)

        def on_step_end(_pipe: Any, step: int, _timestep: Any, values: dict[str, Any]):
            cancellation.raise_if_cancelled()
            cancellation.report_progress(
                0.15 + (0.75 * (step + 1) / max(1, steps)),
                f"SDXL denoising {step + 1}/{steps}",
            )
            return values

        cancellation.report_progress(0.14, "Preparing SDXL Inpainting")
        try:
            result = pipeline(
                prompt=request.prompt or "clean background matching the surrounding scene",
                negative_prompt=request.negative_prompt or None,
                image=source_input,
                mask_image=mask_input,
                num_inference_steps=steps,
                guidance_scale=request.guidance_scale,
                strength=request.strength,
                generator=generator,
                width=size[0],
                height=size[1],
                callback_on_step_end=on_step_end,
            ).images[0]
        except KyvenError:
            raise
        except Exception as exc:
            raise KyvenError(
                ErrorCode.INFERENCE_FAILED,
                "SDXL Inpainting failed.",
                technical_detail=str(exc),
                suggested_action="Try Preview quality, Low Memory, or a tighter ROI.",
            ) from exc
        rgb = np.asarray(result.convert("RGB").resize(original_size, Image.Resampling.LANCZOS))
        return InpaintPrediction(
            rgb=rgb,
            metadata={
                "seed": seed,
                "steps": steps,
                "guidance_scale": request.guidance_scale,
                "strength": request.strength,
                "render_quality": request.render_quality,
                "model_input": list(size),
                "low_memory": request.low_memory,
            },
        )

    def unload(self) -> None:
        self._pipeline = None
        self._offloaded = False
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
