"""Host-neutral models for image inpainting jobs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kyven.errors import ErrorCode, KyvenError
from kyven.segment.models import BoxPrompt, ExecutionProfile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class InpaintRequest:
    source: Path
    mask: Path
    output: Path
    model_mask: Path | None = None
    mask_output: Path | None = None
    patch_output: Path | None = None
    provider_id: str = "lama-2025jan-onnx"
    profile: ExecutionProfile = ExecutionProfile.BALANCED
    crop_mode: str = "auto"
    roi: BoxPrompt | None = None
    context_padding: int = 128
    mask_grow: int = 12
    edge_color_match: float = 1.0
    mask_threshold: float = 0.5
    invert_mask: bool = False
    mask_channel: str = "luminance"
    processing_size: int = 0
    preprocess_mask: bool = True
    prompt: str = ""
    negative_prompt: str = ""
    seed: int = 0
    steps: int = 25
    guidance_scale: float = 6.0
    strength: float = 0.99
    low_memory: bool = True
    render_quality: str = "final"

    def validate(self) -> None:
        for label, path in (("Source", self.source), ("Mask", self.mask)):
            if not path.is_file():
                raise KyvenError(ErrorCode.INVALID_REQUEST, f"{label} image does not exist: {path}")
        if self.model_mask is not None and not self.model_mask.is_file():
            raise KyvenError(
                ErrorCode.INVALID_REQUEST,
                f"Model mask image does not exist: {self.model_mask}",
            )
        if self.crop_mode not in {"auto", "manual", "full"}:
            raise KyvenError(ErrorCode.INVALID_REQUEST, "Inpaint crop mode must be auto, manual, or full.")
        if self.crop_mode == "manual" and self.roi is None:
            raise KyvenError(ErrorCode.INVALID_REQUEST, "Manual inpaint crop mode requires an ROI.")
        if self.context_padding < 0:
            raise KyvenError(ErrorCode.INVALID_REQUEST, "Padding cannot be negative.")
        if not -128 <= self.mask_grow <= 128:
            raise KyvenError(ErrorCode.INVALID_REQUEST, "Model mask grow must be between -128 and 128 pixels.")
        if not 0.0 <= self.mask_threshold <= 1.0:
            raise KyvenError(ErrorCode.INVALID_REQUEST, "Mask threshold must be between 0 and 1.")
        if not 0.0 <= self.edge_color_match <= 1.0:
            raise KyvenError(ErrorCode.INVALID_REQUEST, "Edge color match must be between 0 and 1.")
        if self.mask_channel not in {"luminance", "alpha"}:
            raise KyvenError(ErrorCode.INVALID_REQUEST, "Mask channel must be luminance or alpha.")
        if self.processing_size and self.processing_size < 128:
            raise KyvenError(ErrorCode.INVALID_REQUEST, "Processing size must be zero or at least 128 pixels.")
        if not 1 <= self.steps <= 100:
            raise KyvenError(ErrorCode.INVALID_REQUEST, "Generative steps must be between 1 and 100.")
        if not 0.0 <= self.guidance_scale <= 20.0:
            raise KyvenError(ErrorCode.INVALID_REQUEST, "Guidance scale must be between 0 and 20.")
        if not 0.01 <= self.strength < 1.0:
            raise KyvenError(ErrorCode.INVALID_REQUEST, "Generative strength must be at least 0.01 and below 1.0.")
        if self.render_quality not in {"preview", "final"}:
            raise KyvenError(ErrorCode.INVALID_REQUEST, "Render quality must be preview or final.")

    def canonical(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "profile": self.profile.value,
            "crop_mode": self.crop_mode,
            "roi": self.roi.canonical() if self.roi else None,
            "context_padding": self.context_padding,
            "mask_grow": self.mask_grow,
            "edge_color_match": self.edge_color_match,
            "mask_threshold": self.mask_threshold,
            "invert_mask": self.invert_mask,
            "mask_channel": self.mask_channel,
            "processing_size": self.processing_size,
            "preprocess_mask": self.preprocess_mask,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "strength": self.strength,
            "low_memory": self.low_memory,
            "render_quality": self.render_quality,
        }

    def cache_key(self, provider_version: str, model_checksum: str) -> str:
        payload = {
            "source_sha256": _sha256(self.source),
            "mask_sha256": _sha256(self.mask),
            "model_mask_sha256": _sha256(self.model_mask) if self.model_mask else None,
            "request": self.canonical(),
            "provider_version": provider_version,
            "model_checksum": model_checksum,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class InpaintCapabilities:
    provider_id: str
    display_name: str
    provider_version: str
    model_checksum: str
    license_name: str
    license_url: str
    supports_cpu: bool
    minimum_vram_mb: int | None


@dataclass(slots=True)
class InpaintPrediction:
    rgb: NDArray[np.uint8]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InpaintResult:
    output: Path
    mask_output: Path | None
    patch_output: Path | None
    cache_key: str
    metadata: dict[str, Any]
