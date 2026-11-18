"""Serializable models for alpha refinement jobs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kyven.errors import ErrorCode, KyvenError
from kyven.hashing import cached_file_sha256 as file_sha256
from kyven.media import validate_image_file
from kyven.segment.models import BoxPrompt, ExecutionProfile


@dataclass(frozen=True, slots=True)
class RefineRequest:
    """Host-neutral request for one source image and coarse mask or trimap."""

    source: Path
    mask: Path
    output: Path
    trimap_output: Path | None = None
    provider_id: str = "vitmatte-small-composition-1k"
    profile: ExecutionProfile = ExecutionProfile.BALANCED
    crop_mode: str = "full"
    roi: BoxPrompt | None = None
    context_padding: int = 128
    generate_trimap: bool = True
    foreground_radius: int = 10
    background_radius: int = 15
    tile_size: int = 0
    tile_overlap: int = 64

    def validate(self) -> None:
        for label, path in (("Source", self.source), ("Mask/trimap", self.mask)):
            validate_image_file(path, label)
        if self.crop_mode not in {"auto", "manual", "full"}:
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="Refinement crop mode must be auto, manual, or full.",
            )
        if self.crop_mode == "manual" and self.roi is None:
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="Manual refinement crop mode requires an ROI.",
            )
        if self.context_padding < 0:
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="Refinement context padding must be zero or greater.",
            )
        if self.foreground_radius < 0 or self.background_radius < 0:
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="Trimap radii must be zero or greater.",
            )
        if self.tile_size and self.tile_size < 128:
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="Refinement tile size must be zero or at least 128 pixels.",
            )
        if self.tile_overlap < 0 or (self.tile_size and self.tile_overlap * 2 >= self.tile_size):
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="Tile overlap must be smaller than half the tile size.",
            )

    def canonical(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "profile": self.profile.value,
            "crop_mode": self.crop_mode,
            "roi": self.roi.canonical() if self.roi else None,
            "context_padding": self.context_padding,
            "generate_trimap": self.generate_trimap,
            "foreground_radius": self.foreground_radius,
            "background_radius": self.background_radius,
            "tile_size": self.tile_size,
            "tile_overlap": self.tile_overlap,
        }

    def cache_key(self, provider_version: str, model_checksum: str) -> str:
        payload = {
            "guidance_version": "cattery-average-trimap-v3",
            "source_sha256": file_sha256(self.source),
            "mask_sha256": file_sha256(self.mask),
            "request": self.canonical(),
            "provider_version": provider_version,
            "model_checksum": model_checksum,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class RefinementCapabilities:
    provider_id: str
    display_name: str
    provider_version: str
    model_checksum: str
    license_name: str
    license_url: str
    supports_cpu: bool
    supports_tiling: bool
    minimum_vram_mb: int | None


@dataclass(slots=True)
class RefinePrediction:
    alpha: NDArray[np.float32]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RefineResult:
    output: Path
    trimap_output: Path | None
    cache_key: str
    metadata: dict[str, Any]
