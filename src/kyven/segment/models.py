"""Serializable request and capability models for segmentation providers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kyven.errors import ErrorCode, KyvenError


class PointLabel(str, Enum):
    """Foreground or background point prompt."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


class ExecutionProfile(str, Enum):
    """Resource profile requested by a host adapter."""

    LOW_MEMORY = "low_memory"
    BALANCED = "balanced"
    QUALITY = "quality"


@dataclass(frozen=True, slots=True)
class PointPrompt:
    """A point prompt in source-image pixel coordinates."""

    x: float
    y: float
    label: PointLabel = PointLabel.POSITIVE

    def canonical(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y, "label": self.label.value}


@dataclass(frozen=True, slots=True)
class BoxPrompt:
    """A rectangular prompt in source-image pixel coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="The rectangle must have a positive width and height.",
                suggested_action="Check the rectangle coordinates.",
            )

    def canonical(self) -> dict[str, float]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


@dataclass(frozen=True, slots=True)
class SegmentRequest:
    """Host-neutral request for a single-frame segmentation."""

    source: Path
    output: Path
    points: tuple[PointPrompt, ...] = ()
    box: BoxPrompt | None = None
    roi: BoxPrompt | None = None
    provider_id: str = "sam2"
    profile: ExecutionProfile = ExecutionProfile.BALANCED
    multimask_output: bool = True
    fill_holes: bool = True
    max_hole_area: int = 2_048

    def validate(self) -> None:
        if not self.source.is_file():
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message=f"Source image does not exist: {self.source}",
                suggested_action="Choose an existing image file.",
            )
        if not self.points and self.box is None:
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="At least one point or box prompt is required.",
                suggested_action="Add a positive point or draw a prompt box.",
            )
        if self.max_hole_area < 0:
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="Maximum hole area must be zero or greater.",
            )

    def canonical(self) -> dict[str, Any]:
        """Return fields that affect inference output, excluding destination path."""

        return {
            "points": [point.canonical() for point in self.points],
            "box": self.box.canonical() if self.box else None,
            "roi": self.roi.canonical() if self.roi else None,
            "provider_id": self.provider_id,
            "profile": self.profile.value,
            "multimask_output": self.multimask_output,
            "fill_holes": self.fill_holes,
            "max_hole_area": self.max_hole_area,
        }

    def cache_key(self, provider_version: str, model_checksum: str) -> str:
        """Build a deterministic key from source bytes and inference inputs."""

        source_hasher = hashlib.sha256()
        with self.source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                source_hasher.update(chunk)
        payload = {
            "source_sha256": source_hasher.hexdigest(),
            "request": self.canonical(),
            "provider_version": provider_version,
            "model_checksum": model_checksum,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """Static provider metadata used for selection and diagnostics."""

    provider_id: str
    display_name: str
    provider_version: str
    model_family: str
    model_variant: str
    model_checksum: str
    license_name: str
    license_url: str
    supports_cpu: bool
    supports_points: bool
    supports_boxes: bool
    minimum_vram_mb: int | None
    supported_profiles: tuple[ExecutionProfile, ...]


@dataclass(slots=True)
class SegmentPrediction:
    """A provider prediction before it is persisted by the engine."""

    mask: NDArray[np.bool_] | NDArray[np.float32]
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SegmentResult:
    """Serializable result returned to CLI and host adapters."""

    output: Path
    score: float
    cache_key: str
    metadata: dict[str, Any]
