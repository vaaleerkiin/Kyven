"""Models and service for SAM 2 video-mask propagation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from kyven.cancellation import CancellationToken
from kyven.errors import ErrorCode, KyvenError
from kyven.segment.models import BoxPrompt, ExecutionProfile, PointPrompt
from kyven.segment.providers.registry import ProviderRegistry


class VideoDirection(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class VideoSegmentRequest:
    frames_dir: Path
    output_pattern: Path
    first_frame: int
    last_frame: int
    key_frame: int
    direction: VideoDirection
    points: tuple[PointPrompt, ...] = ()
    box: BoxPrompt | None = None
    provider_id: str = "sam2.1-small"
    profile: ExecutionProfile = ExecutionProfile.BALANCED
    offload_video_to_cpu: bool = True
    offload_state_to_cpu: bool = True

    def validate(self) -> None:
        if not self.frames_dir.is_dir():
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message=f"Video frame directory does not exist: {self.frames_dir}",
            )
        if self.last_frame < self.first_frame:
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="Video range Last must be greater than or equal to First.",
            )
        if not self.first_frame <= self.key_frame <= self.last_frame:
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="The key frame must be inside the propagation range.",
            )
        if not self.points and self.box is None:
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="Video propagation requires at least one point or box prompt.",
            )
        if "%" not in self.output_pattern.name:
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="Video output_pattern must contain a printf-style frame placeholder.",
            )

    @property
    def key_index(self) -> int:
        return self.key_frame - self.first_frame

    def frame_number(self, index: int) -> int:
        return self.first_frame + index

    def output_for_index(self, index: int) -> Path:
        return Path(str(self.output_pattern) % self.frame_number(index))


@dataclass(frozen=True, slots=True)
class VideoSegmentResult:
    outputs: tuple[Path, ...]
    first_frame: int
    last_frame: int
    key_frame: int
    direction: VideoDirection
    metadata: dict[str, Any]


class VideoSegmentService:
    def __init__(self, registry: ProviderRegistry) -> None:
        self._registry = registry

    def run(
        self,
        request: VideoSegmentRequest,
        cancellation: CancellationToken | None = None,
    ) -> VideoSegmentResult:
        request.validate()
        token = cancellation or CancellationToken()
        token.raise_if_cancelled()
        provider = self._registry.activate(request.provider_id)
        propagate = getattr(provider, "propagate_video", None)
        if propagate is None:
            raise KyvenError(
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                message=f"Provider does not support video propagation: {request.provider_id}",
            )
        return propagate(request, token)
