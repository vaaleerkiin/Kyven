"""Models and service for SAM 2 video-mask propagation."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from kyven.cancellation import CancellationToken
from kyven.errors import ErrorCode, KyvenError
from kyven.segment.models import BoxPrompt, ExecutionProfile, PointPrompt
from kyven.segment.output import write_mask_png_atomic
from kyven.segment.providers.registry import ProviderRegistry
from kyven.segment.roi import expand_mask, resolve_region, translate_box, translate_points


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
    roi: BoxPrompt | None = None
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
        if request.roi is None:
            return propagate(request, token)
        return self._run_with_roi(propagate, request, token)

    @staticmethod
    def _run_with_roi(propagate, request, token: CancellationToken) -> VideoSegmentResult:
        frames = sorted(
            path
            for path in request.frames_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg"}
        )
        if not frames:
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="The video frame directory contains no JPEG frames.",
            )
        with Image.open(frames[0]) as first_image:
            region = resolve_region(request.roi, first_image.width, first_image.height)
        points = translate_points(request.points, region)
        box = translate_box(request.box, region)
        if region.is_full_frame:
            result = propagate(replace(request, points=points, box=box, roi=None), token)
            metadata = dict(result.metadata)
            metadata["processing_roi"] = region.metadata()
            return replace(result, metadata=metadata)

        with tempfile.TemporaryDirectory(prefix="kyven-video-roi-") as directory:
            temporary_root = Path(directory)
            cropped_frames = temporary_root / "frames"
            cropped_frames.mkdir()
            for frame in frames:
                token.raise_if_cancelled()
                with Image.open(frame) as image:
                    if image.size != (region.source_width, region.source_height):
                        raise KyvenError(
                            code=ErrorCode.INVALID_REQUEST,
                            message="All tracking frames must have the same dimensions.",
                        )
                    image.convert("RGB").crop(
                        (region.x0, region.y0, region.x1, region.y1)
                    ).save(cropped_frames / frame.name, format="JPEG", quality=95)

            prepared = replace(
                request,
                frames_dir=cropped_frames,
                output_pattern=temporary_root / "matte.%04d.png",
                points=points,
                box=box,
                roi=None,
            )
            cropped_result = propagate(prepared, token)
            outputs = []
            for index in range(request.last_frame - request.first_frame + 1):
                cropped_output = prepared.output_for_index(index)
                if not cropped_output.is_file():
                    continue
                token.raise_if_cancelled()
                with Image.open(cropped_output) as image:
                    mask = np.asarray(image.convert("L"))
                output = request.output_for_index(index)
                write_mask_png_atomic(output, expand_mask(mask, region))
                outputs.append(output)

        metadata = dict(cropped_result.metadata)
        metadata["processing_roi"] = region.metadata()
        return VideoSegmentResult(
            outputs=tuple(outputs),
            first_frame=cropped_result.first_frame,
            last_frame=cropped_result.last_frame,
            key_frame=cropped_result.key_frame,
            direction=cropped_result.direction,
            metadata=metadata,
        )
