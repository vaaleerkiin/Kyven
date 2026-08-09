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
from kyven.segment.output import (
    confidence_trimap,
    read_logits_npz,
    write_logits_npz_atomic,
    write_mask_png_atomic,
)
from kyven.segment.postprocess import fill_enclosed_holes
from kyven.segment.providers.registry import ProviderRegistry
from kyven.segment.roi import expand_mask, resolve_region, translate_box, translate_points


class VideoDirection(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class VideoCorrection:
    """Prompts applied to one conditioning frame in a SAM 2 tracking state."""

    frame: int
    points: tuple[PointPrompt, ...] = ()
    box: BoxPrompt | None = None


@dataclass(frozen=True, slots=True)
class VideoSegmentRequest:
    frames_dir: Path
    output_pattern: Path
    first_frame: int
    last_frame: int
    key_frame: int
    direction: VideoDirection
    corrections: tuple[VideoCorrection, ...] = ()
    points: tuple[PointPrompt, ...] = ()
    box: BoxPrompt | None = None
    roi: BoxPrompt | None = None
    rois: tuple[tuple[int, BoxPrompt], ...] = ()
    provider_id: str = "sam2.1-small"
    profile: ExecutionProfile = ExecutionProfile.BALANCED
    offload_video_to_cpu: bool = True
    offload_state_to_cpu: bool = True
    fill_holes: bool = True
    max_hole_area: int = 2_048
    raw_output_pattern: Path | None = None
    logits_output_pattern: Path | None = None
    trimap_output_pattern: Path | None = None
    confidence_width: float = 1.0

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
        effective_corrections = self.effective_corrections
        if not effective_corrections:
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="Video propagation requires at least one point or box prompt.",
            )
        correction_frames = [correction.frame for correction in effective_corrections]
        if len(set(correction_frames)) != len(correction_frames):
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="Video corrections contain duplicate frame entries.",
            )
        if any(not self.first_frame <= frame <= self.last_frame for frame in correction_frames):
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="Every correction frame must be inside the propagation range.",
            )
        if any(not correction.points and correction.box is None for correction in effective_corrections):
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="Every video correction requires at least one point or box prompt.",
            )
        if "%" not in self.output_pattern.name:
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="Video output_pattern must contain a printf-style frame placeholder.",
            )
        if self.raw_output_pattern is not None and "%" not in self.raw_output_pattern.name:
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="Video raw_output_pattern must contain a printf-style frame placeholder.",
            )
        for label, pattern in (
            ("logits_output_pattern", self.logits_output_pattern),
            ("trimap_output_pattern", self.trimap_output_pattern),
        ):
            if pattern is not None and "%" not in pattern.name:
                raise KyvenError(ErrorCode.INVALID_REQUEST, f"Video {label} must contain a frame placeholder.")
        if self.confidence_width < 0:
            raise KyvenError(ErrorCode.INVALID_REQUEST, "Confidence Width must be zero or greater.")
        if self.max_hole_area < 0:
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="Maximum hole area must be zero or greater.",
            )
        roi_frames = [frame for frame, _roi in self.rois]
        if len(set(roi_frames)) != len(roi_frames):
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="Animated Processing ROI contains duplicate frame entries.",
            )
        if any(not self.first_frame <= frame <= self.last_frame for frame in roi_frames):
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="Animated Processing ROI contains a frame outside the propagation range.",
            )
        if self.rois and set(roi_frames) != set(range(self.first_frame, self.last_frame + 1)):
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="Animated Processing ROI must contain exactly one entry for every frame.",
            )

    @property
    def key_index(self) -> int:
        return self.key_frame - self.first_frame

    @property
    def effective_corrections(self) -> tuple[VideoCorrection, ...]:
        """Return multi-frame prompts, falling back to the legacy key-frame fields."""

        if self.corrections:
            return self.corrections
        if self.points or self.box is not None:
            return (VideoCorrection(self.key_frame, self.points, self.box),)
        return ()

    def frame_number(self, index: int) -> int:
        return self.first_frame + index

    def output_for_index(self, index: int) -> Path:
        return Path(str(self.output_pattern) % self.frame_number(index))

    def raw_output_for_frame(self, frame: int) -> Path | None:
        if self.raw_output_pattern is None:
            return None
        return Path(str(self.raw_output_pattern) % frame)

    def logits_output_for_frame(self, frame: int) -> Path | None:
        return None if self.logits_output_pattern is None else Path(str(self.logits_output_pattern) % frame)

    def trimap_output_for_frame(self, frame: int) -> Path | None:
        return None if self.trimap_output_pattern is None else Path(str(self.trimap_output_pattern) % frame)

    def roi_for_frame(self, frame: int) -> BoxPrompt | None:
        return dict(self.rois).get(frame, self.roi)


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
        token.report_progress(0.03, "Preparing video propagation")
        propagate = getattr(provider, "propagate_video", None)
        if propagate is None:
            raise KyvenError(
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                message=f"Provider does not support video propagation: {request.provider_id}",
            )
        result = (
            propagate(request, token)
            if request.roi is None and not request.rois
            else self._run_with_roi(propagate, request, token)
        )
        final = self._postprocess_outputs(result, request, token)
        token.report_progress(1.0, "Propagation complete")
        return final

    @staticmethod
    def _postprocess_outputs(
        result: VideoSegmentResult,
        request: VideoSegmentRequest,
        token: CancellationToken,
    ) -> VideoSegmentResult:
        for index, output in enumerate(result.outputs):
            frame = result.first_frame + index
            raw_output = request.raw_output_for_frame(frame)
            if raw_output is not None:
                with Image.open(output) as image:
                    write_mask_png_atomic(raw_output, np.asarray(image.convert("L")))
            logits_output = request.logits_output_for_frame(frame)
            trimap_output = request.trimap_output_for_frame(frame)
            if logits_output is not None and trimap_output is not None and logits_output.is_file():
                write_mask_png_atomic(
                    trimap_output,
                    confidence_trimap(read_logits_npz(logits_output), request.confidence_width),
                )
        if not request.fill_holes:
            return result
        filled_holes = 0
        filled_pixels = 0
        for index, output in enumerate(result.outputs, start=1):
            token.raise_if_cancelled()
            with Image.open(output) as image:
                mask = np.asarray(image.convert("L"))
            filled = fill_enclosed_holes(mask, request.max_hole_area)
            write_mask_png_atomic(output, filled.mask)
            filled_holes += filled.filled_holes
            filled_pixels += filled.filled_pixels
            token.report_progress(
                0.85 + 0.14 * index / max(1, len(result.outputs)),
                f"Post-processing matte {index}/{len(result.outputs)}",
            )
        metadata = dict(result.metadata)
        metadata["postprocess"] = {
            "fill_holes": True,
            "max_hole_area": request.max_hole_area,
            "filled_holes": filled_holes,
            "filled_pixels": filled_pixels,
        }
        return replace(result, metadata=metadata)

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
        expected_frames = request.last_frame - request.first_frame + 1
        if len(frames) != expected_frames:
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message=(
                    f"Expected {expected_frames} tracking frames, found {len(frames)}."
                ),
            )
        regions = []
        for index, frame in enumerate(frames):
            with Image.open(frame) as image:
                frame_roi = request.roi_for_frame(request.frame_number(index))
                if frame_roi is None:
                    frame_roi = BoxPrompt(0, 0, image.width, image.height)
                regions.append(resolve_region(frame_roi, image.width, image.height))
        key_region = regions[request.key_index]

        def translated_correction(correction: VideoCorrection) -> VideoCorrection:
            region = regions[correction.frame - request.first_frame]
            points = translate_points(correction.points, region)
            box = translate_box(correction.box, region)
            # Every animated crop is normalized to the key-frame crop size.
            scale_x = key_region.width / region.width
            scale_y = key_region.height / region.height
            points = tuple(replace(point, x=point.x * scale_x, y=point.y * scale_y) for point in points)
            if box is not None:
                box = BoxPrompt(
                    box.x0 * scale_x,
                    box.y0 * scale_y,
                    box.x1 * scale_x,
                    box.y1 * scale_y,
                )
            return VideoCorrection(correction.frame, points, box)

        corrections = tuple(translated_correction(item) for item in request.effective_corrections)
        primary = next((item for item in corrections if item.frame == request.key_frame), corrections[0])
        if all(region.is_full_frame for region in regions):
            result = propagate(
                replace(
                    request,
                    corrections=corrections,
                    points=primary.points,
                    box=primary.box,
                    roi=None,
                    rois=(),
                ),
                token,
            )
            metadata = dict(result.metadata)
            metadata["processing_roi"] = key_region.metadata()
            metadata["animated_processing_roi"] = bool(request.rois)
            return replace(result, metadata=metadata)

        with tempfile.TemporaryDirectory(prefix="kyven-video-roi-") as directory:
            temporary_root = Path(directory)
            cropped_frames = temporary_root / "frames"
            cropped_frames.mkdir()
            target_size = (key_region.width, key_region.height)
            for index, frame in enumerate(frames):
                token.raise_if_cancelled()
                region = regions[index]
                with Image.open(frame) as image:
                    if image.size != (region.source_width, region.source_height):
                        raise KyvenError(
                            code=ErrorCode.INVALID_REQUEST,
                            message="All tracking frames must have the same dimensions.",
                        )
                    cropped = image.convert("RGB").crop(
                        (region.x0, region.y0, region.x1, region.y1)
                    )
                    if cropped.size != target_size:
                        cropped = cropped.resize(target_size, Image.Resampling.LANCZOS)
                    cropped.save(cropped_frames / frame.name, format="JPEG", quality=95)
                token.report_progress(
                    0.03 + 0.06 * (index + 1) / len(frames),
                    f"Preparing animated ROI {index + 1}/{len(frames)}",
                )

            prepared = replace(
                request,
                frames_dir=cropped_frames,
                output_pattern=temporary_root / "matte.%04d.png",
                logits_output_pattern=temporary_root / "logits.%04d.npz",
                trimap_output_pattern=None,
                corrections=corrections,
                points=primary.points,
                box=primary.box,
                roi=None,
                rois=(),
            )
            cropped_result = propagate(prepared, token)
            outputs = []
            for index in range(request.last_frame - request.first_frame + 1):
                cropped_output = prepared.output_for_index(index)
                if not cropped_output.is_file():
                    continue
                token.raise_if_cancelled()
                with Image.open(cropped_output) as image:
                    mask_image = image.convert("L")
                    region = regions[index]
                    if mask_image.size != (region.width, region.height):
                        mask_image = mask_image.resize(
                            (region.width, region.height),
                            Image.Resampling.BILINEAR,
                        )
                    mask = np.asarray(mask_image)
                output = request.output_for_index(index)
                write_mask_png_atomic(output, expand_mask(mask, region))
                cropped_logits = prepared.logits_output_for_frame(request.frame_number(index))
                logits_output = request.logits_output_for_frame(request.frame_number(index))
                if cropped_logits is not None and logits_output is not None and cropped_logits.is_file():
                    logits = read_logits_npz(cropped_logits)
                    if logits.shape != (region.height, region.width):
                        logits = np.asarray(
                            Image.fromarray(logits, mode="F").resize(
                                (region.width, region.height), Image.Resampling.BILINEAR
                            ), dtype=np.float32
                        )
                    full_logits = np.full(
                        (region.source_height, region.source_width), -100.0, dtype=np.float32
                    )
                    full_logits[region.y0 : region.y1, region.x0 : region.x1] = logits
                    write_logits_npz_atomic(logits_output, full_logits)
                outputs.append(output)

        metadata = dict(cropped_result.metadata)
        metadata["processing_roi"] = key_region.metadata()
        metadata["animated_processing_roi"] = bool(request.rois)
        metadata["processing_roi_frames"] = len(regions)
        return VideoSegmentResult(
            outputs=tuple(outputs),
            first_frame=cropped_result.first_frame,
            last_frame=cropped_result.last_frame,
            key_frame=cropped_result.key_frame,
            direction=cropped_result.direction,
            metadata=metadata,
        )
