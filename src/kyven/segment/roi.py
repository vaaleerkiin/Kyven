"""Processing-region helpers shared by image and video segmentation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from kyven.errors import ErrorCode, KyvenError
from kyven.segment.models import BoxPrompt, PointLabel, PointPrompt


@dataclass(frozen=True, slots=True)
class ResolvedRegion:
    x0: int
    y0: int
    x1: int
    y1: int
    source_width: int
    source_height: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def is_full_frame(self) -> bool:
        return (
            self.x0 == 0
            and self.y0 == 0
            and self.x1 == self.source_width
            and self.y1 == self.source_height
        )

    def metadata(self) -> dict[str, int]:
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "width": self.width,
            "height": self.height,
            "source_width": self.source_width,
            "source_height": self.source_height,
        }


def resolve_region(region: BoxPrompt, width: int, height: int) -> ResolvedRegion:
    resolved = ResolvedRegion(
        x0=max(0, min(width, math.floor(region.x0))),
        y0=max(0, min(height, math.floor(region.y0))),
        x1=max(0, min(width, math.ceil(region.x1))),
        y1=max(0, min(height, math.ceil(region.y1))),
        source_width=width,
        source_height=height,
    )
    if resolved.width <= 0 or resolved.height <= 0:
        raise KyvenError(
            code=ErrorCode.INVALID_REQUEST,
            message="The Processing ROI does not overlap the input image.",
            suggested_action="Move or reset the ROI inside the input format.",
        )
    return resolved


def translate_points(
    points: tuple[PointPrompt, ...],
    region: ResolvedRegion,
) -> tuple[PointPrompt, ...]:
    translated = []
    for point in points:
        inside = region.x0 <= point.x < region.x1 and region.y0 <= point.y < region.y1
        if not inside:
            if point.label is PointLabel.POSITIVE:
                raise KyvenError(
                    code=ErrorCode.INVALID_REQUEST,
                    message="A positive point is outside the Processing ROI.",
                    suggested_action="Move the positive point inside the ROI or enlarge the ROI.",
                )
            continue
        translated.append(
            PointPrompt(point.x - region.x0, point.y - region.y0, point.label)
        )
    return tuple(translated)


def translate_box(box: BoxPrompt | None, region: ResolvedRegion) -> BoxPrompt | None:
    if box is None:
        return None
    x0 = max(float(region.x0), box.x0)
    y0 = max(float(region.y0), box.y0)
    x1 = min(float(region.x1), box.x1)
    y1 = min(float(region.y1), box.y1)
    if x1 <= x0 or y1 <= y0:
        raise KyvenError(
            code=ErrorCode.INVALID_REQUEST,
            message="The model prompt box does not overlap the Processing ROI.",
        )
    return BoxPrompt(x0 - region.x0, y0 - region.y0, x1 - region.x0, y1 - region.y0)


def expand_mask(mask: np.ndarray, region: ResolvedRegion) -> np.ndarray:
    pixels = np.asarray(mask)
    expected = (region.height, region.width)
    if pixels.shape != expected:
        raise KyvenError(
            code=ErrorCode.INFERENCE_FAILED,
            message="The provider returned a mask with the wrong ROI dimensions.",
            technical_detail=f"Expected {expected}, received {pixels.shape}.",
        )
    full = np.zeros((region.source_height, region.source_width), dtype=pixels.dtype)
    full[region.y0 : region.y1, region.x0 : region.x1] = pixels
    return full
