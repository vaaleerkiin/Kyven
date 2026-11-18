"""One processing-crop contract for refinement and its CPU tile preview."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from kyven.segment.models import BoxPrompt
from kyven.segment.roi import ResolvedRegion, resolve_region


def full_region(width: int, height: int) -> ResolvedRegion:
    return ResolvedRegion(0, 0, width, height, width, height)


def with_context(region: ResolvedRegion, padding: int) -> ResolvedRegion:
    """Expand cached trimap bounds without rescanning the image."""

    return replace(region, x0=max(0, region.x0-padding), y0=max(0, region.y0-padding),
                   x1=min(region.source_width, region.x1+padding),
                   y1=min(region.source_height, region.y1+padding))


def refine_region(
    trimap: np.ndarray, crop_mode: str, roi: BoxPrompt | None, padding: int,
) -> ResolvedRegion:
    """Use Manual ROI exactly, Auto trimap bounds plus padding, or the full frame."""

    height, width = trimap.shape
    if roi is not None and crop_mode in {"manual", "full"}:
        return resolve_region(roi, width, height)
    if crop_mode != "auto":
        return full_region(width, height)
    ys, xs = np.nonzero(trimap > 0)
    if not xs.size:
        return full_region(width, height)
    bounds = ResolvedRegion(
        int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1,
        width, height,
    )
    return with_context(bounds, padding)
