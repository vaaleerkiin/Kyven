"""Shared trimap-aware tile planning and blending without model dependencies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from kyven.cancellation import CancellationToken
from kyven.errors import ErrorCode, KyvenError

TileBox = tuple[int, int, int, int]
Inference = Callable[[Image.Image, Image.Image], NDArray[np.float32]]


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    """Keep full-size context at image edges and the established tile grid."""

    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, tile - overlap))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


@dataclass(frozen=True, slots=True)
class TilePlan:
    boxes: tuple[TileBox, ...]
    total: int
    unknown_pixels: int
    grid_boxes: tuple[TileBox, ...]

    @property
    def metadata(self) -> dict[str, int]:
        return {
            "tiles": len(self.boxes),
            "tiles_total": self.total,
            "tiles_skipped": self.total - len(self.boxes),
            "unknown_pixels": self.unknown_pixels,
        }


def plan_tiles(
    trimap: Image.Image,
    tile_size: int,
    overlap: int,
    skip_known: bool,
    cancellation: CancellationToken,
) -> TilePlan:
    """Skip only boxes with no unknown pixels; retain every unknown pixel's context."""

    cancellation.raise_if_cancelled()
    pixels = np.asarray(trimap)
    unknown = (pixels > 1) & (pixels < 254)
    unknown_pixels = int(np.count_nonzero(unknown))
    tile = tile_size or max(trimap.size)
    xs = tile_starts(trimap.width, tile, overlap)
    ys = tile_starts(trimap.height, tile, overlap)
    boxes = []
    grid = []
    for y in ys:
        cancellation.raise_if_cancelled()
        for x in xs:
            box = (x, y, min(x + tile, trimap.width), min(y + tile, trimap.height))
            grid.append(box)
            if not skip_known or unknown[y : box[3], x : box[2]].any():
                boxes.append(box)
    return TilePlan(tuple(boxes), len(grid), unknown_pixels, tuple(grid))


def _weight(box: TileBox, canvas: tuple[int, int], overlap: int) -> NDArray[np.float32]:
    x, y, x1, y1 = box
    width, height = x1 - x, y1 - y
    weight = np.ones((height, width), dtype=np.float32)
    edge = min(overlap, height // 2, width // 2)
    if edge:
        ramp = np.linspace(0.05, 1.0, edge, dtype=np.float32)
        if x > 0:
            weight[:, :edge] *= ramp
        if x1 < canvas[0]:
            weight[:, -edge:] *= ramp[::-1]
        if y > 0:
            weight[:edge, :] *= ramp[:, None]
        if y1 < canvas[1]:
            weight[-edge:, :] *= ramp[::-1, None]
    return weight


def infer_tiles(
    image: Image.Image,
    trimap: Image.Image,
    plan: TilePlan,
    overlap: int,
    infer: Inference,
    cancellation: CancellationToken,
) -> NDArray[np.float32]:
    """Infer active boxes only, then restore known trimap pixels exactly."""

    cancellation.raise_if_cancelled()
    pixels = np.asarray(trimap)
    if not plan.boxes:
        cancellation.report_progress(0.90, "Trimap fully known; no model inference needed")
        return (pixels >= 254).astype(np.float32)
    single = plan.total == 1
    alpha = None if single else np.zeros((image.height, image.width), dtype=np.float32)
    weights = None if single else np.zeros_like(alpha)
    for index, box in enumerate(plan.boxes, 1):
        cancellation.raise_if_cancelled()
        cancellation.report_progress(
            0.25 + 0.65 * (index - 1) / len(plan.boxes),
            f"Refining tile {index}/{len(plan.boxes)}; {plan.total-len(plan.boxes)} known tiles skipped",
        )
        try:
            prediction = np.asarray(infer(image.crop(box), trimap.crop(box)), dtype=np.float32)
        except (RuntimeError, MemoryError) as exc:
            memory_errors = ("out of memory", "not enough memory", "can't allocate memory",
                             "cannot allocate memory")
            if isinstance(exc, MemoryError) or any(text in str(exc).lower() for text in memory_errors):
                raise KyvenError(
                    ErrorCode.INFERENCE_FAILED,
                    "Not enough memory for this refinement tile.",
                    technical_detail=str(exc), recoverable=True,
                    suggested_action="Choose Tiles and reduce Tile Size to 512 px or smaller, "
                                     "or use the CPU device.",
                ) from exc
            raise
        cancellation.raise_if_cancelled()
        x, y, x1, y1 = box
        if prediction.shape != (y1 - y, x1 - x) or not np.isfinite(prediction).all():
            raise KyvenError(ErrorCode.INFERENCE_FAILED,
                             "The refinement model returned an invalid tile alpha.")
        if single:
            alpha = prediction.copy()
        else:
            weight = _weight(box, image.size, overlap)
            alpha[y:y1, x:x1] += prediction * weight
            weights[y:y1, x:x1] += weight
    if weights is not None:
        np.divide(alpha, weights, out=alpha, where=weights > 0)
    alpha[pixels <= 1] = 0.0
    alpha[pixels >= 254] = 1.0
    np.clip(alpha, 0.0, 1.0, out=alpha)
    cancellation.report_progress(0.90, "Refinement tiles complete")
    return alpha
