"""External-vector reprojection of the neural residual, never optical-flow generation."""

from __future__ import annotations

import numpy as np

from kyven.cancellation import CancellationToken


def reproject(previous: np.ndarray, motion: np.ndarray,
              cancellation: CancellationToken | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Backward gather: motion is current -> previous, in top-down pixel coordinates.

    Work in 64-row strips so temporary gather buffers do not scale with frame height.
    Invalid coordinates are rejected instead of creating repeated edge histories.
    """
    height, width = previous.shape[:2]
    output = np.empty_like(previous)
    valid = np.empty((height, width), dtype=bool)
    columns = np.arange(width, dtype=np.float32)[None, :]
    for start in range(0, height, 64):
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        end = min(height, start + 64)
        x = columns + motion[start:end, :, 0]
        y = np.arange(start, end, dtype=np.float32)[:, None] + motion[start:end, :, 1]
        valid[start:end] = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)
        x0, y0 = x.astype(np.int32), y.astype(np.int32)
        x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
        fx, fy = x - x0, y - y0
        if previous.ndim == 3:
            fx, fy = fx[..., None], fy[..., None]
        output[start:end] = ((1 - fy) * ((1 - fx) * previous[y0, x0]
                                        + fx * previous[y0, x1])
                             + fy * ((1 - fx) * previous[y1, x0] + fx * previous[y1, x1]))
    return output, valid


def difference(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Relative linear RGB difference, bounded sensibly around dark pixels."""
    return np.mean(np.abs(current - previous), axis=-1) / (
        0.1 + np.maximum(np.mean(np.abs(current), axis=-1),
                         np.mean(np.abs(previous), axis=-1)))


def stabilize(
    residual: np.ndarray, old_residual: np.ndarray, old_source: np.ndarray,
    source: np.ndarray, mask: np.ndarray, old_mask: np.ndarray,
    motion: np.ndarray, strength: float, rejection: float,
) -> np.ndarray:
    warped_source, valid = reproject(old_source, motion)
    warped_delta, _ = reproject(old_residual, motion)
    warped_mask, _ = reproject(old_mask, motion)
    confidence = np.clip(1 - difference(source, warped_source) / rejection, 0, 1)
    # Newly exposed pixels and changes to the artist mask must not inherit old effects.
    confidence *= np.clip(1 - np.abs(mask - warped_mask) * 4, 0, 1)
    weight = (strength * confidence * valid)[..., None]
    return residual * (1 - weight) + warped_delta * weight
