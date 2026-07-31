"""Dependency-light trimap generation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def normalize_trimap(pixels: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Quantize an artist trimap into background, unknown, and foreground."""

    values = np.asarray(pixels, dtype=np.uint8)
    return np.where(values >= 170, 255, np.where(values <= 85, 0, 128)).astype(np.uint8)


def _box_counts(binary: NDArray[np.bool_], radius: int) -> NDArray[np.uint32]:
    """Count true pixels in an edge-extended square window in O(width * height)."""

    if radius == 0:
        return np.asarray(binary, dtype=np.uint32)
    padded = np.pad(binary, ((radius, radius), (radius, radius)), mode="edge")
    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.uint32)
    integral[1:, 1:] = padded
    np.cumsum(integral, axis=0, dtype=np.uint32, out=integral)
    np.cumsum(integral, axis=1, dtype=np.uint32, out=integral)
    size = radius * 2 + 1
    return (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )


def generate_trimap(
    mask: NDArray[np.uint8],
    foreground_radius: int,
    background_radius: int,
) -> NDArray[np.uint8]:
    """Build a three-state trimap from a coarse binary or grayscale mask."""

    binary = np.asarray(mask, dtype=np.uint8) >= 128
    foreground_pixels = (
        _box_counts(binary, foreground_radius) == (foreground_radius * 2 + 1) ** 2
    )
    possible_foreground = _box_counts(binary, background_radius) > 0
    return np.where(foreground_pixels, 255, np.where(possible_foreground, 128, 0)).astype(np.uint8)
