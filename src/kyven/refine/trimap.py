"""Dependency-light trimap generation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def normalize_trimap(pixels: NDArray[np.uint8] | NDArray[np.float32]) -> NDArray[np.uint8]:
    """Use Cattery ViTMatte's 0.01/0.99 cutoffs for three-state guidance."""

    values = np.asarray(pixels)
    return np.where(values > 0.99 * 255, 255,
                    np.where(values < 0.01 * 255, 0, 128)).astype(np.uint8)


def _box_extreme(pixels: NDArray[np.uint8], radius: int, *, maximum: bool) -> NDArray[np.uint8]:
    """Separable min/max filter with edge extension and linear time/memory."""

    if radius == 0:
        return np.asarray(pixels)
    extreme = np.maximum if maximum else np.minimum
    size = radius * 2 + 1
    result = np.asarray(pixels)
    for axis in (0, 1):
        ordered = np.moveaxis(result, axis, -1)
        length = ordered.shape[-1]
        extra = (-(length + 2 * radius)) % size
        padded = np.pad(ordered, ((0, 0), (radius, radius + extra)), mode="edge")
        blocks = padded.reshape(*padded.shape[:-1], -1, size)
        prefix = extreme.accumulate(blocks, axis=-1).reshape(padded.shape)
        suffix = extreme.accumulate(blocks[..., ::-1], axis=-1)[..., ::-1].reshape(padded.shape)
        filtered = extreme(suffix[..., :length], prefix[..., size - 1 : size - 1 + length])
        result = np.moveaxis(filtered, -1, axis)
    return result


def generate_trimap(
    mask: NDArray[np.uint8],
    foreground_radius: int,
    background_radius: int,
) -> NDArray[np.uint8]:
    """Build a three-state trimap from a coarse binary or grayscale mask."""

    # Match Cattery: erode/dilate the original mask, average, then quantize.
    # Binarizing before morphology changes soft edges and faint background noise.
    pixels = np.asarray(mask, dtype=np.uint8)
    eroded = _box_extreme(pixels, foreground_radius, maximum=False)
    dilated = _box_extreme(pixels, background_radius, maximum=True)
    average = (eroded.astype(np.float32) + dilated.astype(np.float32)) * 0.5
    return normalize_trimap(average)
