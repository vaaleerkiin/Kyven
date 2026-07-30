"""Dependency-light trimap generation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter


def normalize_trimap(pixels: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Quantize an artist trimap into background, unknown, and foreground."""

    values = np.asarray(pixels, dtype=np.uint8)
    return np.where(values >= 170, 255, np.where(values <= 85, 0, 128)).astype(np.uint8)


def generate_trimap(
    mask: NDArray[np.uint8],
    foreground_radius: int,
    background_radius: int,
) -> NDArray[np.uint8]:
    """Build a three-state trimap from a coarse binary or grayscale mask."""

    binary = (np.asarray(mask, dtype=np.uint8) >= 128).astype(np.uint8) * 255
    image = Image.fromarray(binary, mode="L")
    foreground = image
    background_limit = image
    if foreground_radius:
        foreground = image.filter(ImageFilter.MinFilter(foreground_radius * 2 + 1))
    if background_radius:
        background_limit = image.filter(ImageFilter.MaxFilter(background_radius * 2 + 1))
    foreground_pixels = np.asarray(foreground) >= 128
    possible_foreground = np.asarray(background_limit) >= 128
    return np.where(foreground_pixels, 255, np.where(possible_foreground, 128, 0)).astype(np.uint8)
