"""Shared CPU-only input-mask preparation for Inpaint and its preview."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter


def mask_filter_size(radius: int) -> int:
    return max(3, abs(radius) * 2 + 1) | 1


def grow_mask(mask: NDArray[np.uint8], radius: int) -> NDArray[np.uint8]:
    if radius == 0:
        return np.asarray(mask, dtype=np.uint8)
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L")
    filter_size = mask_filter_size(radius)
    filtered = (
        image.filter(ImageFilter.MaxFilter(filter_size))
        if radius > 0
        else image.filter(ImageFilter.MinFilter(filter_size))
    )
    return np.asarray(filtered, dtype=np.uint8)


def prepare_inpaint_masks(
    mask: NDArray[np.uint8],
    *,
    preprocess: bool,
    invert: bool,
    threshold: float,
    model_grow: int,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Return the binary model mask and clean soft input mask for compositing."""

    source = np.asarray(mask, dtype=np.uint8)
    if not preprocess:
        return np.where(source >= 128, 255, 0).astype(np.uint8), source.copy()
    interpreted = 255 - source if invert else source
    cutoff = round(float(threshold) * 255.0)
    # Cattery LaMa treats every non-zero alpha sample as masked. Preserve the
    # normal >= cutoff behavior for explicit thresholds above zero.
    selected = interpreted > 0 if cutoff <= 0 else interpreted >= cutoff
    binary = np.where(selected, 255, 0).astype(np.uint8)
    model_mask = grow_mask(binary, int(model_grow))
    return model_mask, interpreted.copy()
