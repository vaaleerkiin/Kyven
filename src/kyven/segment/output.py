"""Atomic image output helpers shared by image and video segmentation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from kyven.errors import ErrorCode, KyvenError


def write_mask_png_atomic(output: Path, mask: np.ndarray) -> None:
    pixels = np.asarray(mask)
    if pixels.ndim != 2:
        raise KyvenError(
            code=ErrorCode.INFERENCE_FAILED,
            message="The provider returned an invalid mask shape.",
            technical_detail=f"Expected 2 dimensions, received {pixels.shape}.",
        )
    if np.issubdtype(pixels.dtype, np.floating):
        pixels = np.clip(pixels, 0.0, 1.0) * 255.0
    elif np.issubdtype(pixels.dtype, np.bool_):
        pixels = pixels.astype(np.uint8) * 255
    else:
        pixels = np.asarray(pixels, dtype=np.uint8)
        if pixels.size and int(pixels.max()) <= 1:
            pixels = pixels * 255
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.stem}-",
            suffix=".png",
            dir=output.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
        Image.fromarray(pixels.astype(np.uint8), mode="L").save(temporary_path, format="PNG")
        os.replace(temporary_path, output)
    except Exception as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise KyvenError(
            code=ErrorCode.OUTPUT_FAILED,
            message=f"Could not write mask output: {output}",
            technical_detail=str(exc),
            recoverable=True,
            suggested_action="Check the output path and available disk space.",
        ) from exc


def write_logits_npz_atomic(output: Path, logits: np.ndarray) -> None:
    """Persist SAM logits compactly as float16 without exposing them as an image output."""

    pixels = np.asarray(logits, dtype=np.float16)
    if pixels.ndim != 2:
        raise KyvenError(ErrorCode.INFERENCE_FAILED, "SAM logits must be a two-dimensional array.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.stem}-", suffix=".npz", dir=output.parent, delete=False
        ) as stream:
            temporary_path = Path(stream.name)
        np.savez_compressed(temporary_path, logits=pixels)
        os.replace(temporary_path, output)
    except Exception as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise KyvenError(
            ErrorCode.OUTPUT_FAILED,
            f"Could not write SAM logits: {output}",
            technical_detail=str(exc),
        ) from exc


def confidence_trimap(logits: np.ndarray, width: float) -> np.ndarray:
    """Map logit confidence to exact black, mid-gray, and white pixels."""

    if width < 0:
        raise KyvenError(ErrorCode.INVALID_REQUEST, "Confidence Width must be zero or greater.")
    values = np.asarray(logits, dtype=np.float32)
    trimap = np.full(values.shape, 128, dtype=np.uint8)
    trimap[values <= -width] = 0
    trimap[values >= width] = 255
    return trimap


def read_logits_npz(path: Path) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return np.asarray(archive["logits"], dtype=np.float32)
    except (OSError, KeyError, ValueError) as exc:
        raise KyvenError(
            ErrorCode.INVALID_REQUEST,
            f"Could not read cached SAM logits: {path}",
            technical_detail=str(exc),
        ) from exc
