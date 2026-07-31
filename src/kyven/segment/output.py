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
