"""Scene-linear float transport; OpenEXR is loaded only in the backend."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from kyven.atomic_io import replace_with_retry
from kyven.errors import ErrorCode, KyvenError


def _openexr():
    try:
        import OpenEXR
        return OpenEXR
    except ImportError as exc:
        raise KyvenError(
            ErrorCode.DEPENDENCY_MISSING, "DLSS EXR support is not installed.",
            suggested_action='Install Kyven with the dlss extra: pip install ".[dlss]".',
        ) from exc


def read_channels(path: Path) -> dict[str, np.ndarray]:
    try:
        if path.suffix.lower() == ".exr":
            with _openexr().File(str(path), separate_channels=True) as image:
                if len(image.parts) != 1:
                    raise ValueError("Export a single-part, non-deep EXR")
                header = image.header()
                data_min, data_max = header["dataWindow"]
                display_min, display_max = header["displayWindow"]
                if not (np.array_equal(data_min, display_min)
                        and np.array_equal(data_max, display_max)):
                    raise ValueError("Export full-frame EXRs with matching data/display windows")
                channels = {k: np.asarray(v.pixels, dtype=np.float32).copy()
                            for k, v in image.channels().items()}
        else:
            with Image.open(path) as image:
                rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
            channels = dict(zip(("R", "G", "B", "A"), np.moveaxis(rgba, -1, 0)))
        if not channels or any(v.ndim != 2 or not np.isfinite(v).all()
                               for v in channels.values()):
            raise ValueError("Channels must be finite, two-dimensional float arrays")
        return channels
    except (OSError, ValueError, RuntimeError) as exc:
        raise KyvenError(
            ErrorCode.INVALID_REQUEST, f"Could not read DLSS input: {path}",
            technical_detail=str(exc), suggested_action="Re-export full-frame linear float EXRs.",
        ) from exc


def channel(channels: dict[str, np.ndarray], name: str) -> np.ndarray:
    aliases = {"red": "R", "green": "G", "blue": "B", "alpha": "A",
               "rgba.red": "R", "rgba.green": "G", "rgba.blue": "B", "rgba.alpha": "A"}
    key = name if name in channels else aliases.get(name, name)
    if key not in channels:
        raise KyvenError(ErrorCode.INVALID_REQUEST, f"DLSS input channel is missing: {name}")
    return channels[key]


def resize(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear float resize without clipping HDR values or negative vectors."""
    if values.shape[:2] == (size[1], size[0]):
        return values.copy()
    if values.ndim == 2:
        return np.asarray(Image.fromarray(values.astype(np.float32)).resize(
            size, Image.Resampling.BILINEAR), dtype=np.float32)
    return np.stack([resize(values[..., c], size) for c in range(values.shape[2])], axis=-1)


def write_exr(path: Path, rgba: np.ndarray) -> None:
    """Publish lossless float EXR atomically, retaining the original alpha precision."""
    exr = _openexr()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".dlss-", suffix=".exr", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with exr.File({"compression": exr.ZIP_COMPRESSION},
                      {"RGBA": np.ascontiguousarray(rgba, dtype=np.float32)}) as image:
            image.write(str(temporary))
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
