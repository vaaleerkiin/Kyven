"""Shared lightweight media validation for host-provided image files."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from kyven.errors import ErrorCode, KyvenError


def validate_image_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise KyvenError(ErrorCode.INVALID_REQUEST, f"{label} image does not exist: {path}")
    try:
        with Image.open(path) as image:
            image.verify()
            if image.width <= 0 or image.height <= 0:
                raise ValueError("image dimensions must be positive")
    except (OSError, ValueError) as exc:
        raise KyvenError(
            ErrorCode.INVALID_REQUEST,
            f"{label} image is unreadable or corrupt: {path}",
            technical_detail=str(exc),
            suggested_action="Re-export the frame and retry the operation.",
        ) from exc
