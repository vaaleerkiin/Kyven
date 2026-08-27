"""Small atomic cache manifests used to validate and resume completed inference."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from kyven.atomic_io import replace_with_retry

CACHE_MANIFEST_VERSION = 1


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported cache-manifest value: {type(value).__name__}")


def manifest_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".kyven.json")


def _valid_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                values = np.asarray(archive["logits"])
                return values.ndim == 2 and values.size > 0 and bool(np.isfinite(values).all())
        with Image.open(path) as image:
            image.verify()
            return image.width > 0 and image.height > 0
    except (OSError, KeyError, ValueError):
        return False


def read_valid_manifest(
    output: Path,
    cache_key: str,
    required_outputs: Iterable[Path | None],
) -> dict[str, Any] | None:
    path = manifest_path(output)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if payload.get("version") != CACHE_MANIFEST_VERSION or payload.get("cache_key") != cache_key:
        return None
    outputs = [item for item in required_outputs if item is not None]
    if not outputs or not all(_valid_file(item) for item in outputs):
        return None
    return payload


def write_manifest_atomic(
    output: Path,
    cache_key: str,
    metadata: dict[str, Any],
    **result_fields: Any,
) -> None:
    path = manifest_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CACHE_MANIFEST_VERSION,
        "cache_key": cache_key,
        "metadata": metadata,
        **result_fields,
    }
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{path.stem}-", suffix=".json", dir=path.parent
        )
        os.close(descriptor)
        temporary = Path(name)
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default),
            encoding="utf-8",
        )
        replace_with_retry(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
