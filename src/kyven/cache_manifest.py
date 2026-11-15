"""Small atomic cache manifests used to validate and resume completed inference."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from kyven.atomic_io import replace_with_retry
from kyven.cache_layout import manifest_path, revision_path, work_root
from kyven.hashing import cached_file_sha256

CACHE_MANIFEST_VERSION = 1


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported cache-manifest value: {type(value).__name__}")


def _valid_file(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        if path.suffix.lower() == ".exr":
            from kyven.dlss.images import read_channels
            from kyven.errors import KyvenError

            try:
                return bool(read_channels(path))
            except KyvenError:
                return False
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
    if not path.exists():
        path = output.with_suffix(output.suffix + ".kyven.json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        payload = {}
    if not isinstance(payload, dict):
        return None
    if payload.get("cache_key") != cache_key:
        try:
            payload = json.loads(revision_path(output, cache_key).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None
    if not isinstance(payload.get("metadata", {}), dict):
        return None
    if payload.get("version") != CACHE_MANIFEST_VERSION or payload.get("cache_key") != cache_key:
        return None
    outputs = [item for item in required_outputs if item is not None]
    artifacts = payload.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return None
    for item in outputs:
        record = artifacts.get(str(item.resolve()))
        if record is None:
            if payload.get("complete_outputs") or not _valid_file(item):
                return None
            continue
        if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
            return None
        digest = record["sha256"]
        if _matches(item, digest):
            continue
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            return None
        stored = _artifact_path(output, digest, item.suffix)
        if not _matches(stored, digest) or not _valid_file(stored):
            return None
        _copy_atomic(stored, item)
    if not outputs:
        return None
    return payload


def write_manifest_atomic(
    output: Path,
    cache_key: str,
    metadata: dict[str, Any],
    required_outputs: Iterable[Path | None] | None = None,
    **result_fields: Any,
) -> None:
    path = manifest_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for item in (required_outputs if required_outputs is not None else (output,)):
        if item is None or not _valid_file(item):
            continue
        digest = cached_file_sha256(item)
        stored = _artifact_path(output, digest, item.suffix)
        if not _matches(stored, digest):
            _copy_atomic(item, stored)
        artifacts[str(item.resolve())] = {"sha256": digest}
    payload = {
        "version": CACHE_MANIFEST_VERSION,
        "cache_key": cache_key,
        "metadata": metadata,
        "artifacts": artifacts,
        "complete_outputs": required_outputs is not None,
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
        revision = revision_path(output, cache_key)
        _copy_atomic(temporary, revision)
        replace_with_retry(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _artifact_path(output: Path, digest: str, suffix: str) -> Path:
    # Content addressing shares identical artifacts across frames and node results.
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid artifact digest")
    return work_root(output) / "artifacts" / digest[:2] / (digest + suffix)


def _matches(path: Path, digest: str) -> bool:
    try:
        return path.is_file() and cached_file_sha256(path) == digest
    except OSError:
        return False


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".cache-", suffix=destination.suffix,
                                       dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        replace_with_retry(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
