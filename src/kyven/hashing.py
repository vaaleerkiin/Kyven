"""Shared bounded-memory hashing helpers."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

_cache_lock = threading.RLock()
_file_cache: dict[tuple[str, int, int, int], str] = {}


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_file_sha256(path: Path) -> str:
    """Reuse immutable checkpoint hashes while invalidating on size or nanosecond mtime changes."""

    resolved = path.resolve()
    stat = resolved.stat()
    key = (str(resolved), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    with _cache_lock:
        cached = _file_cache.get(key)
    if cached is not None:
        return cached
    digest = file_sha256(resolved)
    after = resolved.stat()
    if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != key[1:]:
        raise OSError(f"File changed while hashing: {resolved}")
    with _cache_lock:
        stale = [item for item in _file_cache if item[0] == str(resolved) and item != key]
        for item in stale:
            _file_cache.pop(item, None)
        if len(_file_cache) >= 4096:
            _file_cache.pop(next(iter(_file_cache)))
        _file_cache[key] = digest
    return digest
