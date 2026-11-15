"""Cache layout and publication shared with lightweight host adapters."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from kyven.atomic_io import replace_with_retry
from kyven.hashing import cached_file_sha256


def work_root(output: Path) -> Path:
    for parent in output.parents:
        if parent.name == "_kyven_work":
            return parent
    return output.parent / "_kyven_work"


def manifest_path(output: Path) -> Path:
    return work_root(output) / "manifests" / f"{output.name}.kyven.json"



def revision_path(output: Path, cache_key: str) -> Path:
    identity = hashlib.sha256((str(output.resolve()) + "\0" + cache_key).encode()).hexdigest()
    return work_root(output) / "manifests" / "revisions" / (identity + ".json")


def copy_if_changed(source: Path, target: Path) -> bool:
    """Publish bytes only when different; preserve durable staging results."""
    if target.is_file() and cached_file_sha256(source) == cached_file_sha256(target):
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".publish-", suffix=target.suffix, dir=target.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        replace_with_retry(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def publish_cached_output(source: Path, target: Path) -> None:
    """Copy an accepted result and remap its manifest without moving private data."""
    origin = manifest_path(source)
    if not origin.is_file():
        origin = source.with_suffix(source.suffix + ".kyven.json")
    if not origin.is_file():
        copy_if_changed(source, target)
        return
    payload = json.loads(origin.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Invalid cache manifest")
    artifacts = payload.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise TypeError("Invalid cache artifact index")
    record = artifacts.pop(str(source.resolve()), None)
    if record is not None:
        if not isinstance(record, dict) or record.get("sha256") != cached_file_sha256(source):
            raise ValueError("Result does not match its cache manifest")
        artifacts[str(target.resolve())] = record
        # The normal DLSS paths share a work root; support external staging too.
        digest = record["sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid artifact digest")
        artifact = work_root(target) / "artifacts" / digest[:2] / (digest + target.suffix)
        copy_if_changed(source, artifact)
    copy_if_changed(source, target)
    destination = manifest_path(target)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if destination.is_file() and destination.read_bytes() == encoded:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".manifest-", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_bytes(encoded)
        replace_with_retry(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
