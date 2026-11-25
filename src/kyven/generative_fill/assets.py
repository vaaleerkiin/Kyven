"""Pinned, checksum-verified model bundle. No inference dependencies."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from kyven.errors import ErrorCode, KyvenError
from kyven.hashing import cached_file_sha256


def manifest_path() -> Path:
    return Path(__file__).parents[1] / "resources/generative_fill/models.json"


def manifest(model_id: str = "") -> dict:
    path = manifest_path()
    if model_id == "moss-sft-int8-edit-lanpaint":
        path = path.with_name("moss_models.json")
    elif model_id == "flux2-klein-base-4b-q4-lanpaint":
        path = path.with_name("klein_base_models.json")
    elif model_id and model_id != "flux2-klein-4b-q4-lanpaint":
        raise ValueError(f"Unknown Generative Fill bundle: {model_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_checksum(model_id: str = "") -> str:
    data = manifest(model_id) if model_id else manifest()
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def bundle_ready(root: Path, *, verify: bool = False, model_id: str = "") -> bool:
    data = manifest(model_id) if model_id else manifest()
    for item in data["files"]:
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != item["size_bytes"]:
            return False
        if verify and cached_file_sha256(path) != item["sha256"]:
            return False
    return True


def download_bundle(root: Path, progress: Callable | None = None,
                    cancelled: Callable | None = None, *, model_id: str = "") -> Path:
    """Resume individual files and publish each only after size/SHA verification."""
    files = (manifest(model_id) if model_id else manifest())["files"]
    total = sum(item["size_bytes"] for item in files)
    received = {item["path"]: 0 for item in files}
    lock = threading.Lock()
    stopped = threading.Event()

    def report(item, count):
        with lock:
            received[item["path"]] = count
            if progress:
                progress(sum(received.values()), total)

    def check_cancelled():
        if stopped.is_set() or (cancelled and cancelled()):
            raise KyvenError(ErrorCode.CANCELLED, "Generative Fill download cancelled.")

    def download(item):
        check_cancelled()
        path = root / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if (path.is_file() and path.stat().st_size == item["size_bytes"]
                and cached_file_sha256(path) == item["sha256"]):
            report(item, item["size_bytes"])
            return
        # Reuse only verified immutable files shared with another bundle.
        for sibling in root.parent.iterdir():
            source = sibling / item["path"]
            if sibling == root or not source.is_file() or source.stat().st_size != item["size_bytes"]:
                continue
            if cached_file_sha256(source) != item["sha256"]:
                continue
            try:
                os.link(source, path)
            except OSError:
                continue
            report(item, item["size_bytes"])
            return
        partial = path.with_name(path.name + ".download")
        offset = partial.stat().st_size if partial.is_file() else 0
        if offset == item["size_bytes"] and cached_file_sha256(partial) == item["sha256"]:
            os.replace(partial, path)
            report(item, item["size_bytes"])
            return
        if offset >= item["size_bytes"]:
            partial.unlink()
            offset = 0
        # Fetch a fresh signed redirect for each attempt; cached expired hub
        # redirects can otherwise return 403 even though the pinned file exists.
        for attempt in range(3):
            check_cancelled()
            offset = partial.stat().st_size if partial.is_file() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            separator = "&" if "?" in item["url"] else "?"
            url = item["url"] + separator + f"download=true&kyven_attempt={time.time_ns()}-{attempt}"
            request = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    if response.status != 206:
                        offset = 0
                    elif not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        partial.unlink(missing_ok=True)
                        raise OSError("Download server returned a mismatched byte range.")
                    with partial.open("ab" if offset else "wb") as stream:
                        downloaded = offset
                        for chunk in iter(lambda: response.read(1024 * 1024), b""):
                            check_cancelled()
                            stream.write(chunk)
                            downloaded += len(chunk)
                            report(item, downloaded)
                if partial.stat().st_size != item["size_bytes"] or cached_file_sha256(partial) != item["sha256"]:
                    partial.unlink(missing_ok=True)
                    raise OSError(f"Generative Fill checksum verification failed: {item['path']}")
                break
            except (urllib.error.URLError, OSError):
                if attempt == 2:
                    raise
                check_cancelled()
                time.sleep(0.25)
        os.replace(partial, path)
        report(item, item["size_bytes"])

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="kyven-fill-download") as executor:
        futures = [executor.submit(download, item) for item in files]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            stopped.set()
            for future in futures:
                future.cancel()
            raise
    return root
