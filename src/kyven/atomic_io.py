"""Atomic filesystem operations resilient to short-lived Windows host locks."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path


def replace_with_retry(temporary: Path, output: Path, attempts: int = 20) -> None:
    """Replace a cache file despite Windows readers that deny rename/delete sharing.

    Nuke may intentionally keep a Read handle open for the current frame. Windows then rejects
    ``os.replace`` even though writing the already-existing file is allowed. After waiting for a
    short-lived lock, update that file in place as a final compatibility fallback. The producer
    has already completed the temporary file, so readers never see model inference in progress.
    """

    for attempt in range(attempts):
        try:
            os.replace(temporary, output)
            return
        except PermissionError:
            if attempt + 1 < attempts:
                time.sleep(min(0.02 + attempt * 0.002, 0.1))
                continue
            if not output.is_file():
                raise
            # Windows Read nodes commonly deny delete/rename sharing but permit writes. This
            # non-atomic fallback is preferable to failing every repeat render of a cached frame.
            with temporary.open("rb") as source, output.open("r+b") as target:
                target.seek(0)
                shutil.copyfileobj(source, target, length=1024 * 1024)
                target.truncate()
                target.flush()
                os.fsync(target.fileno())
            temporary.unlink(missing_ok=True)
            return
