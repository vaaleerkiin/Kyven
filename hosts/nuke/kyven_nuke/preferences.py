"""Persistent user defaults for newly created Kyven nodes."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from kyven_nuke import config

_FALLBACKS = {
    "segment": "sam2.1-small",
    "refine": "vitmatte-small-composition-1k",
    "inpaint": "big-lama-native",
    "generative_fill": "flux2-klein-4b-q4-lanpaint",
}


def _path() -> Path:
    return config.runtime_dir() / "preferences.json"


def load_defaults() -> dict[str, str]:
    try:
        payload = json.loads(_path().read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return dict(_FALLBACKS)
    defaults = dict(_FALLBACKS)
    stored = payload.get("default_models", {}) if isinstance(payload, dict) else {}
    if isinstance(stored, dict):
        for task in defaults:
            value = stored.get(task)
            if isinstance(value, str) and value:
                defaults[task] = value
    return defaults


def default_model_id(task: str) -> str:
    return load_defaults().get(task, _FALLBACKS.get(task, ""))


def default_model_index(task: str, model_ids: tuple[str, ...]) -> int:
    try:
        return model_ids.index(default_model_id(task))
    except ValueError:
        fallback = _FALLBACKS.get(task, "")
        return model_ids.index(fallback) if fallback in model_ids else 0


def set_default_model(task: str, model_id: str) -> None:
    if task not in _FALLBACKS or not model_id:
        raise ValueError("Unknown Kyven model task or empty model id.")
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    defaults = load_defaults()
    defaults[task] = model_id
    payload = json.dumps({"version": 1, "default_models": defaults}, indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".preferences-", suffix=".json", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            Path(temporary_name).unlink(missing_ok=True)
        except OSError:
            pass
