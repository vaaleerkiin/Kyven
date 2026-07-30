"""Local runtime paths used by the Nuke adapter."""

from __future__ import annotations

import os
from pathlib import Path


def root() -> Path:
    return Path(os.environ.get("KYVEN_ROOT", Path(__file__).resolve().parents[3])).resolve()


def executable() -> Path:
    override = os.environ.get("KYVEN_EXECUTABLE")
    if override:
        return Path(override).resolve()
    return root() / ".venv" / "Scripts" / "kyven.exe"


def python_executable() -> Path:
    override = os.environ.get("KYVEN_PYTHON_EXECUTABLE")
    if override:
        return Path(override).resolve()
    return root() / ".venv" / "Scripts" / "python.exe"


def models_dir() -> Path:
    return Path(os.environ.get("KYVEN_MODELS_DIR", root() / "models")).resolve()


def runtime_dir() -> Path:
    return Path(os.environ.get("KYVEN_RUNTIME_DIR", root() / ".runtime")).resolve()


def token_file() -> Path:
    return runtime_dir() / "server.token"


def cache_dir() -> Path:
    return runtime_dir() / "nuke_cache"
