"""Kyven host-independent package with dependency-free top-level imports."""

from __future__ import annotations

from typing import Any

__all__ = ["BoxPrompt", "PointLabel", "PointPrompt", "SegmentRequest"]
__version__ = "0.1.0.dev0"


def __getattr__(name: str) -> Any:
    """Lazily preserve the original public segmentation-schema exports."""
    if name in __all__:
        from kyven.segment.models import BoxPrompt, PointLabel, PointPrompt, SegmentRequest

        return {
            "BoxPrompt": BoxPrompt,
            "PointLabel": PointLabel,
            "PointPrompt": PointPrompt,
            "SegmentRequest": SegmentRequest,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
