"""Color-management helpers for Nuke data images."""

from __future__ import annotations

from typing import Any


def set_data_io(node: Any) -> None:
    """Make a Nuke Read or Write bypass OCIO transforms for masks and mattes."""

    knobs = node.knobs()
    if "raw" in knobs:
        node["raw"].setValue(True)
        return

    # Older/custom Nuke builds may expose only the colorspace enumeration.
    if "colorspace" not in knobs:
        return
    colorspace = node["colorspace"]
    try:
        values = list(colorspace.values())
    except (AttributeError, TypeError):
        values = []
    for value in values:
        normalized = "".join(character for character in str(value).lower() if character.isalnum())
        if normalized == "data" or normalized == "raw" or normalized.endswith("raw"):
            colorspace.setValue(value)
            return
