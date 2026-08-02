"""Color-management helpers for Nuke data images."""

from __future__ import annotations

from typing import Any


def set_data_io(node: Any) -> None:
    """Make a Nuke Read or Write bypass OCIO transforms for masks and mattes."""

    knobs = node.knobs()
    if "raw" in knobs:
        node["raw"].setValue(True)

    # Also select the explicit data space. Some Nuke/OCIO combinations still
    # evaluate the colorspace knob even when the raw checkbox is enabled.
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


def set_interchange_color_io(node: Any) -> None:
    """Choose an explicit sRGB interchange space for AI image files."""

    knobs = node.knobs()
    if "raw" in knobs:
        node["raw"].setValue(False)
    if "colorspace" not in knobs:
        return
    colorspace = node["colorspace"]
    try:
        values = list(colorspace.values())
    except (AttributeError, TypeError):
        values = []
    ranked: list[tuple[int, str]] = []
    for value in values:
        normalized = "".join(character for character in str(value).lower() if character.isalnum())
        if normalized == "srgb":
            ranked.append((0, str(value)))
        elif normalized == "utilitysrgbtexture":
            ranked.append((1, str(value)))
        elif normalized.endswith("srgbtexture"):
            ranked.append((2, str(value)))
    if ranked:
        colorspace.setValue(min(ranked)[1])


def match_color_io(source: Any, target: Any) -> None:
    """Make a cached color Read invert the exact colorspace used by its Write."""

    source_knobs = source.knobs()
    target_knobs = target.knobs()
    if "raw" in target_knobs:
        target["raw"].setValue(False)
    if "colorspace" not in source_knobs or "colorspace" not in target_knobs:
        return
    source_value = source["colorspace"].value
    value = source_value() if callable(source_value) else source_value
    try:
        target["colorspace"].setValue(value)
    except (TypeError, ValueError):
        return
