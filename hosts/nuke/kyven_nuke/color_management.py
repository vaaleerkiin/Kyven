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
    """Choose a texture/input sRGB interchange space for AI image files.

    LaMa consumes ordinary display-encoded 8-bit sRGB pixels. Under ACES, a
    texture/input space is the reversible interchange transform; an Output or
    display sRGB transform is not suitable for a model round trip.
    """

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
        if normalized == "utilitysrgbtexture":
            ranked.append((0, str(value)))
        elif "srgb" in normalized and "texture" in normalized and "linear" not in normalized:
            ranked.append((1, str(value)))
        elif normalized == "srgb":
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


def configure_ai_color_io(source: Any, targets: tuple[Any, ...], *, linear: bool) -> None:
    """Configure a symmetric Nuke -> model -> Nuke color round trip."""

    if linear:
        set_data_io(source)
        for target in targets:
            if target is not None:
                set_data_io(target)
        return
    set_interchange_color_io(source)
    for target in targets:
        if target is not None:
            match_color_io(source, target)
