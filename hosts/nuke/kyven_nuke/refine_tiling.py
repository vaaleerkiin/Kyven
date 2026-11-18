"""Explicit Refine tiling controls and an always-visible effective-settings summary."""

from __future__ import annotations

from typing import Any

from kyven_nuke.knob_tools import suspend_callbacks

MODES = ("Tiles", "Full Crop")
LEGACY_PROFILE_TILES = {"low_memory": 512, "balanced": 1024, "quality": 0}


def preview_info(summary: str) -> str:
    """Reserve the same three rows while the asynchronous preview is loading."""

    return (f'<nobr><b>Tile Preview</b>: {summary}</nobr><br>'
            '<nobr><font color="#37eb69">Green: RUN</font> | Gray: SKIP | '
            '<font color="#ffc323">Yellow: unknown</font> | Cyan: crop boundary</nobr><br>'
            '<nobr>View RGB for colors; Alpha also shows the tile coverage.</nobr>')


def effective_tile_size(node: Any, frame: int | None = None) -> int:
    """Use the explicit size, with compatibility for nodes awaiting an upgrade."""

    from kyven_nuke.node import _knob_value_at

    knobs = node.knobs() if hasattr(node, "knobs") else node.values
    custom = int(_knob_value_at(node["tile_size"], frame) if frame is not None
                 else node["tile_size"].value())
    mode = str(node["tile_mode"].value()) if "tile_mode" in knobs else (
        "Tiles" if custom else "Auto (Profile)")
    if mode == "Full Crop":
        return 0
    if mode == "Auto (Profile)":
        return LEGACY_PROFILE_TILES[str(node["profile"].value())]
    if custom < 128:
        raise ValueError("Tile Size must be at least 128 px.")
    return custom


@suspend_callbacks
def update_controls(node: Any) -> None:
    """Show the actual model tile size and explain the work skipped by trimap."""

    if not {"tile_mode", "tiling_help", "skip_known_tiles"}.issubset(node.knobs()):
        return
    mode = str(node["tile_mode"].value())
    tiled = mode == "Tiles"
    node["tile_size"].setVisible(True)
    node["tile_size"].setEnabled(tiled)
    node["tile_overlap"].setVisible(True)
    node["tile_overlap"].setEnabled(tiled)
    try:
        tile = effective_tile_size(node)
    except ValueError as exc:
        node["tiling_help"].setValue(f'<font color="#ffb070">{exc}</font>')
        return
    node["tile_overlap"].setRange(0, min(256, (tile-1)//2) if tile else 256)
    overlap = int(node["tile_overlap"].value())
    if tile and (overlap < 0 or overlap * 2 >= tile):
        text = f'<font color="#ffb070">Overlap must be 0 to {(tile-1)//2} px.</font>'
    elif tile:
        text = (f"<b>{mode}</b>: {tile} x {tile} px maximum | "
                f"Overlap {overlap} px | Step {tile-overlap} px<br>"
                "Source detail stays 1:1. Smaller squares use less context and memory.")
    else:
        text = (f"<b>{mode}</b>: Full Crop, one model pass over the processing crop; "
                "highest peak memory.")
    if bool(node["skip_known_tiles"].value()):
        text += "<br>Only tiles containing gray trimap run. Known black/white tiles use alpha 0/1."
    else:
        text += "<br>Every tile runs, including known black/white areas."
    node["tiling_help"].setValue(text)


@suspend_callbacks
def ensure_controls(node: Any) -> None:
    """Keep explicit Tiles / Full Crop modes and preserve legacy effective settings."""

    from kyven_nuke.node import (
        _add_knob,
        _add_section,
        _ensure_double_slider,
        _nuke,
        _place_knob_after,
    )

    nuke = _nuke()
    if "tiling_section" not in node.knobs():
        _add_section(nuke, node, "tiling_section", "TILING / MEMORY")
    previous = int(node["tile_size"].value())
    existing_mode = node.knobs().get("tile_mode")
    previous_mode = str(existing_mode.value()) if existing_mode is not None else None
    selected_mode = "Full Crop" if previous_mode == "Full Crop" else "Tiles"
    if previous_mode == "Auto (Profile)" or (previous_mode is None and previous == 0):
        previous = LEGACY_PROFILE_TILES[str(node["profile"].value())]
        selected_mode = "Full Crop" if previous == 0 else "Tiles"
        node["tile_size"].clearAnimated()
        node["tile_size"].setValue(previous or 1024)
    elif previous == 0 and previous_mode == "Full Crop":
        node["tile_size"].setValue(1024)
    if existing_mode is None:
        mode = nuke.Enumeration_Knob("tile_mode", "Tiling Mode", list(MODES))
        mode.setValue(selected_mode)
        _add_knob(nuke, node, mode)
    elif tuple(existing_mode.values()) != MODES:
        existing_mode.clearAnimated()
        existing_mode.setValues(list(MODES))
        existing_mode.setValue(selected_mode)
    if "skip_known_tiles" not in node.knobs():
        skip = nuke.Boolean_Knob("skip_known_tiles", "Skip Black / White Tiles")
        skip.setValue(True)
        _add_knob(nuke, node, skip)
    if "tiling_help" not in node.knobs():
        _add_knob(nuke, node, nuke.Text_Knob("tiling_help", "", ""))
    output = str(node['output_mode'].value()) if 'output_mode' in node.knobs() else ''
    if 'show_tiles' not in node.knobs():
        show = nuke.Boolean_Knob('show_tiles', 'Show Tiles')
        show.setValue(output == 'Tile Preview')
        _add_knob(nuke, node, show)
    if 'tile_preview_previous_output' not in node.knobs():
        previous_output = nuke.String_Knob('tile_preview_previous_output', 'Previous Output')
        previous_output.setValue(output if output and output != 'Tile Preview'
                                 else 'Source + Refined Alpha')
        previous_output.setVisible(False)
        node.addKnob(previous_output)
    if "preview_tiles" not in node.knobs():
        button = nuke.PyScript_Knob("preview_tiles", "Preview Tiles")
        button.setCommand("__import__('kyven_nuke.refine_tile_preview', "
                          "fromlist=['show_preview']).show_preview()")
        _add_knob(nuke, node, button)
    if "tile_preview_info" not in node.knobs():
        _add_knob(nuke, node, nuke.Text_Knob("tile_preview_info", "",
            preview_info("Enable Show Tiles to inspect the current plan.")))
    if "tile_preview_labels" not in node.knobs():
        _add_knob(nuke, node, nuke.Boolean_Knob("tile_preview_labels", "Show Tile Labels"))
    if "tile_preview_bounds" not in node.knobs():
        bounds = nuke.BBox_Knob("tile_preview_bounds", "Trimap Bounds")
        bounds.setVisible(False)
        node.addKnob(bounds)
    if "tile_preview_bounds_valid" not in node.knobs():
        valid = nuke.Boolean_Knob("tile_preview_bounds_valid", "Trimap Bounds Valid")
        valid.setVisible(False)
        node.addKnob(valid)
    if "tile_preview_size" not in node.knobs():
        size = nuke.Int_Knob("tile_preview_size", "Effective Tile Size")
        size.setVisible(False)
        node.addKnob(size)
    node["tile_preview_size"].setExpression(
        "tile_mode==1 ? 0 : floor(tile_size)")
    _ensure_double_slider(node, "tile_size", "Tile Size (px)", 128, 2048)
    _ensure_double_slider(node, "tile_overlap", "Tile Overlap (px)", 0, 256)
    anchor = "open_model_manager"
    for name in ("tiling_section", "tile_mode", "tile_size", "tile_overlap",
                 "skip_known_tiles", "tiling_help", "show_tiles", "preview_tiles", "tile_preview_labels",
                 "tile_preview_info"):
        _place_knob_after(node, name, anchor)
        anchor = name
    update_controls(node)
