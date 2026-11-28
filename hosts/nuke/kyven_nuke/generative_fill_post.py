"""Native cached-patch color/edge controls. No inference or worker calls here."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from kyven_nuke.color_management import set_data_io
from kyven_nuke.knob_tools import suspend_callbacks
from kyven_nuke.node import _add_section, _ensure_double_slider, _nuke, _place_knob_after

MAGE_RESOLUTIONS = (256, 512, 768, 1024, 1280, 1536, 1792, 2048)
FLUX_RESOLUTIONS = (256, 512, 768, 1024)
LIVE_CONTROLS = {"fill_color_mode", "fill_color_match", "fill_match_luminance",
                 "fill_match_chroma", "edge_softness", "fill_blend_margin", "fill_preview_blend_mask"}


@suspend_callbacks
def ensure_controls(node: Any, mage: bool) -> None:
    nuke = _nuke()
    if "working_size" in node.knobs():
        old_size = int(node["working_size"].value())
        if not mage and old_size > 1024:
            old_size = 1024
            node["working_size"].setValue(old_size)
        resolutions = list(MAGE_RESOLUTIONS if mage else FLUX_RESOLUTIONS)
        if old_size not in resolutions:
            resolutions.append(old_size)
            resolutions.sort()
        labels = [str(value) for value in resolutions]
        if "working_resolution" not in node.knobs():
            node.addKnob(nuke.Enumeration_Knob("working_resolution", "Processing Resolution", labels))
            _place_knob_after(node, "working_resolution", "model")
        node["working_resolution"].setValues(labels)
        node["working_resolution"].setValue(str(old_size))
        node["working_resolution"].setTooltip(
            "Maximum crop side used by the model, not final output size. Mage supports "
            "up to 2048; FLUX up to 1024. Larger sizes need more memory. Generate again to apply.")
        node["working_size"].setVisible(False)
    if "fill_color_mode" not in node.knobs():
        return
    old_mode = str(node["fill_color_mode"].value())
    modes = ["Lab", "Off"] if mage else ["Auto", "Context", "Background", "Off"]
    node["fill_color_mode"].setValues(modes)
    node["fill_color_mode"].setValue(old_mode if old_mode in modes else modes[0])
    node["fill_color_mode"].setTooltip(
        "Lab matches clean context using Reinhard Lab statistics. Off shows the raw "
        "generated patch. Mage changes update the cached patch immediately after one generation."
        if mage else "Choose worker color matching for FLUX; generate again to apply.")
    for name, label in (("fill_match_luminance", "Match Luminance"),
                        ("fill_match_chroma", "Match Chroma")):
        _ensure_double_slider(node, name, label, 0, 1, 1)
        node[name].setVisible(mage)
        node[name].setTooltip("0 preserves the generated value; 1 applies the fitted Lab "
                              "match. Updates cached Mage pixels in Nuke without generation.")
    _place_knob_after(node, "fill_match_luminance", "fill_color_match")
    _place_knob_after(node, "fill_match_chroma", "fill_match_luminance")
    if "fill_preview_blend_mask" not in node.knobs():
        preview = nuke.Boolean_Knob("fill_preview_blend_mask", "Preview Blend Mask")
        preview.setTooltip("Show the final live blend alpha, after Grow / Erode and ordinary Gaussian Blur.")
        node.addKnob(preview)
    node["fill_preview_blend_mask"].setVisible(mage)
    _ensure_double_slider(node, "fill_blend_margin", "Grow / Erode (px)", -128, 128, 0)
    node["fill_blend_margin"].setVisible(mage)
    node["fill_blend_margin"].setTooltip(
        "After generation only: positive grows the blend mask, negative shrinks it. "
        "Source pixels; never changes the mask sent to Mage. Limited to the generated crop.")
    node["edge_softness"].setLabel("Mask Blur (px)" if mage else "Edge Softness (px)")
    node["edge_softness"].setRange(0, 128 if mage else 32)
    node["edge_softness"].setFlag(nuke.STARTLINE)
    node["fill_preview_blend_mask"].setFlag(nuke.STARTLINE)
    if "fill_result_mask_section" not in node.knobs():
        _add_section(nuke, node, "fill_result_mask_section", "RESULT MASK / AFTER GENERATION")
    node["fill_result_mask_section"].setVisible(mage)
    # Relocate the whole block in either direction, preserving values and animation.
    block = ["fill_result_mask_section", "fill_blend_margin", "edge_softness",
             "fill_preview_blend_mask"]
    names = list(node.knobs())
    anchor = "output_help"
    start = min(names.index(name) for name in block + [anchor])
    tail = [node[name] for name in names[start:]]
    ordered = [knob for knob in tail if knob.name() not in block]
    index = next(i for i, knob in enumerate(ordered) if knob.name() == anchor) + 1
    ordered[index:index] = [node[name] for name in block]
    if tail != ordered:
        for knob in tail:
            node.removeKnob(knob)
        for knob in ordered:
            node.addKnob(knob)
    node["fill_color_match"].setTooltip(
        "Master amount of the cached Lab match. 0 keeps raw generated colors; 1 applies "
        "the selected luminance/chroma amounts. Updates without inference."
        if mage else "Amount of worker color matching. Generate again to apply FLUX changes.")
    node["fill_color_match"].setFlag(nuke.STARTLINE)
    for name in ("fill_color_match", "fill_match_luminance", "fill_match_chroma"):
        node[name].setEnabled(str(node["fill_color_mode"].value()) != "Off")
    if "fill_color_radius" in node.knobs():
        node["fill_color_radius"].setVisible(not mage)
    if "fill_native_ready" not in node.knobs():
        ready = nuke.Boolean_Knob("fill_native_ready", "Native Patch Available")
        ready.setVisible(False)
        node.addKnob(ready)
    node["fill_native_ready"].setExpression(
        '[python {kyven_nuke.generative_fill_post.native_ready(nuke.thisNode())}] + 0*frame')
    if mage:
        node["edge_softness"].setTooltip(
            "Ordinary Gaussian Blur after Grow / Erode, in Source pixels. "
            "0 disables blur. Updates cached pixels immediately; never changes the model mask.")
        node["create_result_read"].setLabel("Create Base Result Read")
        node["create_result_read"].setTooltip(
            "Read the cached base result. Use a normal Write on this Group to save the live "
            "Color Match and edge adjustments, Source alpha and auxiliary channels.")
    else:
        node["create_result_read"].setLabel("Create Result Read")


def native_ready(node: Any) -> int:
    """One file check per knob evaluation, never per pixel or a model request."""
    if not str(node["model"].value()).startswith("Mage-Flow Edit Turbo"):
        return 0
    read = node.node("KyvenFillLabRead")
    if read is None or not str(read["file"].value()):
        return 0
    nuke = _nuke()
    path = nuke.filename(read, nuke.REPLACE)
    raw = node.node("KyvenPatchRead")
    raw_path = nuke.filename(raw, nuke.REPLACE) if raw is not None else None
    if not path or not raw_path or not os.path.isfile(path):
        return 0
    try:
        # Native patches carry crop coverage in RGBA. Legacy/FLUX RGB patches
        # must not accidentally activate an older neighboring Lab sidecar.
        with open(raw_path, "rb") as stream:
            header = stream.read(26)
        return int(len(header) == 26 and header[:8] == b"\x89PNG\r\n\x1a\n" and header[25] == 6)
    except OSError:
        return 0


def ensure_graph(node: Any) -> None:
    """Blend full-strength/raw cached patches in Lab, then feather native alpha."""
    nuke = _nuke()
    # Called inside the Group by generative_fill_node._ensure_fill_graph.
    raw = nuke.toNode("KyvenPatchRead")
    if raw is None:
        raw = nuke.nodes.Read(name="KyvenPatchRead")
        raw["on_error"].setValue("black")
        set_data_io(raw)

    def get(kind: str, name: str) -> Any:
        current = nuke.toNode(name)
        if current is not None and current.Class() != kind:
            nuke.delete(current)
            current = None
        return current or getattr(nuke.nodes, kind)(name=name)

    matched = get("Read", "KyvenFillLabRead")
    raw_path = str(raw["file"].value())
    matched_path = Path(raw_path).with_name(Path(raw_path).stem + ".lab.png").as_posix() if raw_path else ""
    if str(matched["file"].value()) != matched_path:
        matched["file"].setValue(matched_path)
    for name in ("first", "last", "origfirst", "origlast"):
        if name in raw.knobs() and name in matched.knobs():
            matched[name].setValue(raw[name].value())
    matched["on_error"].setValue("black")
    set_data_io(matched)
    if raw_path:
        matched["reload"].execute()
    for name, image in (("KyvenFillRawLab", raw), ("KyvenFillMatchedLab", matched)):
        convert = get("Colorspace", name)
        convert.setInput(0, image)
        convert["colorspace_in"].setValue("sRGB")
        convert["colorspace_out"].setValue("CIE-Lab")
    mix = get("MergeExpression", "KyvenFillLabMix")
    mix.setInput(0, nuke.toNode("KyvenFillRawLab"))
    mix.setInput(1, nuke.toNode("KyvenFillMatchedLab"))
    for i, channel in enumerate(("r", "g", "b")):
        amount = "fill_match_luminance" if i == 0 else "fill_match_chroma"
        mix[f"expr{i}"].setValue(
            f"B{channel}+(A{channel}-B{channel})*parent.fill_color_match*parent.{amount}")
    mix["expr3"].setValue("Ba")
    restore = get("Colorspace", "KyvenFillLabToSRGB")
    restore.setInput(0, mix)
    restore["colorspace_in"].setValue("CIE-Lab")
    restore["colorspace_out"].setValue("sRGB")
    color = get("Switch", "KyvenFillLiveColor")
    color.setInput(0, raw)
    color.setInput(1, restore)
    color["which"].setExpression(
        "parent.fill_native_ready && parent.fill_color_mode==0 && parent.fill_color_match>0")
    nuke.toNode("KyvenPatchFromSRGB").setInput(0, color)

    # The artist alpha is processed only here, after the cached image exists.
    erode = get("FilterErode", "KyvenFillLiveErode")
    erode.setInput(0, nuke.toNode("KyvenFillPermission"))
    erode["channels"].setValue("rgba")
    erode["size"].setExpression("-parent.fill_blend_margin")
    blur = get("Blur", "KyvenFillLiveBlur")
    blur.setInput(0, erode)
    blur["channels"].setValue("rgba")
    blur["filter"].setValue("gaussian")
    blur["size"].setExpression("parent.edge_softness")
    feather = get("MergeExpression", "KyvenFillLiveFeather")
    feather.setInput(0, blur)
    feather.setInput(1, raw)
    for i, channel in enumerate(("r", "g", "b", "a")):
        feather[f"expr{i}"].setValue(
            f"clamp(B{channel},0,1)*(parent.fill_native_ready ? clamp(Aa,0,1) : 1)")
    alpha = get("Switch", "KyvenFillLiveAlpha")
    alpha.setInput(0, nuke.toNode("KyvenFillClippedMask"))
    alpha.setInput(1, feather)
    alpha["which"].setExpression("parent.fill_native_ready")
    nuke.toNode("KyvenPatchMaskAlpha").setInput(1, alpha)
    public = get("Switch", "KyvenFillLivePublicMask")
    public.setInput(0, nuke.toNode("KyvenFillPublicMask"))
    public.setInput(1, feather)
    public["which"].setExpression("parent.fill_native_ready")
    nuke.toNode("KyvenGeneratedPatch").setInput(1, public)
    nuke.toNode("KyvenResultSourceAlpha").setInput(1, public)
    preview = get("Switch", "KyvenFillResultMaskPreview")
    preview.setInput(0, blur)
    preview.setInput(1, feather)
    preview["which"].setExpression("parent.fill_native_ready")
    output = nuke.toNode("KyvenOutputSwitch")
    output.setInput(7, preview)
    output["which"].setExpression(
        "parent.fill_preview_blend_mask ? 7 : (parent.preview_model_mask ? 6 : parent.output_mode)")
