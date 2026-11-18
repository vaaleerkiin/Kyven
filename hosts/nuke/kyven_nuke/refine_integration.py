"""Optional Nuke-only blending of a cached local matte into incoming alpha."""

from __future__ import annotations

from typing import Any

HELP = (
    "Optional output blend into Mask alpha; no model processing.<br>"
    "Manual ROI only. Feather softens inward; use a Write to save the integrated result."
)


def update_controls(node: Any) -> None:
    """Keep integration independent of model and trimap controls."""

    if not {"roi_blend", "roi_feather"}.issubset(node.knobs()):
        return
    manual = str(node["crop_mode"].value()) == "manual"
    for name in ("integration_section", "roi_blend", "roi_feather", "integration_help"):
        if name in node.knobs():
            node[name].setVisible(manual)
    node["roi_blend"].setEnabled(manual)
    node["roi_feather"].setEnabled(manual and bool(node["roi_blend"].value()))


def ensure_controls(node: Any) -> None:
    """Add opt-in controls without altering existing values or model settings."""

    from kyven_nuke.node import (
        _add_knob,
        _add_section,
        _ensure_double_slider,
        _nuke,
        _place_knob_after,
    )

    nuke = _nuke()
    if "integration_section" not in node.knobs():
        _add_section(nuke, node, "integration_section", "ALPHA INTEGRATION / OPTIONAL")
    if "roi_blend" not in node.knobs():
        _add_knob(nuke, node, nuke.Boolean_Knob("roi_blend", "Blend ROI into Input Alpha"))
    _ensure_double_slider(node, "roi_feather", "ROI Feather (px)", 0, 512, 64)
    if "integration_help" not in node.knobs():
        _add_knob(nuke, node, nuke.Text_Knob("integration_help", "", HELP))
    else:
        node["integration_help"].setValue(HELP)
    anchor = "output_help"
    for name in ("integration_section", "roi_blend", "roi_feather", "integration_help"):
        _place_knob_after(node, name, anchor)
        anchor = name
    update_controls(node)


def build_graph(nuke: Any, refined: Any, original_alpha: Any) -> Any:
    """Build inside the Refine Group; return the optional integrated matte branch."""

    ramp = nuke.toNode("KyvenROIIntegrationRamp")
    if ramp is None:
        ramp = nuke.nodes.Expression(name="KyvenROIIntegrationRamp")
    # The cached full-format matte supplies the canvas, including recovered hairs
    # outside the incoming mask's bounding box.
    ramp.setInput(0, refined)
    bounds = (
        ("rx0", "max(0,floor(min(parent.processing_roi.x,parent.processing_roi.r)))"),
        ("ry0", "max(0,floor(min(parent.processing_roi.y,parent.processing_roi.t)))"),
        ("rx1", "min(width,ceil(max(parent.processing_roi.x,parent.processing_roi.r)))"),
        ("ry1", "min(height,ceil(max(parent.processing_roi.y,parent.processing_roi.t)))"),
    )
    for index, (name, expression) in enumerate(bounds):
        ramp[f"temp_name{index}"].setValue(name)
        ramp[f"temp_expr{index}"].setValue(expression)
    x_width = (
        "min(max(1,parent.roi_feather),"
        "max(1,floor((rx1-rx0-1)/max(1,(rx0>0)+(rx1<width)))))"
    )
    y_width = (
        "min(max(1,parent.roi_feather),"
        "max(1,floor((ry1-ry0-1)/max(1,(ry0>0)+(ry1<height)))))"
    )
    distance = (
        f"min(min(rx0<=0 ? 1 : (x-rx0)/({x_width}),"
        f"rx1>=width ? 1 : (rx1-1-x)/({x_width})),"
        f"min(ry0<=0 ? 1 : (y-ry0)/({y_width}),"
        f"ry1>=height ? 1 : (ry1-1-y)/({y_width})))"
    )
    value = (
        "rx0>=rx1 || ry0>=ry1 ? 1 : "
        "(x>=rx0 && x<rx1 && y>=ry0 && y<ry1 ? "
        f"(parent.roi_feather<=0 ? 1 : max(0,min(1,{distance}))) : 0)"
    )
    for index in range(4):
        ramp[f"expr{index}"].setValue(value)

    weight = nuke.toNode("KyvenROIIntegrationMask")
    if weight is None:
        weight = nuke.nodes.Expression(name="KyvenROIIntegrationMask")
    weight.setInput(0, ramp)
    for index, channel in enumerate(("r", "g", "b", "a")):
        weight[f"expr{index}"].setValue(f"{channel}*{channel}*(3-2*{channel})")

    blend = nuke.toNode("KyvenROIIntegrationBlend")
    if blend is None:
        blend = nuke.nodes.Keymix(name="KyvenROIIntegrationBlend")
    blend.setInput(0, original_alpha)
    blend.setInput(1, refined)
    blend.setInput(2, weight)
    blend["channels"].setValue("rgba")
    blend["maskChannel"].setValue("rgba.alpha")
    blend["mix"].setValue(1)

    switch = nuke.toNode("KyvenROIIntegrationSwitch")
    if switch is None:
        switch = nuke.nodes.Switch(name="KyvenROIIntegrationSwitch")
    switch.setInput(0, refined)
    switch.setInput(1, blend)
    switch["which"].setExpression(
        "parent.roi_blend && parent.crop_mode==1 && KyvenMatteSwitch.which"
    )
    return switch
