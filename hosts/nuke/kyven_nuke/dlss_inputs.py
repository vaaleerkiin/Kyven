"""One float, single-part EXR snapshot per DLSS frame. No image decoding in Nuke."""
from __future__ import annotations

from pathlib import Path
from typing import Any

CONTRACT = "multilayer-exr-v1"
MOTION = tuple("kyven_dlss_motion." + c for c in ("x", "y", "forward_x", "forward_y"))
DEPTH = "kyven_dlss_depth.Z"
MASK = "kyven_dlss_mask.value"
WRITER = "KyvenDlssInputWrite"


def nuke_channel(name: str) -> str:
    aliases = {"R": "red", "G": "green", "B": "blue", "A": "alpha",
               "red": "red", "green": "green", "blue": "blue", "alpha": "alpha"}
    return "rgba." + aliases[name] if name in aliases else name


def mappings(node: Any, settings: dict[str, Any]) -> list[tuple[int, str, list[tuple[str, str]]]]:
    """Resolve artist-selected channels to distinct transport layers."""
    motion = [(settings.get("motion_x", "backward.u"), MOTION[0]),
              (settings.get("motion_y", "backward.v"), MOTION[1])]
    if settings.get("temporal") and settings.get("motion_consistency"):
        motion += [(settings["forward_x"], MOTION[2]), (settings["forward_y"], MOTION[3])]
    return [(1, "Motion", motion if node.input(1) is not None else []),
            (2, "Depth", [(settings.get("depth_channel", "depth.Z"), DEPTH)]
             if node.input(2) is not None else []),
            (3, "Mask", [("rgba.alpha", MASK)]
             if settings["mask_source"] == "input" and node.input(3) is not None else [])]


def snapshot_state(node: Any, frame: int) -> dict[str, Any]:
    from kyven_nuke.node import _knob_value_at

    names = ("motion_x", "motion_y", "forward_x", "forward_y", "depth_channel",
             "motion_consistency", "temporal", "mask_source")
    settings = {name: _knob_value_at(node[name], frame) for name in names}
    choice = settings["mask_source"]
    settings["mask_source"] = "input" if choice in {"Mask Input", "Auto"} else "none"
    return {"contract": CONTRACT,
            "channels": {label: [[nuke_channel(src), dst] for src, dst in pairs]
                         for _, label, pairs in mappings(node, settings) if pairs}}


def ensure_graph(nuke: Any, node: Any) -> Any:
    """Install once, including on restored nodes; keep result and artist inputs intact."""
    for layer, channels in (("kyven_dlss_motion", MOTION),
                            ("kyven_dlss_depth", (DEPTH,)), ("kyven_dlss_mask", (MASK,))):
        if layer not in nuke.layers():
            nuke.Layer(layer, list(channels))
    node.begin()
    try:
        source = nuke.toNode("KyvenDlssInputRGBA")
        if source is None:
            source = nuke.nodes.Remove(name="KyvenDlssInputRGBA")
            source.setInput(0, nuke.toNode("Source"))
            source["operation"].setValue("keep")
            source["channels"].setValue("rgba")
        previous = source
        for name in ("Motion", "Depth", "Mask"):
            copy = nuke.toNode("KyvenDlssPack" + name)
            if copy is None:
                copy = nuke.nodes.Copy(name="KyvenDlssPack" + name)
                copy["disable"].setValue(True)
            copy.setInput(0, previous)
            copy.setInput(1, nuke.toNode(name))
            previous = copy
        bounds = nuke.toNode("KyvenDlssInputBounds")
        if bounds is None:
            bounds = nuke.nodes.Crop(name="KyvenDlssInputBounds")
            bounds["box"].setValue([0, 0, 0, 0])
            bounds["box"].setExpression("input.width", 2)
            bounds["box"].setExpression("input.height", 3)
            bounds["reformat"].setValue(False)
        bounds.setInput(0, previous)
        writer = nuke.toNode(WRITER)
        if writer is None:
            from kyven_nuke.color_management import set_data_io

            writer = nuke.nodes.Write(name=WRITER)
            writer["file_type"].setValue("exr")
            writer["channels"].setValue("all")
            writer["datatype"].setValue("32 bit float")
            writer["compression"].setValue("Zip (16 scanlines)")
            writer["interleave"].setValue("channels, layers and views")
            writer["autocrop"].setValue(False)
            set_data_io(writer)
        writer.setInput(0, bounds)
        for name in ("Source", "Motion", "Depth", "Mask"):
            legacy = nuke.toNode("KyvenDlss" + name + "Write")
            if legacy is not None:
                nuke.delete(legacy)
        return writer
    finally:
        node.end()


def configure(nuke: Any, node: Any, settings: dict[str, Any]) -> Any:
    """Check guide dimensions/channels before Copy could silently fill them with black."""
    source = node.input(0)
    source.forceValidate()
    missing = {"rgba.red", "rgba.green", "rgba.blue"} - set(source.channels())
    if missing:
        raise RuntimeError(f"DLSS Source is missing RGB channels: {sorted(missing)}")
    plan = mappings(node, settings)
    for index, label, pairs in plan:
        if not pairs:
            continue
        guide = node.input(index)
        guide.forceValidate()
        if (guide.width(), guide.height()) != (source.width(), source.height()):
            raise RuntimeError(f"DLSS {label} must match Source dimensions. Reformat it before connecting.")
        missing = {nuke_channel(channel) for channel, _ in pairs} - set(guide.channels())
        if missing:
            raise RuntimeError(f"DLSS {label} is missing channels: {sorted(missing)}")
    writer = ensure_graph(nuke, node)
    node.begin()
    try:
        for _, label, pairs in plan:
            copy = nuke.toNode("KyvenDlssPack" + label)
            for index in range(4):
                src, dst = pairs[index] if index < len(pairs) else ("none", "none")
                copy[f"from{index}"].setValue(nuke_channel(src))
                copy[f"to{index}"].setValue(dst)
            copy["disable"].setValue(not pairs)
    finally:
        node.end()
    return writer


def payload(node: Any, settings: dict[str, Any], path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"source": str(path)}
    if node.input(1) is not None:
        result.update(motion=str(path), motion_x=MOTION[0], motion_y=MOTION[1])
        if settings.get("temporal") and settings.get("motion_consistency"):
            result.update(forward_x=MOTION[2], forward_y=MOTION[3])
    if node.input(2) is not None:
        result.update(depth=str(path), depth_channel=DEPTH)
    if settings["mask_source"] == "input":
        result.update(mask=str(path), mask_channel=MASK)
    return result
