"""Compare shared host contracts across all Kyven node families, without inference."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "hosts/nuke"), str(ROOT / "src")]
import kyven_nuke  # noqa: F401
import nuke
from kyven_nuke import dlss_node, generative_fill_node, inpaint_node, refine_node
from kyven_nuke import node as segment

destination = Path(os.environ["KYVEN_NODE_CONSISTENCY_REPORT"])
fmt = nuke.addFormat("48 32 0 0 48 32 1 NodeConsistency")
base = nuke.nodes.Constant(format=fmt.name())
source = nuke.nodes.Expression(inputs=[base], expr0="x/64-0.25", expr1="y/64+1",
                                expr2="0.12345679", expr3="0.37")
mask = nuke.nodes.Expression(inputs=[base], expr0="0", expr1="0", expr2="0",
                              expr3="x>10 && x<35 && y>8 && y<24")
factories = [
    ("Segment", segment.create_segment_node),
    ("Refine", refine_node.create_refine_node),
    ("Inpaint", inpaint_node.create_inpaint_node),
    ("Generative Fill", generative_fill_node.create_generative_fill_node),
    ("DLSS", dlss_node.create_dlss_node),
]
records = []
for kind, create in factories:
    for selected in nuke.selectedNodes():
        selected.setSelected(False)
    current = create()
    source_input = int(current.node("Source")["number"].value())
    current.setInput(source_input, source)
    if current.node("Mask") is not None:
        current.setInput(int(current.node("Mask")["number"].value()), mask)
    if "live_mode" in current.knobs():
        current["live_mode"].setValue(False)
    for name in ("kyven_uuid", "cache_root", "cache_version", "kyven_status",
                 "start_server_control", "stop_server_control", "process_frame", "cancel"):
        assert name in current.knobs(), (kind, name)
    modes = list(current["output_mode"].values())
    bypass = next(label for label in modes if label in {"Source", "Bypass", "Source (Bypass)"})
    current["output_mode"].setValue(bypass)
    for x, y in ((2.5, 3.5), (24.5, 16.5), (45.5, 27.5)):
        for channel in ("rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"):
            assert current.sample(channel, x, y) == source.sample(channel, x, y), (kind, channel)
    assert current["input_cache_scope"].value() == "Current Frame", kind
    assert current["process_frame"].visible() and not current["process_range"].visible(), kind
    current["input_cache_scope"].setValue("Frame Range")
    assert current["process_range"].visible() and not current["process_frame"].visible(), kind
    from kyven_nuke.input_cache import ensure_controls
    ensure_controls(current)
    assert current["input_cache_scope"].value() == "Frame Range", kind
    current["input_cache_scope"].setValue("Current Frame")
    assert not current["process_range"].visible() and current["process_frame"].visible(), kind
    assert segment._begin_node_operation(current), kind
    assert not current["process_frame"].enabled() and current["cancel"].enabled(), kind
    segment._set_busy(current.fullName(), False)
    assert current["process_frame"].enabled() and not current["cancel"].enabled(), kind
    records.append({"kind": kind, "name": current.name(), "uuid": current["kyven_uuid"].value(),
                    "source_input": source_input, "bypass": bypass, "source_exact": True,
                    "cache_controls": True, "server_controls": True, "busy_controls": True})
assert len({item["uuid"] for item in records}) == len(records)
graph = destination.with_suffix(".nk")
nuke.scriptSaveAs(graph.as_posix(), overwrite=1)
nuke.scriptClear()
nuke.scriptOpen(graph.as_posix())
for item in records:
    current = nuke.toNode(item["name"])
    assert current["kyven_uuid"].value() == item["uuid"]
    assert current["output_mode"].value() == item["bypass"]
    assert not current["kyven_busy"].value()
    assert current.node("Source")["number"].value() == item["source_input"]
    item["reopened"] = True
destination.write_text(json.dumps({"ok": True, "nodes": records}, indent=2))
print(json.dumps(records), flush=True)
