"""Native Nuke controls, legacy migration and float/alpha preservation; no inference."""
import json
import os
import sys
import traceback
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"hosts/nuke"), str(ROOT/"src")]
import kyven_nuke  # noqa: F401 - callback namespace
import nuke
from kyven_nuke import generative_fill_node as fill
from kyven_nuke import inpaint_node

destination = Path(os.environ["KYVEN_ZIMAGE_NUKE_REPORT"])
report = {"checks": []}


def check(name, value):
    assert value, name
    report["checks"].append(name)


try:
    node = fill.create_generative_fill_node()
    check("Klein remains default", fill._model_id(node) == fill.PROVIDER_ID)
    check("MOSS retired from selector", not any("MOSS" in x for x in node["model"].values()))
    node["model"].setValue(1)
    check("Native Inpaint selection", fill._model_id(node) == fill.MODEL_IDS[1])
    check("native memory/step defaults", node["steps"].value() == 28 and node["working_size"].value() == 768)
    check("context inference overlap default", node["fill_mask_overlap"].value() == 64)
    check("irrelevant LanPaint solver disabled", not node["thinking_steps"].enabled())
    check("native guidance enabled", node["zimage_guidance"].enabled())
    node["zimage_control_scale"].setValue(0.85)
    node["steps"].setValue(30)
    fill._ensure_fill_graph(node)
    check("refresh preserves native settings", node["steps"].value() == 30 and node["zimage_control_scale"].value() == 0.85)
    callback = node["knobChanged"].value()
    node["knobChanged"].setValue("")
    node["model"].setValues([fill.MODEL_LABELS[0], "Z-Image Base Q4 + Native ControlNet Inpaint"])
    node["model"].setValue(1)
    node["knobChanged"].setValue(callback)
    fill._ensure_model_control(node)
    check("older native label retains selection", fill._model_id(node) == fill.MODEL_IDS[1] and node["steps"].value() == 30)
    check("native VAE tile size automatic", not node["vae_tile_size"].enabled())
    node["prompt"].setValue("Remove the person")
    with patch.object(fill.inpaint_node, "_payload", return_value={}):
        payload = fill._payload(node)
    check("native provider exported", payload["model_id"] == fill.MODEL_IDS[1])
    check("fractional context strength exported", payload["fill"]["zimage_control_scale"] == 0.85)
    fmt = nuke.addFormat("16 16 0 0 16 16 1 KyvenNativeFillTest")
    base = nuke.nodes.Constant(format=fmt.name())
    source = nuke.nodes.Expression(inputs=[base], expr0="x/8+1", expr1="y/8-0.5", expr2="0.123456", expr3="0.37")
    mask = nuke.nodes.Expression(inputs=[base], expr0="0", expr1="0", expr2="0",
        expr3="x>=4 && x<12 && y>=4 && y<12 ? 1 : 0")
    generated = nuke.nodes.Constant(format=fmt.name())
    generated["color"].setValue([0.9, 0.1, 0.7, 1])
    patch_path = destination.with_name("native-synthetic-patch.exr")
    writer = nuke.nodes.Write(inputs=[generated], file=patch_path.as_posix(), file_type="exr")
    writer["channels"].setValue("rgba")
    writer["raw"].setValue(True)
    nuke.execute(writer, 1, 1)
    cache_mask = nuke.nodes.Expression(inputs=[base], expr0="x>=4 && x<12 && y>=4 && y<12 ? 1 : 0", expr1="r", expr2="r", expr3="r")
    mask_path = destination.with_name("native-synthetic-mask.exr")
    writer.setInput(0, cache_mask)
    writer["file"].setValue(mask_path.as_posix())
    nuke.execute(writer, 1, 1)
    node.setInput(inpaint_node.SOURCE_INPUT, source)
    node.setInput(inpaint_node.MASK_INPUT, mask)
    inpaint_node._set_patch(node, patch_path)
    inpaint_node._set_processed_mask(node, mask_path)
    check("native cached pixels affect selected area", abs(nuke.sample(node, "rgba.red", 8, 8)-nuke.sample(source, "rgba.red", 8, 8)) > 0.1)
    for channel in ("rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"):
        for x, y in ((1, 1), (14, 14)):
            check(f"float Source preserved {channel} {x},{y}", nuke.sample(node, channel, x, y) == nuke.sample(source, channel, x, y))
    check("source alpha retained inside selection", nuke.sample(node, "rgba.alpha", 8, 8) == nuke.sample(source, "rgba.alpha", 8, 8))
    rgb = nuke.nodes.Remove(inputs=[source], operation="keep", channels="rgb")
    node.setInput(inpaint_node.SOURCE_INPUT, rgb)
    check("RGB input has opaque alpha and no missing-channel error", nuke.sample(node, "rgba.alpha", 1, 1) == 1 and not node.error())
    callback = node["knobChanged"].value()
    node["knobChanged"].setValue("")
    node["model"].setValues([fill.MODEL_LABELS[0], "MOSS SFT INT8 + Edit + LanPaint (Experimental)"])
    node["model"].setValue(1)
    for name in ("zimage_guidance", "zimage_control_scale", "zimage_negative_prompt"):
        node.removeKnob(node[name])
    node["knobChanged"].setValue(callback)
    fill._ensure_fill_graph(node)
    check("retired MOSS node migrates to native", fill._model_id(node) == fill.MODEL_IDS[1] and node["steps"].value() == 28)
    node["model"].setValue(0)
    check("Klein remains usable", node["steps"].value() == 4 and node["thinking_steps"].enabled())
    check("Torch stays outside Nuke", "torch" not in sys.modules)
    report["ok"] = True
except Exception:  # noqa: BLE001 - record any native Nuke failure in the report
    report["ok"] = False
    report["error"] = traceback.format_exc()
finally:
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    nuke.scriptExit()
