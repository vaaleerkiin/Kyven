"""Native Nuke graph contract; uses synthetic cached pixels, no inference."""
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "hosts/nuke"), str(ROOT / "src")]
import kyven_nuke  # noqa: F401 - Nuke evaluates node callbacks in this global namespace
import nuke
from kyven_nuke import generative_fill_node as fill
from kyven_nuke import inpaint_node
from kyven_nuke.node import _begin_node_operation

report = {"checks": []}
destination = Path(os.environ["KYVEN_FILL_REPORT"])


def check(label, value):
    assert value, label
    report["checks"].append(label)


try:
    fmt = nuke.addFormat("16 16 0 0 16 16 1 KyvenFillTest")
    base = nuke.nodes.Constant(format=fmt.name())
    source = nuke.nodes.Expression(inputs=[base], expr0="x/8+1", expr1="y/8-0.5",
                                   expr2="0.123456", expr3="0.37")
    mask = nuke.nodes.Expression(inputs=[base], expr0="0", expr1="0", expr2="0",
                                 expr3="x>=4 && x<12 && y>=4 && y<12 ? 1 : 0")
    node = fill.create_generative_fill_node()
    node.setInput(inpaint_node.SOURCE_INPUT, source)
    node.setInput(inpaint_node.MASK_INPUT, mask)
    check("Advanced tab exists", "advanced" in node.knobs())
    check("Qwen system instruction is editable", bool(node["qwen_system_prompt"].value()))
    check("Nuke generator button routes to Kyven", "generative_fill_node" in node["process_frame"].value())
    check("8 GB default working size", node["working_size"].value() == 768)
    check("instruction-trained Klein model", "Klein" in node["model"].value())
    check("distilled four-step default", node["steps"].value() == 4)
    check("Result is the default", node["output_mode"].value() == "Result")
    check("inference grow does not select output mask",
          node.node("KyvenEffectiveMask").input(0).name() == "KyvenFillPermission")
    node["prompt"].setValue("a red ceramic vase")
    node["mask_grow"].setValue(8)
    generated = nuke.nodes.Constant(format=fmt.name())
    generated["color"].setValue([0.9, 0.1, 0.7, 1])
    patch_path = destination.with_name("synthetic-patch.exr")
    writer = nuke.nodes.Write(inputs=[generated], file=patch_path.as_posix(), file_type="exr")
    writer["channels"].setValue("rgba")
    writer["raw"].setValue(True)
    nuke.execute(writer, 1, 1)
    cache_mask = nuke.nodes.Expression(inputs=[base],
        expr0="x>=4 && x<12 && y>=4 && y<12 ? 1 : 0",
        expr1="r", expr2="r", expr3="r")
    mask_path = destination.with_name("synthetic-mask.exr")
    writer.setInput(0, cache_mask)
    writer["file"].setValue(mask_path.as_posix())
    nuke.execute(writer, 1, 1)
    inpaint_node._set_patch(node, patch_path)
    inpaint_node._set_processed_mask(node, mask_path)
    check("generated cache changes the selected region",
          abs(nuke.sample(source, "rgba.red", 8, 8) - nuke.sample(node, "rgba.red", 8, 8)) > 0.1)
    for x, y in ((1, 1), (14, 14), (1, 8), (8, 1)):
        for channel in ("rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"):
            check(f"Source preserved {channel} at {x},{y}",
                  nuke.sample(source, channel, x, y) == nuke.sample(node, channel, x, y))
    mask["expr3"].setValue("0")
    for channel in ("rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"):
        check(f"live empty mask clips the cached patch {channel}",
              nuke.sample(source, channel, 8, 8) == nuke.sample(node, channel, 8, 8))
    original_prompt = "женщина в чёрных очках"
    translated_job = {"status": "succeeded", "result": {"metadata": {
        "auto_translated": True, "original_prompt": original_prompt,
        "effective_prompt": "a woman wearing black sunglasses"}}}
    node["prompt"].setValue(original_prompt)
    _begin_node_operation(node)
    inpaint_node._apply(node.fullName(), translated_job, patch_path, mask_path, patch_path)
    check("automatic translation updates the visible prompt",
          node["prompt"].value() == "a woman wearing black sunglasses")
    node["prompt"].setValue("a newer artist edit")
    _begin_node_operation(node)
    inpaint_node._apply(node.fullName(), translated_job, patch_path, mask_path, patch_path)
    check("late translation preserves a newer artist edit", node["prompt"].value() == "a newer artist edit")
    rgb_source = nuke.nodes.Remove(inputs=[source], operation="keep", channels="rgb")
    # Saved preview groups may still have the original unsafe Copy connection.
    node.node("KyvenFillSourceAlpha").setInput(1, node.node("Source"))
    fill._ensure_fill_graph(node)
    node.setInput(inpaint_node.SOURCE_INPUT, rgb_source)
    check("RGB fixture has no alpha", "rgba.alpha" not in rgb_source.channels())
    check("RGB source receives opaque alpha", nuke.sample(node, "rgba.alpha", 1, 1) == 1)
    check("RGB source renders without missing-channel error", not node.error())
    for channel in ("rgba.red", "rgba.green", "rgba.blue"):
        check(f"RGB Source preserved {channel}",
              nuke.sample(rgb_source, channel, 1, 1) == nuke.sample(node, channel, 1, 1))
    node.setInput(inpaint_node.SOURCE_INPUT, source)
    check("RGBA alpha survives reconnecting", nuke.sample(node, "rgba.alpha", 1, 1) ==
          nuke.sample(source, "rgba.alpha", 1, 1))
    rgb_source["disable"].setValue(True)
    node.setInput(inpaint_node.SOURCE_INPUT, rgb_source)
    check("upstream alpha changes are detected without rebuilding", nuke.sample(node, "rgba.alpha", 1, 1) ==
          nuke.sample(source, "rgba.alpha", 1, 1))
    rgb_source["disable"].setValue(False)
    check("upstream RGB changes restore opaque alpha", nuke.sample(node, "rgba.alpha", 1, 1) == 1)
    node["prompt"].setValue("a red ceramic vase")
    script = destination.with_suffix(".nk")
    node_name = node.name()
    nuke.scriptSaveAs(str(script), overwrite=1)
    nuke.scriptClear()
    nuke.scriptOpen(str(script))
    reopened = nuke.toNode(node_name)
    check("prompt survives reopening", reopened["prompt"].value() == "a red ceramic vase")
    check("system prompt survives reopening", bool(reopened["qwen_system_prompt"].value()))
    check("RGB alpha fallback survives reopening", nuke.sample(reopened, "rgba.alpha", 1, 1) == 1)
    report["ok"] = True
except Exception:  # noqa: BLE001 - record any Nuke smoke-test failure before exiting
    report["ok"] = False
    report["traceback"] = traceback.format_exc()
finally:
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    nuke.scriptExit()
