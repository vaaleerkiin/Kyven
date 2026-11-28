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
    check("8 GB default working size", node["working_size"].value() == 512)
    check("Mage default model", "Mage" in node["model"].value())
    check("Mage processing resolution reaches 2048", node["working_size"].maximum() == 2048)
    check("distilled four-step default", node["steps"].value() == 4)
    check("three supported model choices", len(node["model"].values()) == 3)
    check("Mage experimental boundary controls hidden", not node["fill_seam_repair"].visible())
    check("Mage hides LanPaint controls", not node["thinking_steps"].visible())
    check("Mage hides unused VAE tiling", not node["vae_tile_size"].visible())
    check("Mage hides Base guidance", not node["moss_guidance"].visible())
    check("single complete install action", node["install_fill_runtime"].label() == "Install / Repair Model")
    check("Color Match available", node["fill_color_mode"].visible() and node["fill_color_match"].visible())
    check("Color Match defaults to Auto", node["fill_color_mode"].value() == "Auto")
    check("ordinary edge softness is accessible", node["edge_softness"].visible() and node["edge_softness"].value() == 4)
    check("strict artist alpha is the default", node["fill_blend_margin"].value() == 0)
    node["fill_blend_margin"].setValue(2)
    check("expanded blend support reaches the requested margin", node.node("KyvenFillBlendSupport").sample("rgba.red", 2.5, 8.5) > 0)
    check("expanded blend support stays inside the requested margin", node.node("KyvenFillBlendSupport").sample("rgba.red", 0.5, 8.5) == 0)
    node["fill_blend_margin"].setValue(0)
    check("Mage disables LanPaint settings", not node["thinking_steps"].enabled())
    node["model"].setValue(1)
    fill._ensure_model_control(node)
    check("Base selection routes correctly", fill._model_id(node) == fill.MODEL_IDS[1])
    check("Base guidance enabled", node["moss_guidance"].enabled())
    node["model"].setValue(2)
    fill._ensure_model_control(node)
    check("Mage selection routes correctly", fill._model_id(node) == fill.MAGE_PROVIDER_ID)
    check("Mage guidance disabled", not node["moss_guidance"].enabled())
    node["model"].setValue(0)
    fill._ensure_model_control(node)
    check("FLUX still enables LanPaint", node["thinking_steps"].enabled())
    check("FLUX keeps its qualified resolution range", node["working_size"].maximum() == 1024)
    check("FLUX shows LanPaint controls", node["thinking_steps"].visible())
    node["model"].setValue(2)
    fill._ensure_model_control(node)
    node["working_size"].setValue(2048)
    node["model"].setValue(0)
    fill._ensure_model_control(node)
    check("switching to FLUX clamps a Mage-only resolution", node["working_size"].value() == 1024)
    node["model"].setValue(2)
    node["working_size"].setValue(512)
    fill._ensure_model_control(node)
    node["model"].setValues([*fill.MODEL_LABELS[:2], "Mage-Flow Edit Turbo Q6 + Boundary Repair"])
    node["model"].setValue(2)
    fill._ensure_model_control(node)
    check("old Mage label migrates without selecting FLUX", fill._model_id(node) == fill.MAGE_PROVIDER_ID)

    check("Result is the default", node["output_mode"].value() == "Result")
    check("inference grow does not select output mask",
          node.node("KyvenEffectiveMask").input(0).name() == "KyvenFillPermission")
    node["prompt"].setValue("a red ceramic vase")
    payload = fill._payload(node, source, destination.with_name("source.png"),
        destination.with_name("mask.png"), destination.with_name("model-mask.png"),
        destination.with_name("out.png"), destination.with_name("processed.png"),
        destination.with_name("patch.png"), "alpha")
    check("Mage payload bypasses LaMa model indexes", payload["model_id"] == fill.MAGE_PROVIDER_ID)
    check("Mage uses standard edges without neural repair", payload["fill"]["seam_repair"] == "off" and payload["fill"]["seamless_blend"] == "off")
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
    expanded_mask = nuke.nodes.Expression(inputs=[base],
        expr0="x>=2 && x<14 && y>=2 && y<14 ? 1 : 0", expr1="r", expr2="r", expr3="r")
    expanded_path = destination.with_name("synthetic-expanded-mask.exr")
    writer.setInput(0, expanded_mask)
    writer["file"].setValue(expanded_path.as_posix())
    nuke.execute(writer, 1, 1)
    node["fill_blend_margin"].setValue(2)
    inpaint_node._set_processed_mask(node, expanded_path)
    check("explicit expanded band receives generated RGB", abs(node.sample("rgba.red", 3.5, 8.5)-source.sample("rgba.red", 3.5, 8.5)) > .1)
    check("outside expanded band preserves Source", node.sample("rgba.red", .5, 8.5) == source.sample("rgba.red", .5, 8.5))
    for mode in (1, 3):
        node["output_mode"].setValue(mode)
        expected_alpha = node.node("KyvenFillClippedMask").sample("rgba.red", 3.5, 8.5)
        check(f"expanded permission is present in output alpha mode {mode}",
              expected_alpha > 0 and abs(node.sample("rgba.alpha", 3.5, 8.5)-expected_alpha) < 1e-6)
    node["output_mode"].setValue(0)
    node["model"].setValue(0)
    fill._ensure_model_control(node)
    check("FLUX ignores a hidden Mage blend margin", node.sample("rgba.red", 3.5, 8.5) == source.sample("rgba.red", 3.5, 8.5))
    node["model"].setValue(2)
    node["fill_blend_margin"].setValue(0)
    fill._ensure_model_control(node)
    inpaint_node._set_processed_mask(node, mask_path)
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
