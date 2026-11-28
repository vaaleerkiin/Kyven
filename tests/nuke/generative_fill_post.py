"""Exercise real cached native Mage controls with synthetic pixels, no inference."""
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "hosts/nuke"), str(ROOT / "src")]
import kyven_nuke  # noqa: F401 - Nuke knob callbacks
import nuke
from kyven_nuke import generative_fill_node as fill
from kyven_nuke import generative_fill_post as post
from kyven_nuke import inpaint_node

destination = Path(os.environ["KYVEN_FILL_POST_REPORT"])
report = {"checks": []}


def check(label, condition):
    assert condition, label
    report["checks"].append(label)


try:
    fmt = nuke.addFormat("96 64 0 0 96 64 1 MagePostTest")
    base = nuke.nodes.Constant(format=fmt.name())
    source = nuke.nodes.Expression(inputs=[base], expr0="x/64-0.2", expr1="y/64+1",
                                   expr2="0.123456789", expr3="0.37")
    mask = nuke.nodes.Expression(inputs=[base], expr0="0", expr1="0", expr2="0",
                                 expr3="x>=24 && x<72 && y>=16 && y<48")
    node = fill.create_generative_fill_node()
    node.setInput(1, source)
    node.setInput(0, mask)
    node["edge_softness"].setValue(0)
    def forbidden_server():
        raise AssertionError("A live look adjustment requested the inference server")
    fill.ensure_server = forbidden_server
    check("ordinary blur is disabled by default", node["edge_softness"].value() == 0)
    check("Mage hides generation growth and cutoff", all(not node[k].visible()
          for k in ("mask_grow", "mask_threshold", "fill_mask_overlap")))
    check("no compound removal preset", "fill_smooth_removal" not in node.knobs()
          or not node["fill_smooth_removal"].visible())
    names = list(node.knobs())
    check("result mask block follows output", names.index("output_help")
          < names.index("fill_result_mask_section") < names.index("fill_blend_margin")
          < names.index("edge_softness") < names.index("fill_preview_blend_mask"))
    node["fill_preview_blend_mask"].setValue(True)
    check("result mask can be previewed before generation", node.sample("rgba.red", 48.5, 32.5) == 1
          and node.sample("rgba.red", 1.5, 1.5) == 0 and not node.error())
    node["fill_preview_blend_mask"].setValue(False)
    model_input = node.node("KyvenInpaintModelMaskWrite").input(0)
    samples = [(x+.5, 32.5) for x in range(96)]
    model_before = [model_input.sample("rgba.red", x, y) for x, y in samples]
    check("model export has no implicit growth", model_before ==
          [int(24 <= x < 72) for x in range(96)])
    check("preview uses the exact exported mask", node.node("KyvenOutputSwitch").input(6) == model_input)
    check("two active model choices", list(node["model"].values()) == list(fill.MODEL_LABELS))
    check("resolution is a visible dropdown", node["working_resolution"].visible()
          and not node["working_size"].visible())
    check("Mage menu reaches native 2048 maximum", "2048" in node["working_resolution"].values())
    node["working_resolution"].setValue("2048")
    check("menu sends pixels rather than enum index", node["working_size"].value() == 2048)
    patch = destination.with_name("raw.0001.png")
    matched = destination.with_name("raw.0001.lab.png")
    constant = nuke.nodes.Constant(format=fmt.name())
    writer = nuke.nodes.Write(inputs=[constant], file_type="png")
    writer["channels"].setValue("rgba")
    writer["raw"].setValue(True)
    for path, color in ((patch, [.8, .2, .2, 1]), (matched, [.2, .8, .3, 1])):
        constant["color"].setValue(color)
        writer["file"].setValue(path.as_posix())
        nuke.execute(writer, 1, 1)
    nuke.frame(1)
    inpaint_node._set_patch(node, patch)
    check("actual paired files activate native controls", bool(node["fill_native_ready"].value()))
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (patch, matched)}
    before = node.sample("rgba.red", 48.5, 32.5)
    node["fill_color_match"].setValue(0)
    raw = node.sample("rgba.red", 48.5, 32.5)
    check("strength changes cached RGB", abs(before-raw) > .1)
    node["fill_color_match"].setValue(1)
    node["fill_match_luminance"].setValue(0)
    a = node.node("KyvenFillLabMix").sample("rgba.red", 48.5, 32.5)
    b = node.node("KyvenFillRawLab").sample("rgba.red", 48.5, 32.5)
    check("luminance can remain raw", abs(a-b) < 1e-6)
    node["fill_match_chroma"].setValue(0)
    for channel in ("rgba.green", "rgba.blue"):
        check("chroma can remain raw " + channel,
              abs(node.node("KyvenFillLabMix").sample(channel, 48.5, 32.5)
                  - node.node("KyvenFillRawLab").sample(channel, 48.5, 32.5)) < 1e-6)
    node["fill_color_mode"].setValue("Off")
    check("Off bypasses Lab roundtrip", abs(node.sample("rgba.red", 48.5, 32.5)-raw) < 1e-6)
    node["fill_preview_blend_mask"].setValue(True)
    edge_before = node.sample("rgba.red", 24.5, 32.5)
    node["edge_softness"].setValue(8)
    edge_after = node.sample("rgba.red", 24.5, 32.5)
    check("edge softness changes native alpha", edge_before > .99 and 0 <= edge_after < .8)
    check("blend mask alpha equals its grayscale preview",
          abs(node.sample("rgba.alpha", 24.5, 32.5)-edge_after) < 1e-6)
    check("ordinary blur extends across the original edge", 0 < node.sample("rgba.red", 23.5, 32.5) < .5)
    reference = nuke.nodes.Blur(inputs=[node.node("KyvenFillPermission")])
    reference["channels"].setValue("rgba")
    reference["filter"].setValue("gaussian")
    reference["size"].setValue(8)
    check("blur matches ordinary native Gaussian Blur without automatic erosion", all(
        abs(node.sample("rgba.red", x, y)-reference.sample("rgba.red", x, y)) < 1e-6
        for x, y in samples))
    node["edge_softness"].setValue(0)
    node["fill_blend_margin"].setValue(4)
    check("positive grow expands result alpha", node.sample("rgba.red", 22.5, 32.5) > .99)
    node["fill_blend_margin"].setValue(-4)
    check("negative grow erodes result alpha", node.sample("rgba.red", 26.5, 32.5) == 0)
    check("post mask controls never alter exported model pixels", model_before ==
          [model_input.sample("rgba.red", x, y) for x, y in samples])
    node["mask_grow"].setValue(18)
    node["mask_threshold"].setValue(.8)
    node["fill_mask_overlap"].setValue(32)
    check("hidden legacy controls cannot modify Mage model export", model_before ==
          [model_input.sample("rgba.red", x, y) for x, y in samples])
    node["fill_blend_margin"].setValue(0)
    node["edge_softness"].setValue(8)
    node["fill_preview_blend_mask"].setValue(False)
    for x, y in ((1.5, 1.5), (90.5, 60.5)):
        for channel in ("rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"):
            check(f"float Source stays exact {channel} {x},{y}",
                  node.sample(channel, x, y) == source.sample(channel, x, y))
    check("Result preserves Source alpha inside edit", node.sample("rgba.alpha", 48.5, 32.5)
          == source.sample("rgba.alpha", 48.5, 32.5))
    check("no patch writes while adjusting look", all(
        hashlib.sha256(path.read_bytes()).hexdigest() == sha for path, sha in hashes.items()))
    node["fill_color_mode"].setValue("Lab")
    node["fill_match_luminance"].setValue(1)
    node["fill_match_chroma"].setValue(1)
    node["edge_softness"].setValue(0)
    old_pixel = node.sample("rgba.red", 48.5, 32.5)
    constant["color"].setValue([.7, .3, .8, 1])
    writer["file"].setValue(matched.as_posix())
    nuke.execute(writer, 1, 1)
    inpaint_node._set_patch(node, patch)
    check("reprocessing reloads the matched sidecar at the same path",
          abs(node.sample("rgba.red", 48.5, 32.5)-old_pixel) > .1)
    second = patch.with_name("raw.0002.png")
    second.write_bytes(patch.read_bytes())
    inpaint_node._set_patch(node, patch.with_name("raw.%04d.png"), 1, 2)
    nuke.frame(2)
    check("a missing partner does not activate another frame's Lab patch",
          not node["fill_native_ready"].valueAt(2))
    check("missing partner falls back without a graph error", not node.error())
    check("a missing partner renders the raw fallback",
          abs(node.sample("rgba.red", 48.5, 32.5)-raw) < 1e-6)
    nuke.frame(1)
    check("timeline returns to the prepared pair", node["fill_native_ready"].valueAt(1))
    inpaint_node._set_patch(node, patch)
    coverage = nuke.nodes.Expression(inputs=[constant], expr3="x>=40 && x<80")
    writer.setInput(0, coverage)
    for path in (patch, matched):
        writer["file"].setValue(path.as_posix())
        nuke.execute(writer, 1, 1)
    inpaint_node._set_patch(node, patch)
    node["edge_softness"].setValue(8)
    node["fill_preview_blend_mask"].setValue(True)
    check("crop coverage does not introduce extra inward erosion",
          node.sample("rgba.red", 40.5, 32.5) > .99)
    check("manual crop coverage cannot expose ungenerated pixels",
          node.sample("rgba.red", 39.5, 32.5) == 0)
    node["fill_preview_blend_mask"].setValue(False)
    args = (node, source, patch, patch, patch, patch, patch, patch, "alpha")
    payload_before = fill._payload(*args)
    node["fill_color_match"].setValue(.37)
    node["edge_softness"].setValue(3)
    node["fill_blend_margin"].setValue(-7.5)
    check("look changes leave generation request unchanged", fill._payload(*args) == payload_before)
    node["model"].setValue(fill.MODEL_LABELS[0])
    check("FLUX dropdown stops at 1024", "2048" not in node["working_resolution"].values()
          and node["working_size"].value() <= 1024)
    node["model"].setValue(fill.MODEL_LABELS[1])
    script = destination.with_suffix(".nk")
    name = node.name()
    nuke.scriptSaveAs(script.as_posix(), overwrite=1)
    nuke.scriptClear()
    nuke.scriptOpen(script.as_posix())
    reopened = nuke.toNode(name)
    check("controls survive reopen", reopened["fill_color_match"].value() == .37
          and reopened["edge_softness"].value() == 3)
    check("paired cache survives reopen", post.native_ready(reopened))
    report["ok"] = True
except Exception:  # noqa: BLE001 - persist failures from the separate Nuke process
    report.update(ok=False, traceback=traceback.format_exc())
finally:
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
