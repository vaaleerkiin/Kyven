"""Native Nuke identity round trip: coordinates, odd formats and nonzero bbox.

No inference: returned model pixels equal the actual exported sRGB input.
This isolates host placement/transfer errors from intentional model edits.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "hosts/nuke"), str(ROOT / "src")]
import kyven_nuke  # noqa: F401
import nuke
from kyven_nuke import generative_fill_node as fill
from kyven_nuke import inpaint_node

destination = Path(os.environ["KYVEN_FILL_ALIGNMENT_REPORT"])
destination.parent.mkdir(parents=True, exist_ok=True)
results = []
for width, height, cropped in [(641, 389, False), (641, 389, True), (2048, 1080, False)]:
    label = f"{width}x{height}" + ("-bbox" if cropped else "")
    fmt = nuke.addFormat(f"{width} {height} 0 0 {width} {height} 1 Align{label}")
    base = nuke.nodes.Constant(format=fmt.name())
    source = nuke.nodes.Expression(inputs=[base], expr0="0.05+0.6*x/width",
        expr1="0.04+0.5*y/height", expr2="((floor(x/13)+floor(y/17))%2)*0.55+0.1",
        expr3="0.37")
    if cropped:
        source = nuke.nodes.Crop(inputs=[source], reformat=False)
        source["box"].setValue([17, 11, width-9, height-15])
        source["intersect"].setValue(True)
        assert source.bbox().x() > 0 and source.bbox().y() > 0, str(source.bbox())
    mask = nuke.nodes.Expression(inputs=[base], expr0="0", expr1="0", expr2="0",
        expr3=f"x>73 && x<{width-91} && y>47 && y<{height-59}")
    node = fill.create_generative_fill_node()
    node.setInput(inpaint_node.SOURCE_INPUT, source)
    node.setInput(inpaint_node.MASK_INPUT, mask)
    node["edge_softness"].setValue(0)
    node["mask_grow"].setValue(0)
    node["fill_mask_overlap"].setValue(0)
    writer = node.node("KyvenInpaintSourceWrite")
    export = destination.parent / f"{label}-input.tif"
    writer["file"].setValue(export.as_posix())
    nuke.execute(writer, 1, 1)
    # The same exported pixels are the identity prediction. No hand-made patch.
    inpaint_node._set_patch(node, export)
    errors = []
    outside = []
    for y in range(7, height-7, max(1, height//31)):
        for x in range(7, width-7, max(1, width//37)):
            selected = mask.sample("rgba.alpha", x, y) > 0
            for channel in ("rgba.red", "rgba.green", "rgba.blue"):
                error = abs(source.sample(channel, x, y)-node.sample(channel, x, y))
                errors.append(error)
                if not selected:
                    outside.append(error)
            assert abs(node.sample("rgba.alpha", x, y)-source.sample("rgba.alpha", x, y)) < 1e-7
    # One display-encoded 8-bit step can exceed 0.004 in scene-linear RGB.
    # Spatial registration is checked independently on the rendered grid.
    assert max(errors) < 0.009, (label, max(errors))
    assert max(outside) == 0, (label, max(outside))
    for title, image in (("source", source), ("result", node)):
        output = nuke.nodes.Write(inputs=[image], file=(destination.parent / f"{label}-{title}.png").as_posix(), file_type="png")
        output["channels"].setValue("rgb")
        output["colorspace"].setValue("sRGB")
        nuke.execute(output, 1, 1)
    results.append({"case": label, "max_linear_rgb_error": max(errors), "outside_exact": True,
                    "source_alpha_exact": True, "samples": len(errors)})
destination.write_text(json.dumps(results, indent=2))
print(json.dumps(results), flush=True)
