"""Render deployed WAFT on a photographic plate with a controlled 8 px/frame pan.

    Nuke16.0.exe --nukex -i -t --gpu scripts/capture_waft_example.py

Uses the source prepared by capture_node_examples.py. This is a real Inference
render with controlled motion, not an independently photographed video sequence.
"""

import json
import time
from pathlib import Path

import nuke

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".runtime/documentation-real"
nuke.pluginAddPath((ROOT / "plugins/Cattery/WAFT").as_posix())
nuke.root()["first_frame"].setValue(1)
nuke.root()["last_frame"].setValue(3)
source = nuke.nodes.Read(file=(WORK / "source.png").as_posix())
source["raw"].setValue(True)
pan = nuke.nodes.Transform(inputs=[source])
pan["translate"].setExpression("(frame-2)*8", 0)
node = nuke.createNode("WAFT", inpanel=False)
node.setInput(0, pan)
node["in_colorspace"].setValue("sRGB")
node["processing"].setValue(0)
node["precision"].setValue(0)
node["useGPUIfAvailable"].setValue(True)
node["output_mode"].setValue(0)
write = nuke.nodes.Write(inputs=[node], file=(WORK / "waft.exr").as_posix())
write["file_type"].setValue("exr")
write["channels"].setValue("all")
write["datatype"].setValue("32 bit float")
started = time.monotonic()
nuke.execute(write, 2, 2)
for frame in (2, 3):
    preview = nuke.nodes.Write(inputs=[pan], file=(WORK / f"waft-frame-{frame}.png").as_posix())
    preview["file_type"].setValue("png")
    preview["channels"].setValue("rgb")
    preview["raw"].setValue(True)
    nuke.execute(preview, frame, frame)
    nuke.delete(preview)
report = {"nuke": nuke.NUKE_VERSION_STRING, "seconds": round(time.monotonic()-started, 3),
          "model": "WAFT-a1 FP16 / deployed v5 / 5 iterations", "working_side": 512,
          "motion": "Controlled horizontal translation of the real plate: +8 source pixels/frame",
          "frame": 2, "channels": node.channels(), "grid_suppression": 1,
          "edge_threshold": 0.25}
(WORK / "waft-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report), flush=True)
