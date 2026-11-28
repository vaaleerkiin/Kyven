"""Publish real inference images and labelled comparison sheets for the manuals.

Run capture_node_examples.py and capture_waft_example.py first. This script only
lays out their real pixels; it does not synthesize or retouch model results.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from kyven.dlss.images import read_channels

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".runtime/documentation-real"
RAW = ROOT / "docs/images/examples"
BOARDS = ROOT / "docs/images/nodes"
BG, FG, MUTED, CYAN = "#101a29", "#edf2f8", "#aab8ca", "#77ddd6"


def font(size):
    return ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", size)


def load(name):
    return Image.open(WORK / name).convert("RGB")


def board(name, title, settings, panels, note):
    width, height = 1600, 650
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 16), title, font=font(30), fill=FG)
    draw.text((24, 61), settings, font=font(19), fill=CYAN)
    for index, (label, image, caption) in enumerate(panels):
        x = 24 + index*526
        draw.rounded_rectangle((x, 105, x+500, 558), radius=10, fill="#1c293b")
        draw.text((x+12, 117), label, font=font(20), fill=CYAN)
        preview = image.copy()
        preview.thumbnail((476, 350), Image.Resampling.LANCZOS)
        canvas.paste(preview, (x+12+(476-preview.width)//2, 157+(350-preview.height)//2))
        draw.text((x+12, 524), caption, font=font(16), fill=MUTED)
    draw.text((24, 583), note, font=font(18), fill=FG)
    draw.text((24, 619), "REAL INFERENCE  |  Local photographic plate, 1024 x 540  |  Raw images and settings linked in the guide",
              font=font(17), fill=MUTED)
    canvas.save(BOARDS / f"{name}.png", optimize=True)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    BOARDS.mkdir(parents=True, exist_ok=True)
    mapping = {"source.png": "source.png", "segment.png": "segment-alpha.png",
               "edge-guide.png": "segment-edge-guide.png", "trimap.png": "refine-trimap.png",
               "refine.png": "refine-alpha.png", "removal-mask.png": "removal-mask.png",
               "inpaint.png": "inpaint-result.png", "fill.png": "generative-fill-result.png",
               "dlss.png": "dlss-result-srgb.png", "waft-frame-2.png": "waft-frame-2.png",
               "waft-frame-3.png": "waft-frame-3.png"}
    report = json.loads((WORK / "report.json").read_text())
    report["waft"] = json.loads((WORK / "waft-report.json").read_text())
    source = load("source.png")
    source_pixels = np.asarray(source)
    alpha = np.asarray(Image.open(WORK / "segment.png").convert("L"))
    points = source.copy()
    drawing = ImageDraw.Draw(points)
    for x, y in ((595, 200), (605, 325), (605, 495), (526, 395), (686, 391)):
        drawing.ellipse((x-8, y-8, x+8, y+8), fill="#103326", outline="#8dffbb", width=2)
        drawing.line((x-4, y, x+4, y), fill="#8dffbb", width=2)
        drawing.line((x, y-4, x, y+4), fill="#8dffbb", width=2)
    yy, xx = np.indices(alpha.shape)
    checker = np.where(((xx//20+yy//20) % 2)[..., None], 58, 88)
    cutout = np.uint8(source_pixels*(alpha[..., None]/255) + checker*(1-alpha[..., None]/255))
    Image.fromarray(cutout).save(RAW / "segment-cutout.png")
    board("segment", "SEGMENT / Actual SAM 2 result", "SAM 2.1 Small | Five positive points | Full frame | Fill Holes off",
          [("01  SOURCE + POINTS", points, "Point overlay marks the exact model prompts"),
           ("02  ALPHA", load("segment.png"), "Unretouched model matte"),
           ("03  CUTOUT", Image.fromarray(cutout), "Source multiplied by the real alpha")],
          "RGB alone stays unchanged in Source + Alpha. View Alpha or Cutout to inspect the segmentation.")
    crop = (515, 145, 710, 390)
    board("refine", "REFINE / Actual ViTMatte edge detail", "ViTMatte Small | Erode 5 / Dilate 8 px | Tiles 512 / overlap 64 | Same detail crop in all panels",
          [("01  SOURCE DETAIL", source.crop(crop), "Crop: x 515-710, y 145-390 (top-left origin)"),
           ("02  MODEL TRIMAP", load("trimap.png").crop(crop), "Actual black / gray / white guidance"),
           ("03  REFINED ALPHA", load("refine.png").crop(crop), "Actual soft alpha; no edge retouch")],
          "The source hair and sleeve edges are solved from RGB plus trimap. The full alpha is linked below.")
    removal = np.asarray(Image.open(WORK / "removal-mask.png").convert("L"))
    board("inpaint", "INPAINT / Actual Big-LaMa removal", "Big-LaMa Native | Standard | Auto crop + 128 px context | Edge softness 0",
          [("01  SOURCE", source, "Original photographic plate"),
           ("02  REMOVAL ALPHA", load("removal-mask.png"), "SAM silhouette grown by 14 source pixels"),
           ("03  RESULT", load("inpaint.png"), "Actual reconstruction, including its artifacts")],
          "Background geometry is estimated. Inspect invented lines and texture before accepting the clean-up.")
    board("generative-fill", "GENERATIVE FILL / Actual Mage removal", "Mage-Flow Edit Turbo Q6 | Seed 42 | 4 steps | Processing Resolution 512 | Color Match Auto",
          [("01  SOURCE", source, "Same original plate as the Inpaint example"),
           ("02  EDIT PERMISSION", load("removal-mask.png"), "Result Mask Grow 0; original mask bounds apply"),
           ("03  RESULT", load("fill.png"), "Actual generated background, without retouch")],
          "Prompt: Remove the man. Restore the storefront and pavement behind him. Preserve the white car and lighting.")
    dlss = load("dlss.png")
    diff = Image.fromarray(np.uint8(np.clip(np.abs(np.asarray(dlss).astype(float)-source_pixels)*8, 0, 255)))
    crop = (490, 135, 730, 405)
    board("dlss", "DLSS / Actual RTX reconstruction detail", "Temporal off | Processing Size 1 | Intensity 1 | Preserve Source Color on | Display: linear EXR to sRGB",
          [("01  SOURCE DETAIL", source.crop(crop), "Crop: x 490-730, y 135-405 (top-left origin)"),
           ("02  RESULT DETAIL", dlss.crop(crop), "Actual DLSS output; same crop and display"),
           ("03  ABS DIFFERENCE x8", diff.crop(crop), "Absolute display-RGB difference, amplified 8x")],
          "This is one still-frame example. Temporal stability is not evaluated by this comparison.")
    channels = read_channels(WORK / "waft.exr")
    u, v = channels["forward.u"], channels["forward.v"]
    magnitude = np.hypot(u, v)
    hue = (np.arctan2(v, u)/(2*math.pi)+1) % 1
    hsv = np.stack((hue*255, np.ones_like(u)*255, np.clip(magnitude/16, 0, 1)*255), -1).astype(np.uint8)
    flow = Image.fromarray(hsv, mode="HSV").convert("RGB")
    flow.save(RAW / "waft-forward-flow.png")
    np.savez_compressed(RAW / "waft-forward-vectors.npz", u=u, v=v)
    central = (slice(32, -32), slice(32, -32))
    measured = {"median_forward_u": float(np.median(u[central])),
                "median_forward_v": float(np.median(v[central])),
                "display": "HSV hue=atan2(v,u); value=clip(magnitude/16,0,1); saturation=1"}
    report["waft"].update(measured)
    board("waft", "WAFT / Actual native Nuke Inference", "WAFT-a1 FP16 v5 | Working side 512 | Grid Suppression 1 | Controlled +8 px/frame pan on a real photo",
          [("01  FRAME 2", load("waft-frame-2.png"), "Native Nuke Transform: zero offset"),
           ("02  FRAME 3", load("waft-frame-3.png"), "Native Nuke Transform: +8 px horizontal"),
           ("03  ACTUAL FORWARD FLOW", flow, "Hue = direction; brightness = magnitude / 16 px")],
          f"Measured interior median: forward.u {measured['median_forward_u']:.2f} px, forward.v {measured['median_forward_v']:.2f} px. This is a controlled pan, not recorded motion.")
    checks = {}
    for key, filename in (("inpaint", "inpaint.png"), ("fill", "fill.png")):
        output = np.asarray(load(filename))
        checks[key] = {"outside_mask_exact": bool(np.array_equal(output[removal == 0], source_pixels[removal == 0])),
                       "changed_selected_pixels": int(np.count_nonzero(np.any(output != source_pixels, -1) & (removal > 0)))}
        assert checks[key]["outside_mask_exact"] and checks[key]["changed_selected_pixels"] > 0
    refined = np.asarray(Image.open(WORK / "refine.png").convert("L"))
    checks["refine"] = {"partial_alpha_pixels": int(np.count_nonzero((refined > 0) & (refined < 255)))}
    assert checks["refine"]["partial_alpha_pixels"] > 0
    report["checks"] = checks
    report["artifacts"] = {}
    for original, destination in mapping.items():
        shutil.copyfile(WORK / original, RAW / destination)
        report["artifacts"][destination] = hashlib.sha256((RAW / destination).read_bytes()).hexdigest()

    def portable(value):
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [portable(item) for item in value]
        if isinstance(value, str):
            return value.replace(str(ROOT), "<Kyven>").replace(ROOT.as_posix(), "<Kyven>")
        return value

    (RAW / "capture.json").write_text(json.dumps(portable(report), indent=2), encoding="utf-8")
    print(json.dumps({"checks": checks, "waft": measured, "boards": 6}, indent=2))


if __name__ == "__main__":
    main()
