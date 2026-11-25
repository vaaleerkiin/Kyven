"""Local full-model qualification; artist images and weights are not distributed."""
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, ImageDraw

from kyven.cancellation import CancellationToken
from kyven.generative_fill.models import ZIMAGE_PROVIDER_ID, FillSettings, GenerativeFillRequest
from kyven.generative_fill.zimage_provider import ZImageInpaintProvider
from kyven.inpaint.service import InpaintService
from kyven.segment.providers.registry import ProviderRegistry

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, default=ROOT/".runtime/generative-fill/removal-source.png")
parser.add_argument("--mask", type=Path, default=ROOT/".runtime/generative-fill/removal-mask.png")
parser.add_argument("--output", type=Path, default=ROOT/".runtime/zimage-research/qualification")
parser.add_argument("--steps", type=int, default=28)
parser.add_argument("--size", type=int, default=768)
parser.add_argument("--guidance", type=float, default=3.0)
parser.add_argument("--control-scale", type=float, default=0.9)
parser.add_argument("--prompt", default="Remove the man. Reconstruct the storefront, road and sidewalk that he obscures. Match the original lighting, black levels, color, contrast, grain and depth of field. Keep the car and all other existing objects unchanged.")
args = parser.parse_args()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)


class LoggingToken(CancellationToken):
    def report_progress(self, progress, message):
        previous = self.progress_snapshot()[1]
        super().report_progress(progress, message)
        if previous != message:
            print(f"{progress:.0%} {message}", flush=True)


provider = ZImageInpaintProvider(ROOT/"models")
original_predict = provider.predict


def record(request, token):
    result = original_predict(request, token)
    Image.fromarray(result.rgb).save(output/"raw-before-color-and-composite.png")
    Image.open(request.source).save(output/"source-crop.png")
    Image.open(request.mask).save(output/"inference-mask.png")
    return result


provider.predict = record
registry = ProviderRegistry()
registry.register(ZIMAGE_PROVIDER_ID, lambda: provider)
request = GenerativeFillRequest(args.source, args.mask, output/"result.png", provider_id=ZIMAGE_PROVIDER_ID,
    patch_output=output/"patch.png", mask_output=output/"merge-mask.png",
    mask_grow=0, context_padding=256, edge_softness=0,
    fill=FillSettings(prompt=args.prompt, steps=args.steps, working_size=args.size,
                      zimage_guidance=args.guidance, zimage_control_scale=args.control_scale,
                      mask_overlap=64, timeout_seconds=3600))
start = time.monotonic()
try:
    result = InpaintService(registry).run(request, cancellation=LoggingToken())
finally:
    provider.unload()
before, image = np.asarray(Image.open(args.source)), np.asarray(Image.open(result.output))
selected = np.asarray(Image.open(args.mask).convert("L")) > 0
assert np.array_equal(image[~selected], before[~selected])
data = {"seconds": round(time.monotonic()-start, 2), "outside_mask_max_error": 0,
        "metadata": result.metadata, "settings": asdict(request.fill)}
(output/"report.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
images = [Image.open(args.source)]
labels = ["Original"]
baseline = ROOT/".runtime/moss-research/qualification/flux2-klein-4b-q4-lanpaint/result.png"
if (baseline.is_file() and result.metadata.get("edit_operation") == "remove"
        and args.mask.resolve() == (ROOT/".runtime/generative-fill/removal-mask.png").resolve()):
    images.append(Image.open(baseline))
    labels.append("FLUX.2 Klein Q4")
images.append(Image.open(result.output))
labels.append("Z-Image Base Q4 / Native Inpaint")
canvas = Image.new("RGB", (1024*len(images), 570), "#222222")
draw = ImageDraw.Draw(canvas)
for i, (picture, label) in enumerate(zip(images, labels)):
    canvas.paste(picture.resize((1024, 540)), (i*1024, 30))
    draw.text((i*1024+12, 8), label, fill="white")
canvas.save(output/"comparison.png")
print(json.dumps(data), flush=True)
