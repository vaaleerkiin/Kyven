"""Real same-frame comparison; outputs are local QA artifacts, never test fixtures."""
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from kyven.cancellation import CancellationToken
from kyven.generative_fill.assets import bundle_ready
from kyven.generative_fill.models import (
    MOSS_PROVIDER_ID,
    PROVIDER_ID,
    FillSettings,
    GenerativeFillRequest,
)
from kyven.generative_fill.moss_provider import MossLanPaintProvider
from kyven.generative_fill.provider import KleinLanPaintProvider
from kyven.inpaint.service import InpaintService
from kyven.segment.providers.registry import ProviderRegistry

parser = argparse.ArgumentParser()
parser.add_argument("--await-models", action="store_true")
parser.add_argument("--size", type=int, default=768)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--thinking-steps", type=int, default=2)
parser.add_argument("--output", type=Path, default=ROOT / ".runtime/moss-research/qualification")
args = parser.parse_args()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
source = ROOT / ".runtime/generative-fill/removal-source.png"
mask = ROOT / ".runtime/generative-fill/removal-mask.png"
prompt = "Remove the man. Reconstruct the storefront, road and sidewalk that he obscures. Match the original lighting, black levels, color, contrast, grain and depth of field. Keep the car and all other existing objects unchanged."
report_path = output / "report.json"
report = json.loads(report_path.read_text()) if report_path.exists() else {}


class LoggingToken(CancellationToken):
    def report_progress(self, progress, message):
        previous = self.progress_snapshot()[1]
        super().report_progress(progress, message)
        if message != previous:
            print(f"{progress:.0%} {message}", flush=True)


def run(model_id, provider, steps):
    directory = output / model_id
    directory.mkdir(exist_ok=True)
    original_predict = provider.predict
    def record(request, token):
        result = original_predict(request, token)
        Image.fromarray(result.rgb).save(directory/"raw-before-color-and-composite.png")
        Image.open(request.source).save(directory/"source-crop.png")
        Image.open(request.mask).save(directory/"inference-mask.png")
        return result
    provider.predict = record
    registry = ProviderRegistry()
    registry.register(model_id, lambda: provider)
    request = GenerativeFillRequest(source, mask, directory/"result.png", provider_id=model_id,
        patch_output=directory/"patch.png", mask_output=directory/"merge-mask.png",
        mask_grow=0, context_padding=256, edge_softness=0,
        fill=FillSettings(prompt=prompt, steps=steps, working_size=args.size,
                          thinking_steps=args.thinking_steps if model_id == MOSS_PROVIDER_ID else 2,
                          timeout_seconds=3600))
    print("Running", model_id, flush=True)
    started = time.monotonic()
    try:
        result = InpaintService(registry).run(request, cancellation=LoggingToken())
    finally:
        provider.unload()
    image, before = np.asarray(Image.open(result.output)), np.asarray(Image.open(source))
    selected = np.asarray(Image.open(mask).convert("L")) > 0
    assert np.array_equal(image[~selected], before[~selected])
    inside_boundary = selected & ~(np.asarray(Image.fromarray(selected.astype(np.uint8)*255).filter(ImageFilter.MinFilter(5))) > 0)
    blur = np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(2))).astype(float)
    residual = np.abs(image.astype(float)-blur)[inside_boundary]
    data = {"seconds": round(time.monotonic()-started, 2), "outside_mask_max_error": 0,
        "inside_boundary_highpass_median": float(np.median(residual)),
        "metadata": result.metadata, "settings": asdict(request.fill)}
    (directory/"report.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    report[model_id] = data
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(model_id, json.dumps(data), flush=True)


if PROVIDER_ID not in report:
    run(PROVIDER_ID, KleinLanPaintProvider(ROOT/"models"), 4)
deadline = time.monotonic()+10800
while not bundle_ready(ROOT/"models"/MOSS_PROVIDER_ID, model_id=MOSS_PROVIDER_ID):
    if not args.await_models or time.monotonic() > deadline:
        raise RuntimeError("Verified MOSS model download is not complete")
    time.sleep(5)
if MOSS_PROVIDER_ID not in report:
    run(MOSS_PROVIDER_ID, MossLanPaintProvider(ROOT/"models"), args.steps)
images = [Image.open(source), Image.open(output/PROVIDER_ID/"result.png"),
          Image.open(output/MOSS_PROVIDER_ID/"result.png")]
labels = ["Original", "FLUX.2 Klein Q4", "MOSS SFT INT8 + Edit"]
canvas = Image.new("RGB", (1024*3, 570), "#222222")
draw = ImageDraw.Draw(canvas)
for i, (image, label) in enumerate(zip(images, labels)):
    canvas.paste(image.resize((1024,540)), (i*1024,30))
    draw.text((i*1024+12,8),label,fill="white")
canvas.save(output/"comparison.png")
print("Comparison saved:",output/"comparison.png",flush=True)
