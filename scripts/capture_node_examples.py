"""Run real Kyven providers on a photographic plate for the documentation.

    python scripts/capture_node_examples.py --source path/to/plate.png

Outputs and service caches stay in .runtime/documentation-real. The companion
Nuke script renders native WAFT and the gallery builder publishes selected images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kyven.cancellation import CancellationToken
from kyven.dlss.images import read_channels, write_exr
from kyven.dlss.models import DlssRequest
from kyven.dlss.provider import ExternalDlssProvider
from kyven.dlss.service import DlssService
from kyven.generative_fill.models import FillSettings, GenerativeFillRequest
from kyven.inpaint.models import InpaintRequest
from kyven.inpaint.service import InpaintService
from kyven.models.catalog import ModelCatalog
from kyven.refine.models import RefineRequest
from kyven.refine.service import RefineService
from kyven.segment.models import PointPrompt, SegmentRequest
from kyven.segment.service import SegmentService

WORK = ROOT / ".runtime/documentation-real"


class Progress(CancellationToken):
    def report_progress(self, progress: float, message: str) -> None:
        super().report_progress(progress, message)
        print(f"{progress:.0%} {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "docs/images/examples/source.png")
    parser.add_argument("--only", choices=["segment", "refine", "inpaint", "fill", "dlss"])
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    source = WORK / "source.png"
    image = Image.open(args.source).convert("RGB")
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    image.save(source)
    registry = ModelCatalog.builtin().registry(ROOT / "models", "cuda")
    registry.register("dlss-external", ExternalDlssProvider)
    report_path = WORK / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report["source"] = {"size": image.size,
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "origin": "Existing local street qualification plate"}

    def run(name, service, request):
        if args.only and args.only != name:
            return
        print(f"START {name}", flush=True)
        started = time.monotonic()
        try:
            result = service.run(request, cancellation=Progress())
            metadata = result.get("metadata", {}) if isinstance(result, dict) else result.metadata
            report[name] = {"seconds": round(time.monotonic() - started, 3),
                            "request": asdict(request), "metadata": metadata}
            report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            print(f"DONE {name}", flush=True)
        finally:
            registry.unload_all()

    try:
        run("segment", SegmentService(registry), SegmentRequest(
            source, WORK / "segment.png", provider_id="sam2.1-small",
            points=(PointPrompt(595, 200), PointPrompt(605, 325), PointPrompt(605, 495),
                    PointPrompt(526, 395), PointPrompt(686, 391)),
            trimap_output=WORK / "edge-guide.png"))
        run("refine", RefineService(registry), RefineRequest(
            source, WORK / "segment.png", WORK / "refine.png",
            trimap_output=WORK / "trimap.png", tile_size=512, tile_overlap=64,
            foreground_radius=5, background_radius=8))
        if (WORK / "segment.png").exists():
            mask = Image.open(WORK / "segment.png").convert("L")
            mask.filter(ImageFilter.MaxFilter(29)).save(WORK / "removal-mask.png")
        run("inpaint", InpaintService(registry), InpaintRequest(
            source, WORK / "removal-mask.png", WORK / "inpaint.png",
            provider_id="big-lama-native", crop_mode="auto", context_padding=128,
            mask_grow=0, edge_softness=0))
        run("fill", InpaintService(registry), GenerativeFillRequest(
            source, WORK / "removal-mask.png", WORK / "fill.png",
            provider_id="mage-flow-edit-turbo-q6", context_padding=256,
            edge_softness=4, mask_grow=0,
            fill=FillSettings(prompt="Remove the man. Restore the storefront and pavement "
                              "behind him. Preserve the white car and the original lighting.",
                              seed=42, working_size=512, seamless_blend="off", seam_repair="off")))
        rgb = np.asarray(image).astype(np.float32) / 255
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055)**2.4)
        write_exr(WORK / "source.exr", np.concatenate(
            (linear, np.ones((*linear.shape[:2], 1), dtype=np.float32)), axis=-1))
        run("dlss", DlssService(registry), DlssRequest(
            WORK / "source.exr", WORK / "dlss.exr",
            ROOT / ".runtime/dlss/bin/KyvenDLSSWorker.exe", session="documentation-real",
            max_dimension=1024, temporal=False, intensity=1.0, preserve_source_color=True))
        if (WORK / "dlss.exr").exists():
            channels = read_channels(WORK / "dlss.exr")
            linear_result = np.stack([channels[c] for c in ("R", "G", "B")], axis=-1)
            display = np.where(linear_result <= 0.0031308, linear_result * 12.92,
                               1.055 * np.maximum(linear_result, 0)**(1/2.4) - 0.055)
            Image.fromarray(np.uint8(np.clip(display, 0, 1)*255 + 0.5)).save(WORK / "dlss.png")
    finally:
        registry.unload_all()


if __name__ == "__main__":
    main()
