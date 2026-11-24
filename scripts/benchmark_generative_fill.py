"""Qualify one local source/mask pair through the real Kyven provider."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kyven.cancellation import CancellationToken
from kyven.generative_fill.models import (
    EnhancePromptRequest,
    FillSettings,
    GenerativeFillRequest,
)
from kyven.inpaint.service import InpaintService
from kyven.models.catalog import ModelCatalog


class Progress(CancellationToken):
    def report_progress(self, progress: float, message: str) -> None:
        super().report_progress(progress, message)
        print(f"{progress:.0%}: {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--models-dir", default=Path("models"), type=Path)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--enhance", action="store_true")
    parser.add_argument("--working-size", default=768, type=int)
    parser.add_argument("--context-padding", default=256, type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    registry = ModelCatalog.builtin().registry(args.models_dir, args.device)
    service = InpaintService(registry)
    report = {"prompt": args.prompt}
    try:
        prompt = args.prompt
        if args.enhance:
            started = time.monotonic()
            enhanced = service.enhance_prompt(EnhancePromptRequest(prompt=prompt,
                max_new_tokens=128), Progress())
            prompt = enhanced["prompt"]
            report["enhancement"] = enhanced
            report["enhance_seconds"] = time.monotonic() - started
            (args.output_dir / "prompt-enhancement.json").write_text(
                json.dumps(enhanced, indent=2, ensure_ascii=False), encoding="utf-8")
            print("Improved prompt:", json.dumps(prompt, ensure_ascii=True), flush=True)
        request = GenerativeFillRequest(
            args.source.resolve(), args.mask.resolve(), (args.output_dir / "result.png").resolve(),
            mask_output=(args.output_dir / "merge-mask.png").resolve(),
            patch_output=(args.output_dir / "patch.png").resolve(),
            context_padding=args.context_padding, mask_grow=8, edge_softness=4,
            fill=FillSettings(prompt=prompt, working_size=args.working_size),
        )
        started = time.monotonic()
        result = service.run(request, Progress())
        report["generation_seconds"] = time.monotonic() - started
        report["metadata"] = result.metadata
        report["settings"] = asdict(request.fill)
        source = np.asarray(Image.open(request.source).convert("RGB"))
        mask = np.asarray(Image.open(request.mask).convert("L"))
        output = np.asarray(Image.open(result.output).convert("RGB"))
        report["outside_mask_exact"] = bool(np.array_equal(source[mask == 0], output[mask == 0]))
        report["selected_pixels_changed"] = int(np.count_nonzero(np.any(source != output, axis=-1) & (mask > 0)))
        assert report["outside_mask_exact"]
        assert report["selected_pixels_changed"] > 0
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        registry.unload_all()


if __name__ == "__main__":
    main()
