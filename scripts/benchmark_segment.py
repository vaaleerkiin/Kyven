"""Measure cold/warm SAM 2 segmentation time and peak CUDA memory."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from kyven.models.catalog import ModelCatalog
from kyven.segment.models import PointPrompt, SegmentRequest
from kyven.segment.service import SegmentService


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", default="sam2.1-small")
    parser.add_argument("--models-dir", default=Path("models"), type=Path)
    parser.add_argument("--point", required=True, help="X,Y")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    x, y = (float(item) for item in args.point.split(","))
    registry = ModelCatalog.builtin().registry(args.models_dir, "cuda")
    service = SegmentService(registry)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    torch.cuda.reset_peak_memory_stats()
    for index in range(2):
        request = SegmentRequest(
            source=args.input,
            output=args.output_dir / f"benchmark_{index}.png",
            points=(PointPrompt(x, y),),
            provider_id=args.model,
        )
        start = time.perf_counter()
        result = service.run(request)
        torch.cuda.synchronize()
        runs.append(
            {
                "kind": "cold" if index == 0 else "warm",
                "seconds": time.perf_counter() - start,
                "score": result.score,
            }
        )
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "model": args.model,
                "runs": runs,
                "peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
                "peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
