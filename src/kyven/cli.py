"""Command-line adapter for the host-independent Kyven engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kyven.errors import KyvenError
from kyven.segment.models import BoxPrompt, ExecutionProfile, PointLabel, PointPrompt, SegmentRequest
from kyven.segment.providers.registry import default_registry
from kyven.segment.service import SegmentService

DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"


def _point(value: str) -> PointPrompt:
    try:
        x, y, label = value.split(",")
        return PointPrompt(float(x), float(y), PointLabel(label.lower()))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "Point must be X,Y,positive or X,Y,negative."
        ) from exc


def _box(value: str) -> BoxPrompt:
    try:
        x0, y0, x1, y1 = (float(item) for item in value.split(","))
        return BoxPrompt(x0, y0, x1, y1)
    except (ValueError, TypeError, KyvenError) as exc:
        raise argparse.ArgumentTypeError("Box must be X0,Y0,X1,Y1.") from exc


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser."""

    parser = argparse.ArgumentParser(prog="kyven", description="Local AI masking engine")
    subparsers = parser.add_subparsers(dest="command", required=True)
    segment = subparsers.add_parser("segment", help="Create a matte from point or box prompts")
    segment.add_argument("--input", required=True, type=Path)
    segment.add_argument("--output", required=True, type=Path)
    segment.add_argument("--checkpoint", required=True)
    segment.add_argument("--model-config", default=DEFAULT_SAM2_CONFIG)
    segment.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    segment.add_argument(
        "--profile",
        default=ExecutionProfile.BALANCED.value,
        choices=tuple(profile.value for profile in ExecutionProfile),
    )
    segment.add_argument("--point", action="append", default=[], type=_point)
    segment.add_argument("--box", type=_box)
    segment.add_argument("--single-mask", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    args = build_parser().parse_args(argv)
    try:
        if args.command == "segment":
            registry = default_registry(
                checkpoint=args.checkpoint,
                model_config=args.model_config,
                device=args.device,
            )
            request = SegmentRequest(
                source=args.input,
                output=args.output,
                points=tuple(args.point),
                box=args.box,
                profile=ExecutionProfile(args.profile),
                multimask_output=not args.single_mask,
            )
            result = SegmentService(registry).run(request)
            print(
                json.dumps(
                    {
                        "output": str(result.output),
                        "score": result.score,
                        "cache_key": result.cache_key,
                        "metadata": result.metadata,
                    },
                    indent=2,
                )
            )
            return 0
    except KyvenError as exc:
        print(json.dumps(exc.to_dict(), indent=2), file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

