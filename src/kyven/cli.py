"""Command-line adapter for the host-independent Kyven engine."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import tempfile
from pathlib import Path

from kyven.errors import ErrorCode, KyvenError
from kyven.inpaint.models import InpaintRequest
from kyven.inpaint.service import InpaintService
from kyven.models.catalog import ModelCatalog
from kyven.refine.models import RefineRequest
from kyven.refine.service import RefineService
from kyven.segment.models import (
    BoxPrompt,
    ExecutionProfile,
    PointLabel,
    PointPrompt,
    SegmentRequest,
)
from kyven.segment.service import SegmentService


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


def _model_ids(task: str | None = None) -> tuple[str, ...]:
    return tuple(spec.model_id for spec in ModelCatalog.builtin().list(task))


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models-dir", default=Path("models"), type=Path)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser."""

    parser = argparse.ArgumentParser(prog="kyven", description="Local AI tools for compositing")
    subparsers = parser.add_subparsers(dest="command", required=True)

    segment = subparsers.add_parser("segment", help="Create a matte from point or box prompts")
    segment.add_argument("--input", required=True, type=Path)
    segment.add_argument("--output", required=True, type=Path)
    segment.add_argument("--model", default="sam2.1-small", choices=_model_ids("segment"))
    _add_runtime_options(segment)
    segment.add_argument(
        "--profile",
        default=ExecutionProfile.BALANCED.value,
        choices=tuple(profile.value for profile in ExecutionProfile),
    )
    segment.add_argument("--point", action="append", default=[], type=_point)
    segment.add_argument("--box", type=_box)
    segment.add_argument("--single-mask", action="store_true")

    refine = subparsers.add_parser("refine", help="Refine a coarse mask or artist trimap")
    refine.add_argument("--input", required=True, type=Path)
    refine.add_argument("--mask", required=True, type=Path)
    refine.add_argument("--output", required=True, type=Path)
    refine.add_argument(
        "--model",
        default="vitmatte-small-composition-1k",
        choices=_model_ids("refine"),
    )
    _add_runtime_options(refine)
    refine.add_argument(
        "--profile",
        default=ExecutionProfile.BALANCED.value,
        choices=tuple(profile.value for profile in ExecutionProfile),
    )
    refine.add_argument("--manual-trimap", action="store_true")
    refine.add_argument("--foreground-radius", default=10, type=int)
    refine.add_argument("--background-radius", default=15, type=int)
    refine.add_argument("--roi", type=_box)
    refine.add_argument("--tile-size", default=0, type=int)
    refine.add_argument("--tile-overlap", default=64, type=int)

    inpaint = subparsers.add_parser("inpaint", help="Remove masked content from an image")
    inpaint.add_argument("--input", required=True, type=Path)
    inpaint.add_argument("--mask", required=True, type=Path)
    inpaint.add_argument("--output", required=True, type=Path)
    inpaint.add_argument("--model", default="lama-2025jan-onnx", choices=_model_ids("inpaint"))
    _add_runtime_options(inpaint)
    inpaint.add_argument("--profile", default="balanced", choices=tuple(p.value for p in ExecutionProfile))
    inpaint.add_argument("--crop-mode", default="auto", choices=("auto", "manual", "full"))
    inpaint.add_argument("--roi", type=_box)
    inpaint.add_argument("--context-padding", default=128, type=int)
    inpaint.add_argument("--mask-grow", default=12, type=int)
    inpaint.add_argument("--edge-color-match", default=1.0, type=float)
    inpaint.add_argument("--mask-threshold", default=0.5, type=float)
    inpaint.add_argument("--invert-mask", action="store_true")
    inpaint.add_argument("--no-mask-preprocess", action="store_true")
    inpaint.add_argument("--processed-mask-output", type=Path)
    inpaint.add_argument("--processing-size", default=0, type=int)

    serve = subparsers.add_parser("serve", help="Run the authenticated local inference server")
    _add_runtime_options(serve)
    serve.add_argument("--port", default=8765, type=int)
    serve.add_argument("--token-file", required=True, type=Path)

    models = subparsers.add_parser("models", help="Inspect or download trusted models")
    model_commands = models.add_subparsers(dest="model_command", required=True)
    list_models = model_commands.add_parser("list", help="List catalog models")
    list_models.add_argument("--models-dir", default=Path("models"), type=Path)
    download = model_commands.add_parser("download", help="Download and verify one model")
    download.add_argument("model_id", choices=_model_ids())
    download.add_argument("--models-dir", default=Path("models"), type=Path)
    return parser


def _load_or_create_token(path: Path) -> str:
    if path.is_file():
        token = path.read_text(encoding="utf-8").strip()
        if len(token) < 32:
            raise KyvenError(
                code=ErrorCode.AUTHENTICATION_FAILED,
                message="Existing Kyven token file contains an invalid token.",
                suggested_action="Delete the token file and restart the server.",
            )
        return token
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(token)
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return token


def _available_vram_mb() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(0).total_memory / (1024**2))
    except ImportError:
        pass
    return None


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    args = build_parser().parse_args(argv)
    catalog = ModelCatalog.builtin()
    try:
        if args.command == "segment":
            registry = catalog.registry(args.models_dir, args.device)
            request = SegmentRequest(
                source=args.input,
                output=args.output,
                points=tuple(args.point),
                box=args.box,
                provider_id=args.model,
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
        if args.command == "refine":
            registry = catalog.registry(args.models_dir, args.device)
            request = RefineRequest(
                source=args.input,
                mask=args.mask,
                output=args.output,
                provider_id=args.model,
                profile=ExecutionProfile(args.profile),
                roi=args.roi,
                generate_trimap=not args.manual_trimap,
                foreground_radius=args.foreground_radius,
                background_radius=args.background_radius,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
            )
            result = RefineService(registry).run(request)
            print(
                json.dumps(
                    {
                        "output": str(result.output),
                        "cache_key": result.cache_key,
                        "metadata": result.metadata,
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "serve":
            from kyven.segment.video import VideoSegmentService
            from kyven.server.app import KyvenServer, ServerConfig
            from kyven.server.jobs import JobManager

            models_dir = args.models_dir.resolve()
            token = _load_or_create_token(args.token_file.resolve())
            registry = catalog.registry(models_dir, args.device)
            config = ServerConfig(
                token=token,
                models_dir=models_dir,
                port=args.port,
                available_vram_mb=_available_vram_mb(),
            )
            server = KyvenServer(
                config,
                JobManager(
                    SegmentService(registry),
                    VideoSegmentService(registry),
                    RefineService(registry),
                    InpaintService(registry),
                ),
                registry,
                catalog,
            )
            print(json.dumps({"status": "ready", "host": "127.0.0.1", "port": server.port}))
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.close()
            return 0
        if args.command == "inpaint":
            registry = catalog.registry(args.models_dir, args.device)
            result = InpaintService(registry).run(InpaintRequest(
                source=args.input, mask=args.mask, output=args.output,
                mask_output=args.processed_mask_output,
                provider_id=args.model, profile=ExecutionProfile(args.profile),
                crop_mode=args.crop_mode, roi=args.roi, context_padding=args.context_padding,
                mask_grow=args.mask_grow,
                edge_color_match=args.edge_color_match,
                mask_threshold=args.mask_threshold, invert_mask=args.invert_mask,
                preprocess_mask=not args.no_mask_preprocess,
                processing_size=args.processing_size,
            ))
            print(json.dumps({"output": str(result.output), "mask_output": str(result.mask_output) if result.mask_output else None, "cache_key": result.cache_key, "metadata": result.metadata}, indent=2))
            return 0
        if args.command == "models" and args.model_command == "list":
            print(
                json.dumps(
                    {
                        "models": [
                            spec.snapshot(args.models_dir, _available_vram_mb())
                            for spec in catalog.list()
                        ]
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "models" and args.model_command == "download":
            path = catalog.download(args.model_id, args.models_dir)
            print(json.dumps({"model_id": args.model_id, "path": str(path)}, indent=2))
            return 0
    except KyvenError as exc:
        print(json.dumps(exc.to_dict(), indent=2), file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
