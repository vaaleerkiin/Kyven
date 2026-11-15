"""Opt-in RTX benchmark with analytically known external translation vectors.

The vectors are test ground truth, not an optical-flow implementation. No media
or model is downloaded. Compare native NR history against the finite residual buffer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kyven.dlss.images import read_channels, resize, write_exr
from kyven.dlss.models import DlssRequest
from kyven.dlss.provider import ExternalDlssProvider
from kyven.dlss.service import DlssService
from kyven.segment.providers.registry import ProviderRegistry


def qualify(worker: Path, output: Path, source: Path | None, frames: int = 8) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if source is not None:
        channels = read_channels(source)
        rgb = resize(np.stack([channels[c] for c in ("R", "G", "B")], -1), (512, 288))
    else:
        y, x = np.mgrid[:288, :512].astype(np.float32)
        detail = 0.08 * np.sin(x / 2) * np.cos(y / 3)
        noise = np.random.default_rng(7).normal(0, 0.012, x.shape).astype(np.float32)
        rgb = np.stack((0.1 + x / 850, 0.12 + y / 700, 0.2 + x / 1300), -1)
        rgb = np.maximum(rgb + detail[..., None] + noise[..., None], 0)
    motion = np.zeros((288, 512, 4), np.float32)
    motion[..., 0], motion[..., 2] = -1, 1  # R/G backward, B/A forward.
    motion_path = output / "external-ground-truth.exr"
    write_exr(motion_path, motion)
    sources = []
    for frame in range(frames):
        rgba = np.concatenate((np.roll(rgb, frame, axis=1),
                               np.full((288, 512, 1), 0.7, np.float32)), -1)
        path = output / f"source.{frame:04d}.exr"
        write_exr(path, rgba)
        sources.append((path, rgba))
    registry = ProviderRegistry()
    registry.register("dlss-external", ExternalDlssProvider)
    report = {}
    try:
        for name, strength, count in (("native_only", 0.0, 1), ("buffered", 0.75, 3)):
            service = DlssService(registry)
            previous = None
            errors, stats = [], []
            for frame, (path, original) in enumerate(sources):
                result = service.run(DlssRequest(
                    path, output / f"{name}.{frame:04d}.exr", worker, motion=motion_path,
                    frame=frame, session=name, temporal=True, reset=frame == 0,
                    temporal_strength=strength, history_frames=count, history_budget_mb=64,
                    motion_consistency=True, forward_x="B", forward_y="A", max_dimension=512))
                channels = read_channels(Path(result["output"]))
                np.testing.assert_array_equal(channels["A"], original[..., 3])
                delta = np.stack([channels[c] for c in ("R", "G", "B")], -1) - original[..., :3]
                assert np.isfinite(delta).all()
                if previous is not None and frame >= 3:
                    error = delta[:, 16:-16] - previous[:, 15:-17]
                    errors.append(float(np.sqrt(np.mean(error**2))))
                previous = delta
                stats.append({key: result["metadata"][key] for key in (
                    "history_reason", "history_frames_buffered", "history_cpu_bytes",
                    "temporal_samples_used", "temporal_reused_fraction")})
            report[name] = {"motion_aligned_residual_rms": float(np.mean(errors)), "frames": stats}
            print(name, report[name]["motion_aligned_residual_rms"], flush=True)
        baseline = report["native_only"]["motion_aligned_residual_rms"]
        report["rms_reduction_fraction"] = (
            1 - report["buffered"]["motion_aligned_residual_rms"] / baseline if baseline else 0)
        assert report["buffered"]["frames"][-1]["temporal_samples_used"] == 3
        assert all(item["history_reason"] == "continuous" for item in report["buffered"]["frames"][1:])
    finally:
        registry.unload_all()
    (output / "qualification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    print(json.dumps(qualify(args.worker.resolve(), args.output.resolve(), args.source), indent=2))
