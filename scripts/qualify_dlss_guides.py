"""Opt-in Windows/RTX qualification of native control guides and HDR restoration.

Uses a synthetic scene by default. --source accepts a local full-frame EXR; no
media is downloaded or uploaded. Saves EXRs and a machine-readable report.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from kyven.cancellation import CancellationToken
from kyven.dlss.color import prepare
from kyven.dlss.images import channel, read_channels, resize, write_exr
from kyven.dlss.models import DlssRequest
from kyven.dlss.provider import ExternalDlssProvider
from kyven.dlss.service import DlssService
from kyven.segment.providers.registry import ProviderRegistry


def qualify(worker: Path, output: Path, source: Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if source is not None:
        channels = read_channels(source)
        rgb = resize(np.stack([channel(channels, c) for c in ("R", "G", "B")], -1), (512, 288))
    else:
        y, x = np.mgrid[:288, :512].astype(np.float32)
        texture = 0.08 * np.sin(x / 2) * np.cos(y / 3)
        noise = np.random.default_rng(7).normal(0, 0.012, x.shape).astype(np.float32)
        rgb = np.stack((0.1 + x / 850, 0.12 + y / 700, 0.2 + x / 1300), -1)
        rgb = np.clip(rgb + texture[..., None] + noise[..., None], 0, None)
    bounded, _ = prepare(rgb)
    rgba = np.concatenate((bounded, np.ones((*bounded.shape[:2], 1), np.float32)), -1)
    request = DlssRequest(output / "source.exr", output / "result.exr", worker)
    split = np.zeros(bounded.shape[:2], np.float32)
    split[:, 256:] = 1
    guides = {"full": None, "black": np.zeros_like(split), "white": np.ones_like(split),
              "split": split, "soft": np.full_like(split, 0.5)}
    provider = ExternalDlssProvider()
    report: dict = {"native": {}, "service": {}}
    predictions = {}
    try:
        for name, mask in guides.items():
            predicted = provider.predict(request, rgba, None, mask, None, True, CancellationToken())
            predictions[name] = predicted
            write_exr(output / f"native-{name}.exr", predicted)
            delta = np.abs(predicted[..., :3] - bounded)
            report["native"][name] = {
                "mean_change": float(delta.mean()),
                "left_change": float(delta[:, :192].mean()),
                "right_change": float(delta[:, 320:].mean()),
            }
        np.testing.assert_array_equal(predictions["black"][..., :3], bounded)
        np.testing.assert_array_equal(predictions["white"], predictions["full"])
        assert report["native"]["full"]["mean_change"] > 1e-5, "NR had no measurable effect"
        np.testing.assert_array_equal(predictions["split"][:, :192, :3], bounded[:, :192])
        assert report["native"]["split"]["right_change"] > 1e-5
        assert not np.array_equal(predictions["soft"], predictions["full"])
        assert not np.array_equal(predictions["soft"], predictions["black"])
    finally:
        provider.unload()

    # True scene-linear HDR, negative values, soft alpha and external zero vectors.
    hdr = np.concatenate((rgb * 16, split[..., None]), -1)
    hdr[32:64, 32:96, 0] = -0.25
    hdr[96:128, 32:96, :3] = (1e6, 5e5, -0.125)
    write_exr(request.source, hdr)
    motion = output / "external-zero-vectors.exr"
    write_exr(motion, np.zeros_like(hdr))
    registry = ProviderRegistry()
    registry.register("dlss-external", ExternalDlssProvider)
    service = DlssService(registry)
    try:
        for name, value in (("full", None), ("black", 0), ("white", 1), ("split", split),
                            ("soft", 0.5)):
            current = replace(request, output=output / f"linear-{name}.exr", protect_outside=False)
            if value is not None:
                guide = np.full_like(hdr, value) if np.isscalar(value) else np.repeat(value[..., None], 4, -1)
                mask_path = output / f"guide-{name}.exr"
                write_exr(mask_path, guide)
                current = replace(current, mask_source="input", mask=mask_path)
            result = service.run(current)
            channels = read_channels(current.output)
            generated = np.stack([channels[c] for c in ("R", "G", "B")], -1)
            assert np.isfinite(generated).all()
            np.testing.assert_array_equal(channels["A"], hdr[..., 3])
            if name == "black":
                np.testing.assert_array_equal(generated, hdr[..., :3])
            if name == "split":
                np.testing.assert_array_equal(generated[:, :192], hdr[:, :192, :3])
            report["service"][name] = {
                "mean_change": float(np.abs(generated - hdr[..., :3]).mean()),
                "min": float(generated.min()), "max": float(generated.max()),
                "hdr_rolloff": result["metadata"]["hdr_rolloff"],
            }
        for frame in (1, 2, 3):
            current = replace(request, output=output / f"temporal-{frame}.exr", frame=frame,
                              temporal=True, motion=motion, mask_source="source_alpha",
                              protect_outside=False)
            result = service.run(current)
            channels = read_channels(current.output)
            generated = np.stack([channels[c] for c in ("R", "G", "B")], -1)
            np.testing.assert_array_equal(generated[:, :192], hdr[:, :192, :3])
            report["service"][f"temporal-{frame}"] = result["metadata"]["history_reason"]
        assert report["service"]["temporal-2"] == "continuous"
        assert report["service"]["temporal-3"] == "continuous"
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
