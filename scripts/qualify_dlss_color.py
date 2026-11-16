"""Opt-in RTX check of native color shifts and Source color preservation."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from kyven.dlss.images import read_channels, write_exr
from kyven.dlss.models import DlssRequest
from kyven.dlss.provider import ExternalDlssProvider
from kyven.dlss.service import DlssService
from kyven.segment.providers.registry import ProviderRegistry


def qualify(worker: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[:288, :512].astype(np.float32)
    texture = 0.02 * np.sin(x / 7) * np.cos(y / 9)
    gray = np.repeat((0.08 + x / 850 + texture)[..., None], 3, -1)
    chart = np.empty_like(gray)
    for index, color in enumerate(((0.65, 0.09, 0.03), (0.03, 0.6, 0.12),
                                   (0.04, 0.08, 0.65), (0.18, 0.18, 0.18))):
        chart[:, index * 128:(index + 1) * 128] = color
    chart += texture[..., None]
    hdr = chart * 16
    hdr[32:64, 32:64] = (-0.125, 2, 8)
    hdr[96:128, 32:96] = (1e5, 5e4, -0.125)
    registry = ProviderRegistry()
    registry.register('dlss-external', ExternalDlssProvider)
    service = DlssService(registry)
    report = {}
    try:
        for name, rgb in (('gray', gray), ('chart', chart), ('hdr', hdr)):
            alpha = np.broadcast_to(np.linspace(-0.2, 1.2, 512, dtype=np.float32), rgb.shape[:2])
            rgba = np.concatenate((rgb, alpha[..., None]), -1)
            source = output / f'{name}-source.exr'
            write_exr(source, rgba)
            base = DlssRequest(source, output / 'result.exr', worker)
            original_norm = np.max(np.abs(rgb), -1)
            original_chroma = rgb / np.maximum(original_norm[..., None], 1e-30)
            report[name] = {}
            for enabled in (False, True):
                request = replace(base, preserve_source_color=enabled,
                                  output=output / f'{name}-preserve-{enabled}.exr')
                result = service.run(request)
                channels = read_channels(request.output)
                generated = np.stack([channels[c] for c in ('R', 'G', 'B')], -1)
                norm = np.max(np.abs(generated), -1)
                valid = (original_norm > 0) & (norm > 0)
                chroma = generated / np.maximum(norm[..., None], 1e-30)
                error = float(np.abs(chroma[valid] - original_chroma[valid]).max())
                np.testing.assert_array_equal(channels['A'], alpha)
                assert np.isfinite(generated).all()
                if enabled:
                    assert error < 1e-6, (name, error)
                report[name][str(enabled)] = {
                    'max_normalized_rgb_error': error,
                    'mean_rgb_change': float(np.abs(generated - rgb).mean()),
                    'source_color': result['metadata']['source_color'],
                }
        assert report['chart']['True']['mean_rgb_change'] > 1e-6
    finally:
        registry.unload_all()
    (output / 'qualification.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(qualify(args.worker.resolve(), args.output.resolve()), indent=2))
