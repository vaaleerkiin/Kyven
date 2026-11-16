"""Opt-in licensed Nuke contract test; ordinary tests remain host independent."""
import json
import os
import subprocess
from pathlib import Path

import pytest


def test_real_nuke_nodes_guidance_upgrades_and_reopen(tmp_path):
    executable = os.environ.get('KYVEN_NUKE_EXECUTABLE')
    if not executable:
        pytest.skip('Set KYVEN_NUKE_EXECUTABLE for licensed Nuke integration checks')
    report = tmp_path/'nuke-smoke.json'
    script = Path(__file__).parent/'nuke/smoke_nodes.py'
    result = subprocess.run([executable, '--nukex', '-i', '-t', str(script)],
        env={**os.environ, 'KYVEN_NUKE_SMOKE_REPORT': str(report)},
        capture_output=True, text=True, timeout=90, check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    assert report.is_file(), result.stdout + result.stderr
    payload = json.loads(report.read_text())
    assert payload.get('ok'), json.dumps(payload, indent=2) + result.stderr


def test_real_nuke_dlss_snapshot_is_lossless_single_part_and_survives_reopen(tmp_path):
    executable = os.environ.get('KYVEN_NUKE_EXECUTABLE')
    if not executable:
        pytest.skip('Set KYVEN_NUKE_EXECUTABLE for licensed Nuke integration checks')
    import numpy as np
    exr = pytest.importorskip('OpenEXR')
    shape = (128, 256)
    random = np.random.default_rng(19)
    source = {name: random.random(shape, dtype=np.float32) for name in ('R', 'G', 'B', 'A')}
    source['R'][0, :3] = [-0.12345679, 1_000_000, 0.12345679]
    source['A'][0, :3] = [-0.12345679, 1.2345679, 0.12345679]
    motion = {name: random.random(shape, dtype=np.float32) * 10 - 5
              for name in ('backward.u', 'backward.v', 'forward.u', 'forward.v')}
    depth = {'depth.Z': random.random(shape, dtype=np.float32)}
    mask = {'A': random.random(shape, dtype=np.float32)}
    for name, channels in zip(('source', 'motion', 'depth', 'mask'), (source, motion, depth, mask)):
        with exr.File({'compression': exr.ZIP_COMPRESSION},
                      {key: value.copy() for key, value in channels.items()}) as image:
            image.write(str(tmp_path / f'{name}.exr'))
    script = Path(__file__).parent / 'nuke/dlss_packed_inputs.py'
    result = subprocess.run([executable, '--nukex', '-i', '-t', str(script)],
        env={**os.environ, 'KYVEN_DLSS_PACKED_TEST': str(tmp_path)},
        capture_output=True, text=True, timeout=90, check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / 'report.json').read_text())
    with exr.File(report['packed'], separate_channels=True) as image:
        assert len(image.parts) == 1
        assert image.header()['compression'] == exr.ZIP_COMPRESSION
        actual = image.channels()
        expected = {**source, 'kyven_dlss_motion.x': motion['backward.u'],
                    'kyven_dlss_motion.y': motion['backward.v'],
                    'kyven_dlss_motion.forward_x': motion['forward.u'],
                    'kyven_dlss_motion.forward_y': motion['forward.v'],
                    'kyven_dlss_depth.Z': depth['depth.Z'], 'kyven_dlss_mask.value': mask['A']}
        assert set(actual) == set(expected)
        for name, values in expected.items():
            assert actual[name].pixels.dtype == np.float32
            np.testing.assert_array_equal(actual[name].pixels, values)
