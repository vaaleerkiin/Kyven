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
    import numpy as np
    from PIL import Image

    from kyven.refine.trimap import generate_trimap

    for export in payload['refine_exports']:
        with Image.open(export['mask']) as image:
            mask = np.asarray(image.convert('L'))
        actual = generate_trimap(mask, export['foreground_radius'], export['background_radius'])
        np.testing.assert_array_equal(actual, export['expected'])


def test_real_nuke_generative_fill_mask_cache_and_reopen(tmp_path):
    executable = os.environ.get('KYVEN_NUKE_EXECUTABLE')
    if not executable:
        pytest.skip('Set KYVEN_NUKE_EXECUTABLE for licensed Nuke integration checks')
    report = tmp_path / 'fill-graph.json'
    script = Path(__file__).parent / 'nuke/generative_fill.py'
    result = subprocess.run([executable, '--nukex', '-i', '-t', str(script)],
        env={**os.environ, 'KYVEN_FILL_REPORT': str(report)},
        capture_output=True, text=True, timeout=90, check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    assert report.is_file(), result.stdout + result.stderr
    payload = json.loads(report.read_text())
    assert payload.get('ok'), json.dumps(payload, indent=2) + result.stderr


def test_real_nuke_refine_optional_integration_does_not_run_model_or_change_cache(tmp_path):
    executable = os.environ.get('KYVEN_NUKE_EXECUTABLE')
    if not executable:
        pytest.skip('Set KYVEN_NUKE_EXECUTABLE for licensed Nuke integration checks')
    from PIL import Image

    matte = tmp_path / 'cached-matte.png'
    Image.new('L', (96, 64), 204).save(matte)
    original = matte.read_bytes()
    report = tmp_path / 'integration.json'
    script = Path(__file__).parent / 'nuke/refine_integration.py'
    result = subprocess.run([executable, '--nukex', '-i', '-t', str(script)],
        env={**os.environ, 'KYVEN_NUKE_INTEGRATION_REPORT': str(report),
             'KYVEN_NUKE_INTEGRATION_MATTE': str(matte)},
        capture_output=True, text=True, timeout=90, check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    assert report.is_file(), result.stdout + result.stderr
    payload = json.loads(report.read_text())
    assert payload.get('ok'), json.dumps(payload, indent=2) + result.stderr
    assert matte.read_bytes() == original


def test_real_nuke_tile_preview_uses_cpu_http_planner_with_no_inference(tmp_path):
    executable = os.environ.get('KYVEN_NUKE_EXECUTABLE')
    if not executable:
        pytest.skip('Set KYVEN_NUKE_EXECUTABLE for licensed Nuke integration checks')
    from kyven.models.catalog import ModelCatalog
    from kyven.segment.providers.registry import ProviderRegistry
    from kyven.segment.service import SegmentService
    from kyven.server.app import KyvenServer, ServerConfig
    from kyven.server.jobs import JobManager

    registry = ProviderRegistry()
    server = KyvenServer(ServerConfig(token='x'*32, models_dir=tmp_path, port=0),
        JobManager(SegmentService(registry)), registry, ModelCatalog.builtin())
    server.start()
    report = tmp_path/'tile-preview.json'
    script = Path(__file__).parent/'nuke/refine_tile_preview.py'
    try:
        result = subprocess.run([executable, '--nukex', '-i', '-t', str(script)],
            env={**os.environ, 'KYVEN_NUKE_TILE_PREVIEW_REPORT': str(report),
                 'KYVEN_NUKE_TILE_PREVIEW_URL': f'http://127.0.0.1:{server.port}'},
            capture_output=True, text=True, timeout=90, check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        assert report.is_file(), result.stdout + result.stderr
        payload = json.loads(report.read_text())
        assert payload.get('ok'), json.dumps(payload, indent=2) + result.stdout + result.stderr
        assert not registry.loaded_ids
    finally:
        server.shutdown()


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


def test_real_nuke_server_start_stop_and_offline_cache(tmp_path):
    executable = os.environ.get('KYVEN_NUKE_EXECUTABLE')
    if not executable:
        pytest.skip('Set KYVEN_NUKE_EXECUTABLE for licensed Nuke integration checks')
    import numpy as np
    exr = pytest.importorskip('OpenEXR')
    rgba = np.full((2, 2, 4), 0.2, np.float32)
    rgba[..., 3] = 0.75
    with exr.File({'compression': exr.ZIP_COMPRESSION}, {'RGBA': rgba}) as image:
        image.write(str(tmp_path / 'cached.exr'))
    script = Path(__file__).parent / 'nuke/dlss_server_controls.py'
    result = subprocess.run([executable, '--nukex', '-i', '-t', str(script)],
        env={**os.environ, 'KYVEN_SERVER_CONTROL_TEST': str(tmp_path)},
        capture_output=True, text=True, timeout=150, check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((tmp_path / 'controls.json').read_text())['passed']
