"""The CPU overlay must describe exactly the boxes consumed by inference."""

import json
import subprocess
import sys
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from kyven.errors import ErrorCode, KyvenError
from kyven.preview import prepare_refine_tiles_preview
from kyven.refine.models import RefineRequest
from kyven.refine.providers.vitmatte import VitMatteProvider
from kyven.refine.service import RefineService
from kyven.refine.tile_preview import TrimapPreviewCache
from kyven.segment.models import BoxPrompt
from kyven.segment.providers.registry import ProviderRegistry


@pytest.fixture
def preview_job(tmp_path):
    source, mask = tmp_path/'source.png', tmp_path/'mask.png'
    pixels = np.zeros((384, 512), np.uint8)
    pixels[50:330, 180:400] = 255
    pixels[40:340, 170:180] = 128
    Image.fromarray(pixels).save(mask)
    Image.new('RGB', (512, 384), (60, 100, 120)).save(source)
    provider = VitMatteProvider('unused-weights', 'unused-config', 'unused-processor')
    provider._load = Mock()
    provider._infer = Mock(side_effect=lambda image, *args:
                           np.full((image.height, image.width), 0.5, np.float32))
    registry = ProviderRegistry()
    registry.register('test-refine', lambda: provider)
    request = RefineRequest(source, mask, tmp_path/'result.png', tmp_path/'trimap.png',
        provider_id='test-refine', tile_size=128, tile_overlap=16, context_padding=12,
        foreground_radius=3, background_radius=4)
    return request, provider, RefineService(registry)


@pytest.mark.parametrize('mode', ['full', 'auto', 'manual'])
@pytest.mark.parametrize('generate', [True, False])
@pytest.mark.parametrize('skip', [True, False])
def test_preview_matches_inference_crop_boxes_and_guidance(preview_job, mode, generate, skip):
    request, provider, service = preview_job
    roi = BoxPrompt(80, 30, 430, 350) if mode == 'manual' else None
    request = replace(request, crop_mode=mode, roi=roi, generate_trimap=generate, skip_known_tiles=skip)
    payload = {**request.canonical(), 'mask': str(request.mask),
               'output': str(request.output.with_name('overlay.png'))}
    before = request.mask.read_bytes()
    preview = prepare_refine_tiles_preview(payload)
    assert request.mask.read_bytes() == before
    provider._load.assert_not_called()
    result = service.run(request)
    assert preview['processing_roi'] == result.metadata['processing_roi']
    for key in ('tiles', 'tiles_total', 'tiles_skipped', 'unknown_pixels'):
        assert preview[key] == result.metadata[key]
    with Image.open(request.trimap_output) as image:
        guidance = np.asarray(image)
    active = [box for box in preview['boxes'] if box['run']]
    assert len(active) == provider._infer.call_count
    for box, call in zip(active, provider._infer.call_args_list):
        actual = np.asarray(call.args[1])
        expected = guidance[box['y0']:box['y1'], box['x0']:box['x1']]
        np.testing.assert_array_equal(actual, expected)
    with Image.open(payload['output']) as overlay:
        assert overlay.mode == 'RGBA'
        assert overlay.size == (512, 384)


def test_preview_highlights_unknown_pixels_and_reports_skips(preview_job):
    request, _, _ = preview_job
    result = prepare_refine_tiles_preview({**request.canonical(), 'mask': str(request.mask),
        'output': str(request.output), 'generate_trimap': False})
    assert 0 < result['tiles'] < result['tiles_total']
    assert result['tiles_skipped'] > 0
    with Image.open(request.output) as image:
        assert image.getpixel((175, 200)) == (255, 195, 35, 200)


def test_square_outlines_remain_visible_over_an_entirely_unknown_crop(preview_job):
    request, _, _ = preview_job
    Image.new('L', (512, 384), 128).save(request.mask)
    result = prepare_refine_tiles_preview({**request.canonical(), 'mask': str(request.mask),
        'output': str(request.output), 'generate_trimap': False})
    assert result['tiles'] == result['tiles_total']
    with Image.open(request.output) as image:
        # A later overlapping tile or the yellow band must not erase an earlier edge.
        assert image.getpixel((127, 80)) == (55, 235, 105, 255)
        assert image.getpixel((112, 80)) == (55, 235, 105, 255)
        assert image.getpixel((80, 127)) == (55, 235, 105, 255)
        assert image.getpixel((80, 112)) == (55, 235, 105, 255)
        assert image.getpixel((80, 80)) == (255, 195, 35, 200)


def test_preview_validates_tile_overlap_before_publishing(preview_job):
    request, _, _ = preview_job
    with pytest.raises(KyvenError) as error:
        prepare_refine_tiles_preview({'mask': str(request.mask), 'output': str(request.output),
                                     'tile_size': 128, 'tile_overlap': 64})
    assert error.value.code == ErrorCode.INVALID_REQUEST
    assert not request.output.exists()


def test_preview_rejects_mismatched_source_format(preview_job):
    request, _, _ = preview_job
    with pytest.raises(KyvenError):
        prepare_refine_tiles_preview({'mask': str(request.mask), 'output': str(request.output),
                                     'image_width': 100, 'image_height': 80})
    assert not request.output.exists()


def test_preview_does_not_import_model_runtime(preview_job):
    request, _, _ = preview_job
    payload = {'mask': str(request.mask), 'output': str(request.output)}
    code = ('import json,sys; from kyven.preview import prepare_refine_tiles_preview; '
            'prepare_refine_tiles_preview(json.loads(sys.argv[1])); '
            'assert "torch" not in sys.modules; assert "transformers" not in sys.modules')
    process = subprocess.run([sys.executable, '-c', code, json.dumps(payload)],
                             capture_output=True, text=True, timeout=30, check=False)
    assert process.returncode == 0, process.stdout + process.stderr


def test_tile_overlap_roi_and_label_changes_reuse_prepared_guidance(preview_job, monkeypatch):
    from kyven.refine import tile_preview

    request, _, _ = preview_job
    cache = TrimapPreviewCache()
    monkeypatch.setattr(tile_preview, '_trimap_cache', cache)
    generate = Mock(wraps=tile_preview.generate_trimap)
    monkeypatch.setattr(tile_preview, 'generate_trimap', generate)
    payload = {**request.canonical(), 'mask': str(request.mask), 'output': str(request.output)}
    first = prepare_refine_tiles_preview(payload)
    assert not first['trimap_cache_hit']
    for changes in ({'tile_size': 256}, {'tile_overlap': 32}, {'context_padding': 30},
                    {'crop_mode': 'manual', 'roi': BoxPrompt(80, 30, 430, 350).canonical()},
                    {'preview_labels': True}):
        result = prepare_refine_tiles_preview({**payload, **changes})
        assert result['trimap_cache_hit']
        assert result['trimap_bounds'] == first['trimap_bounds']
    assert generate.call_count == 1
    changed = prepare_refine_tiles_preview({**payload, 'foreground_radius': 8})
    assert not changed['trimap_cache_hit']
    assert generate.call_count == 2


def test_guidance_cache_invalidates_upstream_edits_and_is_read_only(preview_job):
    request, _, _ = preview_job
    cache = TrimapPreviewCache()
    before, hit = cache.prepare(request)
    assert not hit
    with pytest.raises(ValueError):
        before.pixels[0, 0] = 128
    assert cache.prepare(request)[1]
    Image.new('L', (512, 384), 0).save(request.mask)
    after, hit = cache.prepare(request)
    assert not hit
    assert not after.pixels.any()
    assert before.pixels.any()


def test_guidance_cache_memory_is_bounded(preview_job):
    request, _, _ = preview_job
    cache = TrimapPreviewCache(max_bytes=512*384)
    first, _ = cache.prepare(request)
    assert cache.prepare(request)[1]
    alternate = replace(request, foreground_radius=8)
    cache.prepare(alternate)
    assert cache._bytes <= cache.max_bytes
    assert not cache.prepare(request)[1]
    uncached = TrimapPreviewCache(max_bytes=100)
    assert not uncached.prepare(request)[1]
    assert not uncached.prepare(request)[1]
    assert uncached._bytes == 0
    assert first.pixels.nbytes == 512*384
