"""Sparse trimap tiling must preserve dense inference's context and blended alpha."""

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from kyven.cancellation import CancellationToken
from kyven.errors import ErrorCode, KyvenError
from kyven.refine.models import RefineRequest
from kyven.refine.providers.mematte import MEMatteProvider
from kyven.refine.providers.vitmatte import VitMatteProvider
from kyven.refine.tiling import plan_tiles


@pytest.fixture(params=['vitmatte', 'mematte'])
def tiled_provider(tmp_path, request):
    if request.param == 'vitmatte':
        provider = VitMatteProvider('missing-weights', 'unused-config', 'unused-processor')
    else:
        provider = MEMatteProvider('missing-weights')
    provider._load = Mock()
    source, mask = tmp_path / 'rgb.png', tmp_path / 'trimap.png'
    rgb = np.random.default_rng(9).integers(0, 256, (384, 384, 3), dtype=np.uint8)
    guide = np.zeros((384, 384), np.uint8)
    guide[:, 184:] = 255
    guide[80:330, 180:184] = 128
    Image.fromarray(rgb).save(source)
    Image.fromarray(guide).save(mask)

    def infer(image, trimap, *args):
        pixels = np.asarray(image, np.float32) / 255
        # Depend on the entire tile, not just the unknown pixels, so altered model
        # context or missing overlap contributions would change the comparison.
        context = float(pixels[..., 1].mean()) * 0.2
        return pixels[..., 0] * 0.6 + context

    provider._infer = Mock(side_effect=infer)
    job = RefineRequest(source, mask, tmp_path / 'result.png', tile_size=128, tile_overlap=32)
    return provider, job, guide


def test_sparse_tiles_match_dense_alpha_with_the_same_context(tiled_provider):
    provider, request, guide = tiled_provider
    sparse = provider.predict(request, CancellationToken())
    assert sparse.metadata['tiles'] == provider._infer.call_count == 4
    assert sparse.metadata['tiles_total'] == 16
    assert sparse.metadata['tiles_skipped'] == 12
    assert sparse.metadata['unknown_pixels'] == 1000
    assert all(call.args[0].size == (128, 128) for call in provider._infer.call_args_list)
    provider._infer.reset_mock()
    dense = provider.predict(replace(request, skip_known_tiles=False), CancellationToken())
    assert dense.metadata['tiles'] == provider._infer.call_count == 16
    np.testing.assert_array_equal(sparse.alpha, dense.alpha)
    assert (sparse.alpha[guide == 0] == 0).all()
    assert (sparse.alpha[guide == 255] == 1).all()


@pytest.mark.parametrize('tile', [0, 128])
@pytest.mark.parametrize('kind', ['black', 'white', 'mixed'])
def test_known_crop_never_loads_or_runs_a_model(tiled_provider, tile, kind):
    provider, request, guide = tiled_provider
    guide[:] = 0 if kind == 'black' else 255
    if kind == 'mixed':
        guide[::2, ::2] = 0
    Image.fromarray(guide).save(request.mask)
    result = provider.predict(replace(request, tile_size=tile), CancellationToken())
    provider._load.assert_not_called()
    provider._infer.assert_not_called()
    assert result.metadata['tiles'] == 0
    assert result.metadata['tiles_skipped'] == result.metadata['tiles_total']
    assert result.metadata['device'] == 'not_used'
    np.testing.assert_array_equal(result.alpha, guide / 255)


def test_skip_can_be_disabled_for_a_dense_comparison(tiled_provider):
    provider, request, guide = tiled_provider
    Image.fromarray(np.zeros_like(guide)).save(request.mask)
    result = provider.predict(replace(request, skip_known_tiles=False), CancellationToken())
    assert provider._infer.call_count == 16
    provider._load.assert_called_once()
    assert result.metadata['tiles_skipped'] == 0
    assert not result.alpha.any()


def test_single_full_crop_uses_one_inference_and_correct_canvas(tiled_provider):
    provider, request, _ = tiled_provider
    result = provider.predict(replace(request, tile_size=0), CancellationToken())
    assert result.metadata['tiles_total'] == result.metadata['tiles'] == 1
    assert result.alpha.shape == (384, 384)
    assert provider._infer.call_args.args[0].size == (384, 384)


def test_cancelled_plan_never_loads_model(tiled_provider):
    provider, request, _ = tiled_provider
    token = CancellationToken()
    token.cancel()
    with pytest.raises(KyvenError) as error:
        provider.predict(request, token)
    assert error.value.code == ErrorCode.CANCELLED
    provider._load.assert_not_called()


def test_cancel_between_tiles_stops_immediately(tiled_provider):
    provider, request, _ = tiled_provider
    token = CancellationToken()

    def cancel(image, *args):
        token.cancel()
        return np.zeros((image.height, image.width), np.float32)

    provider._infer.side_effect = cancel
    with pytest.raises(KyvenError) as error:
        provider.predict(request, token)
    assert error.value.code == ErrorCode.CANCELLED
    assert provider._infer.call_count == 1


@pytest.mark.parametrize('bad', ['shape', 'nan', 'inf', 'oom'])
def test_invalid_tile_and_memory_errors_are_structured(tiled_provider, bad):
    provider, request, _ = tiled_provider
    if bad == 'oom':
        provider._infer.side_effect = RuntimeError('CUDA out of memory')
    else:
        provider._infer.side_effect = lambda *args: np.full(
            (2, 2) if bad == 'shape' else (128, 128),
            np.nan if bad == 'nan' else np.inf if bad == 'inf' else 0, np.float32)
    with pytest.raises(KyvenError) as error:
        provider.predict(request, CancellationToken())
    assert error.value.code == ErrorCode.INFERENCE_FAILED
    if bad == 'oom':
        assert error.value.recoverable
        assert '512' in error.value.suggested_action


def test_skip_setting_changes_deterministic_cache_key(tiled_provider):
    _, request, _ = tiled_provider
    key = request.cache_key('3', 'fixture')
    assert key == request.cache_key('3', 'fixture')
    assert key != replace(request, skip_known_tiles=False).cache_key('3', 'fixture')


@pytest.mark.parametrize('size', [(129, 131), (384, 384), (8, 512)])
@pytest.mark.parametrize('overlap', [0, 32, 63])
def test_unknown_pixels_on_borders_and_seams_keep_all_covering_tiles(size, overlap):
    width, height = size
    pixels = np.zeros((height, width), np.uint8)
    random = np.random.default_rng(41)
    pixels[random.integers(0, height, 12), random.integers(0, width, 12)] = 128
    pixels[0, 0] = pixels[-1, -1] = 128
    trimap = Image.fromarray(pixels)
    dense = plan_tiles(trimap, 128, overlap, False, CancellationToken())
    sparse = plan_tiles(trimap, 128, overlap, True, CancellationToken())
    expected = [box for box in dense.boxes
                if (pixels[box[1]:box[3], box[0]:box[2]] == 128).any()]
    assert list(sparse.boxes) == expected
    covered = np.zeros_like(pixels, dtype=bool)
    for x, y, x1, y1 in sparse.boxes:
        covered[y:y1, x:x1] = True
    assert covered[pixels == 128].all()
