"""Guidance contracts across mask preparation, preview, inference and publication."""
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image, ImageFilter
from test_refine import SyntheticRefinementProvider

from kyven.cache_manifest import manifest_path
from kyven.errors import ErrorCode, KyvenError
from kyven.preview import prepare_trimap_preview
from kyven.refine.models import RefinePrediction, RefineRequest
from kyven.refine.service import RefineService
from kyven.refine.trimap import generate_trimap, normalize_trimap
from kyven.segment.models import BoxPrompt
from kyven.segment.providers.registry import ProviderRegistry


@pytest.mark.parametrize('values', [
    [0] * 9, [255] * 9, [0, 1, 64, 85, 127, 128, 169, 170, 255],
])
@pytest.mark.parametrize('radii', [(0, 0), (1, 2), (2, 1), (6, 6)])
def test_positive_alpha_is_binarized_before_morphology(values, radii):
    mask = np.tile(np.asarray(values, dtype=np.uint8), (7, 1))
    binary = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    foreground, background = radii
    eroded = np.asarray(binary.filter(ImageFilter.MinFilter(2 * foreground + 1)) if foreground else binary) > 0
    dilated = np.asarray(binary.filter(ImageFilter.MaxFilter(2 * background + 1)) if background else binary) > 0
    expected = np.where(eroded, 255, np.where(dilated, 128, 0))
    np.testing.assert_array_equal(generate_trimap(mask, *radii), expected)


def test_manual_guide_preserves_uncertainty_and_uses_explicit_boundaries():
    np.testing.assert_array_equal(
        normalize_trimap(np.array([[0, 85, 86, 128, 169, 170, 255]], dtype=np.uint8)),
        [[0, 0, 128, 128, 128, 255, 255]])


@pytest.fixture
def guidance_job(tmp_path):
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[1:9, 2:10] = 64
    mask[3:7, 4:8] = 255
    source_path, mask_path = tmp_path/'source.png', tmp_path/'mask.png'
    Image.new('RGB', (12, 10), (40, 80, 120)).save(source_path)
    Image.fromarray(mask).save(mask_path)

    class Capture(SyntheticRefinementProvider):
        def predict(self, request, cancellation):
            with Image.open(request.mask) as image:
                self.guidance = np.asarray(image).copy()
            return super().predict(request, cancellation)

    provider = Capture()
    registry = ProviderRegistry()
    registry.register('synthetic-refine', lambda: provider)
    request = RefineRequest(source=source_path, mask=mask_path, output=tmp_path/'alpha.png',
        trimap_output=tmp_path/'guide.png', provider_id='synthetic-refine',
        foreground_radius=1, background_radius=2)
    return request, provider, RefineService(registry), mask


@pytest.mark.parametrize('generate', [True, False])
@pytest.mark.parametrize('roi', [None, BoxPrompt(3, 2, 9, 8),
                                      BoxPrompt(-5, -5, 6, 7)])
def test_preview_saved_guide_and_model_input_agree(guidance_job, generate, roi):
    request, provider, service, original = guidance_job
    request = replace(request, generate_trimap=generate, roi=roi,
                      crop_mode='manual' if roi else 'full')
    result = service.run(request)
    preview = request.output.with_name('preview.png')
    prepare_trimap_preview({'mask': str(request.mask), 'output': str(preview),
        'generate_trimap': generate, 'foreground_radius': 1, 'background_radius': 2,
        'roi': roi.canonical() if roi else None})
    with Image.open(preview) as image, Image.open(request.trimap_output) as saved:
        np.testing.assert_array_equal(np.asarray(image), np.asarray(saved))
        guide = np.asarray(saved)
    region = result.metadata['processing_roi']
    x0, y0, x1, y1 = (region[k] for k in ('x0', 'y0', 'x1', 'y1'))
    np.testing.assert_array_equal(guide[y0:y1, x0:x1], provider.guidance)
    outside = np.ones(original.shape, dtype=bool)
    outside[y0:y1, x0:x1] = False
    assert not guide[outside].any()
    with Image.open(request.output) as image:
        np.testing.assert_array_equal(np.asarray(image)[outside], original[outside])


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_alpha_is_rejected_without_successful_output(guidance_job, value):
    request, provider, service, _ = guidance_job
    provider.predict = lambda req, token: RefinePrediction(np.full((10, 12), value), {})
    with pytest.raises(KyvenError) as error:
        service.run(request)
    assert error.value.code == ErrorCode.INFERENCE_FAILED
    assert not request.output.exists()
    assert not manifest_path(request.output).exists()


def test_refine_failure_preserves_existing_completed_output(guidance_job):
    request, provider, service, _ = guidance_job
    service.run(request)
    before = request.output.read_bytes()
    guide_before = request.trimap_output.read_bytes()
    provider.predict = lambda req, token: RefinePrediction(np.full((10, 12), np.nan), {})
    with pytest.raises(KyvenError):
        service.run(replace(request, foreground_radius=2))
    assert request.output.read_bytes() == before
    assert request.trimap_output.read_bytes() == guide_before


def test_outside_backend_roi_is_rejected_before_prediction(guidance_job):
    request, provider, service, _ = guidance_job
    provider.predict = lambda *args: pytest.fail('Invalid ROI must not predict')
    with pytest.raises(KyvenError) as error:
        service.run(replace(request, crop_mode='manual', roi=BoxPrompt(20, 20, 30, 30)))
    assert error.value.code == ErrorCode.INVALID_REQUEST
    assert not request.output.exists()


def test_binary_guidance_snapshot_preserves_original_soft_alpha_outside_roi(guidance_job):
    request, _, service, original = guidance_job
    snapshot = np.empty((*original.shape, 4), dtype=np.uint8)
    snapshot[..., :3] = ((original > 0).astype(np.uint8)*255)[..., None]
    snapshot[..., 3] = original
    Image.fromarray(snapshot).save(request.mask)
    request = replace(request, crop_mode='manual', roi=BoxPrompt(4, 3, 8, 7))
    service.run(request)
    with Image.open(request.output) as output:
        assert np.asarray(output)[2, 3] == 64


@pytest.mark.parametrize('mode', ['auto', 'full'])
def test_zero_mask_has_zero_guidance_and_repeat_is_cached(guidance_job, mode):
    request, provider, service, mask = guidance_job
    Image.fromarray(np.zeros_like(mask)).save(request.mask)
    request = replace(request, crop_mode=mode)
    service.run(request)
    assert not provider.guidance.any()
    provider.predict = lambda *args: pytest.fail('Cached frame must not predict again')
    assert service.run(request).metadata['cache_hit']
