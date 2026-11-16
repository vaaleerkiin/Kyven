"""Invalid model pixels must never be published as successful cached results."""
import numpy as np
import pytest
from PIL import Image
from test_inpaint import FakeInpaintProvider

from kyven.cache_manifest import manifest_path
from kyven.errors import ErrorCode, KyvenError
from kyven.inpaint.models import InpaintPrediction, InpaintRequest
from kyven.inpaint.service import InpaintService
from kyven.segment.output import write_mask_png_atomic
from kyven.segment.providers.registry import ProviderRegistry


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('existing', [False, True])
def test_nonfinite_matte_cannot_replace_existing_image(tmp_path, value, existing):
    output = tmp_path/'mask.png'
    if existing:
        Image.new('L', (3, 2), 87).save(output)
    before = output.read_bytes() if existing else None
    with pytest.raises(KyvenError) as error:
        write_mask_png_atomic(output, np.full((2, 3), value))
    assert error.value.code == ErrorCode.INFERENCE_FAILED
    assert (output.read_bytes() if output.exists() else None) == before


@pytest.mark.parametrize('shape', [(2,), (2, 3, 1), (2, 3, 4)])
def test_invalid_matte_shape_is_rejected(tmp_path, shape):
    with pytest.raises(KyvenError) as error:
        write_mask_png_atomic(tmp_path/'mask.png', np.zeros(shape))
    assert error.value.code == ErrorCode.INFERENCE_FAILED


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_inpaint_prediction_has_no_successful_result(tmp_path, value):
    source, mask, output = (tmp_path/name for name in ('source.png', 'mask.png', 'result.png'))
    Image.new('RGB', (8, 8), (40, 60, 80)).save(source)
    Image.new('L', (8, 8), 255).save(mask)
    provider = FakeInpaintProvider()
    provider.predict = lambda *args: InpaintPrediction(np.full((8, 8, 3), value))
    registry = ProviderRegistry()
    registry.register('fake', lambda: provider)
    with pytest.raises(KyvenError) as error:
        InpaintService(registry).run(InpaintRequest(source, mask, output,
                                                   provider_id='fake', crop_mode='full'))
    assert error.value.code == ErrorCode.INFERENCE_FAILED
    assert not output.exists()
    assert not manifest_path(output).exists()
