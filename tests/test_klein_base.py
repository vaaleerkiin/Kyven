"""Base bundle metadata and optional transport must stay independent of Torch."""
from dataclasses import replace

from kyven.generative_fill.assets import manifest, manifest_checksum
from kyven.generative_fill.base_provider import KleinBaseProvider
from kyven.generative_fill.models import (
    BASE_PROVIDER_ID,
    PROVIDER_ID,
    FillSettings,
    GenerativeFillRequest,
)
from kyven.models.catalog import ModelCatalog


def test_base_bundle_and_registration(tmp_path):
    spec = ModelCatalog.builtin().get(BASE_PROVIDER_ID)
    files = {item["path"]: item for item in manifest(BASE_PROVIDER_ID)["files"]}
    assert spec.commercial_use and spec.runtime == "generative-fill"
    assert spec.sha256 == manifest_checksum(BASE_PROVIDER_ID)
    assert spec.size_bytes == sum(item["size_bytes"] for item in files.values())
    assert "klein-base" in files["flux-2-klein-4b-Q4_K_M.gguf"]["url"]
    provider = KleinBaseProvider(tmp_path)
    assert provider.models.name == BASE_PROVIDER_ID and provider.worker.name == "worker.py"
    assert provider.capabilities.minimum_vram_mb == 8192


def test_base_guidance_changes_cache_but_distilled_ignores_it(tmp_path):
    base = GenerativeFillRequest(tmp_path/"a", tmp_path/"b", tmp_path/"c", provider_id=BASE_PROVIDER_ID,
        fill=FillSettings(prompt="Remove person"))
    changed = replace(base, fill=replace(base.fill, moss_guidance=7.0))
    assert base.canonical() != changed.canonical()
    klein = replace(base, provider_id=PROVIDER_ID)
    assert klein.canonical() == replace(klein, fill=changed.fill).canonical()


def test_overlap_expansion_matches_reference_at_large_radius():
    import numpy as np

    from kyven.inpaint.masks import expand_binary_mask
    mask = np.zeros((43, 53), dtype=np.uint8)
    mask[0, 0] = mask[20:23, 25:29] = 255
    for radius in (0, 4, 32, 64, 128):
        # Independent union-of-squares reference. Some Windows Pillow builds
        # crash in MaxFilter when the kernel exceeds the tiny fixture dimensions.
        expected = np.zeros_like(mask)
        for y, x in zip(*np.nonzero(mask), strict=True):
            expected[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1] = 255
        np.testing.assert_array_equal(expand_binary_mask(mask, radius), expected)


def test_base_auto_stitches_addition_without_changing_outside_mask(tmp_path):
    import numpy as np
    from PIL import Image

    from kyven.inpaint.models import InpaintPrediction
    from kyven.inpaint.service import InpaintService
    from kyven.segment.providers.registry import ProviderRegistry

    source = np.full((64, 64, 3), 70, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:44, 20:44] = 255
    Image.fromarray(source).save(tmp_path/"source.png")
    Image.fromarray(mask).save(tmp_path/"mask.png")

    class Stub(KleinBaseProvider):
        called = False

        def predict(self, request, token):
            crop = np.asarray(Image.open(request.source).convert("RGB"))
            return InpaintPrediction(np.full_like(crop, 120), {"edit_operation": "add"})

        def stitch(self, generated, original, selection, token):
            self.called = True
            return generated, {"applied": True}

    provider = Stub(tmp_path)
    registry = ProviderRegistry()
    registry.register(BASE_PROVIDER_ID, lambda: provider)
    request = GenerativeFillRequest(tmp_path/"source.png", tmp_path/"mask.png", tmp_path/"out.png",
        provider_id=BASE_PROVIDER_ID, fill=FillSettings(prompt="Add a vase", mask_overlap=0),
        mask_grow=0, context_padding=0, edge_softness=0)
    InpaintService(registry).run(request)
    result = np.asarray(Image.open(request.output))
    assert provider.called
    np.testing.assert_array_equal(result[mask == 0], source[mask == 0])
