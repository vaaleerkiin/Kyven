"""Lab accuracy, clean reference selection and durable native Mage patches."""

from dataclasses import replace

import numpy as np
from PIL import Image

from kyven.generative_fill.lab import lab_to_rgb, match_lab_context, rgb_to_lab
from kyven.generative_fill.models import MAGE_PROVIDER_ID, FillSettings, GenerativeFillRequest
from kyven.inpaint.models import InpaintCapabilities, InpaintPrediction
from kyven.inpaint.service import InpaintService
from kyven.models.catalog import ModelCatalog
from kyven.segment.providers.registry import ProviderRegistry


def test_lab_round_trip_and_nuke_convention():
    rgb = np.random.default_rng(7).integers(0, 256, (71, 53, 3), dtype=np.uint8)
    np.testing.assert_array_equal(lab_to_rgb(rgb_to_lab(rgb)), rgb)
    red = rgb_to_lab(np.array([[255, 0, 0]], dtype=np.uint8))[0]
    np.testing.assert_allclose(red, [.5324, .8009, .6720], atol=.0001)


def test_lab_fit_ignores_the_removed_subject_and_corrects_context():
    rng = np.random.default_rng(19)
    source = rng.integers(35, 185, (80, 96, 3), dtype=np.uint8)
    generated = np.uint8(source + np.array((20, 5, 12), dtype=np.uint8))
    mask = np.zeros((80, 96), dtype=np.uint8)
    mask[25:55, 30:65] = 255
    result, first = match_lab_context(generated, source, mask)
    changed_source = source.copy()
    changed_source[mask > 0] = (0, 255, 0)
    second_result, second = match_lab_context(generated, changed_source, mask)
    assert first["applied"] and first["samples"] <= 65536
    assert first == second
    np.testing.assert_array_equal(result, second_result)
    known = mask == 0
    assert np.mean(abs(result[known].astype(float)-source[known])) < 3
    full, details = match_lab_context(generated, source, np.full_like(mask, 255))
    np.testing.assert_array_equal(full, generated)
    assert details["reason"] == "insufficient_clean_context"


def test_constant_context_does_not_explode():
    generated = np.full((50, 50, 3), 110, np.uint8)
    source = np.full_like(generated, 135)
    mask = np.zeros((50, 50), np.uint8)
    mask[20:30, 20:30] = 255
    matched, details = match_lab_context(generated, source, mask)
    assert np.isfinite(matched).all() and details["applied"]
    np.testing.assert_allclose(matched, source, atol=1)


def test_native_patch_pair_is_cached_restored_and_source_safe(tmp_path):
    class Provider:
        calls = 0
        capabilities = InpaintCapabilities(MAGE_PROVIDER_ID, "test Mage", "1", "hash",
                                           "MIT", "https://example.test", True, 0)

        def predict(self, request, cancellation):
            self.calls += 1
            self.mask = np.asarray(Image.open(request.mask).convert("L")).copy()
            rgb = np.asarray(Image.open(request.source).convert("RGB")).copy()
            rgb = np.uint8(np.clip(rgb.astype(float)+(18, 5, 12), 0, 255))
            return InpaintPrediction(rgb, {})

        def unload(self):
            pass

    source = np.random.default_rng(6).integers(30, 180, (80, 96, 3), dtype=np.uint8)
    mask = np.zeros((80, 96), dtype=np.uint8)
    mask[25:55, 30:65] = 255
    mask[24, 30:65] = 32
    Image.fromarray(source).save(tmp_path / "source.png")
    Image.fromarray(mask).save(tmp_path / "mask.png")
    provider = Provider()
    registry = ProviderRegistry()
    registry.register(MAGE_PROVIDER_ID, lambda: provider)
    service = InpaintService(registry)
    request = GenerativeFillRequest(tmp_path / "source.png", tmp_path / "mask.png",
        tmp_path / "out.png", patch_output=tmp_path / "patch.png",
        mask_output=tmp_path / "blend.png", provider_id=MAGE_PROVIDER_ID,
        mask_grow=18, mask_threshold=.8,
        fill=FillSettings(prompt="Remove", blend_margin=20, mask_overlap=32, native_postprocess=True))
    result = service.run(request)
    np.testing.assert_array_equal(provider.mask, np.where(mask > 0, 255, 0).astype(np.uint8))
    assert result.metadata["native_postprocess"]
    assert result.metadata["color_matching"]["method"] == "reinhard_lab"
    original = np.asarray(Image.open(request.output))
    np.testing.assert_array_equal(original[mask == 0], source[mask == 0])
    raw = Image.open(tmp_path / "patch.png")
    matched = Image.open(tmp_path / "patch.lab.png")
    assert raw.mode == matched.mode == "RGBA"
    assert np.all(np.asarray(raw)[..., 3] == 255)
    assert not np.array_equal(np.asarray(raw)[..., :3], np.asarray(matched)[..., :3])
    saved = (tmp_path / "patch.lab.png").read_bytes()
    (tmp_path / "patch.lab.png").unlink()
    service.run(request)
    assert (tmp_path / "patch.lab.png").read_bytes() == saved
    assert provider.calls == 1
    assert request.canonical() != replace(request, fill=replace(request.fill, native_postprocess=False)).canonical()


def test_base_is_retired_but_existing_requests_keep_metadata():
    catalog = ModelCatalog.builtin()
    base = "flux2-klein-base-4b-q4-lanpaint"
    assert catalog.get(base).retired
    assert base not in {entry.model_id for entry in catalog.list()}
    assert base in {entry.model_id for entry in catalog.list(include_retired=True)}


def test_native_contract_survives_job_payload_parsing(tmp_path):
    from kyven.server.jobs import JobManager
    for name in ("source.png", "mask.png"):
        Image.new("RGB", (32, 32)).save(tmp_path / name)
    request = JobManager.generative_fill_request_from_payload({
        "source": str(tmp_path / "source.png"), "mask": str(tmp_path / "mask.png"),
        "output": str(tmp_path / "out.png"), "patch_output": str(tmp_path / "patch.png"),
        "model_id": MAGE_PROVIDER_ID, "fill": {"prompt": "Remove", "native_postprocess": True}})
    assert request.fill.native_postprocess and request.provider_id == MAGE_PROVIDER_ID
