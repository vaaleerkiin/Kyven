from __future__ import annotations

import importlib.util
import subprocess
import sys
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from kyven.cancellation import CancellationToken
from kyven.client import KyvenClient
from kyven.errors import ErrorCode, KyvenError
from kyven.generative_fill.assets import manifest, manifest_checksum
from kyven.generative_fill.models import (
    PROVIDER_ID,
    EnhancePromptRequest,
    FillSettings,
    GenerativeFillRequest,
    working_dimensions,
)
from kyven.generative_fill.provider import KleinLanPaintProvider
from kyven.inpaint.models import InpaintCapabilities, InpaintPrediction
from kyven.inpaint.service import InpaintService
from kyven.models.catalog import ModelCatalog
from kyven.segment.providers.registry import ProviderRegistry
from kyven.segment.service import SegmentService
from kyven.server.app import KyvenServer, ServerConfig
from kyven.server.jobs import JobManager


class SyntheticFill:
    capabilities = InpaintCapabilities(PROVIDER_ID, "Synthetic", "1", "hash", "Apache-2.0", "", True, 8192)

    def __init__(self):
        self.calls = []
        self.unloaded = False

    def predict(self, request, cancellation):
        self.calls.append((request, np.asarray(Image.open(request.mask)).copy()))
        cancellation.raise_if_cancelled()
        source = np.asarray(Image.open(request.source).convert("RGB"))
        return InpaintPrediction(np.full_like(source, 240))

    def enhance(self, request, cancellation):
        cancellation.raise_if_cancelled()
        return {"prompt": "a small red ceramic vase", "original_prompt": request.prompt}

    def unload(self):
        self.unloaded = True


@pytest.fixture
def case(tmp_path):
    source = np.random.default_rng(42).integers(0, 128, (80, 100, 3), dtype=np.uint8)
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[30:45, 42:57] = 255
    mask[29, 43:56] = 32
    source_path, mask_path = tmp_path / "source.png", tmp_path / "mask.png"
    Image.fromarray(source).save(source_path)
    Image.fromarray(mask).save(mask_path)
    request = GenerativeFillRequest(source_path, mask_path, tmp_path / "result.png",
        patch_output=tmp_path / "patch.png", mask_output=tmp_path / "merge.png",
        fill=FillSettings(prompt="a vase", mask_overlap=4), mask_grow=8, context_padding=12, edge_softness=2)
    provider = SyntheticFill()
    registry = ProviderRegistry()
    registry.register(PROVIDER_ID, lambda: provider)
    return source, mask, request, provider, registry


def test_autocrop_grow_and_feather_never_change_outside_original_mask(case):
    source, mask, request, provider, registry = case
    result = InpaintService(registry).run(request)
    outside = mask == 0
    np.testing.assert_array_equal(np.asarray(Image.open(result.output))[outside], source[outside])
    np.testing.assert_array_equal(np.asarray(Image.open(request.patch_output))[outside], source[outside])
    assert np.asarray(Image.open(request.mask_output))[outside].max() == 0
    assert result.metadata["processing_roi"]["width"] < source.shape[1]
    assert provider.calls[0][1].sum() > mask.sum()
    assert np.asarray(Image.open(result.output))[37, 48, 0] > source[37, 48, 0]


def test_empty_mask_does_not_load_worker(case):
    source, _, request, provider, registry = case
    Image.new("L", (100, 80), 0).save(request.mask)
    result = InpaintService(registry).run(request)
    np.testing.assert_array_equal(np.asarray(Image.open(result.output)), source)
    assert not registry.loaded_ids
    assert not provider.calls


def test_default_generation_overlap_never_expands_output_permission(case):
    source, mask, request, provider, registry = case
    request = replace(request, fill=replace(request.fill, mask_overlap=32))
    result = InpaintService(registry).run(request)
    assert provider.calls[0][1].sum() > mask.sum()
    np.testing.assert_array_equal(np.asarray(Image.open(result.output))[mask == 0],
                                  source[mask == 0])


def test_explicit_blend_margin_allows_only_the_requested_extra_band(case):
    from kyven.inpaint.masks import expand_binary_mask

    source, mask, request, _, registry = case
    request = replace(request, fill=replace(request.fill, blend_margin=6), edge_softness=2)
    result = InpaintService(registry).run(request)
    expanded = expand_binary_mask(mask, 6) > 0
    output = np.array(Image.open(result.output))
    np.testing.assert_array_equal(output[~expanded], source[~expanded])
    assert np.any(output[(mask == 0) & expanded] != source[(mask == 0) & expanded])
    assert result.metadata["permission"] == "expanded"
    assert result.metadata["blend_margin"] == 6


def test_cache_changes_with_prompt_seed_and_solver_and_reuses_identical_result(case):
    _, _, request, provider, registry = case
    service = InpaintService(registry)
    first = service.run(request)
    assert service.run(request).metadata["cache_hit"]
    assert len(provider.calls) == 1
    for settings in (replace(request.fill, prompt="a bowl"), replace(request.fill, seed=2),
                     replace(request.fill, step_size=0.1)):
        assert replace(request, fill=settings).cache_key("1", "hash") != first.cache_key


@pytest.mark.parametrize("settings", [
    {"prompt": ""}, {"prompt": "vase", "working_size": 8192},
    {"prompt": "vase", "step_size": float("nan")},
    {"prompt": "vase", "steps": 5, "early_stop": 5},
    {"prompt": "vase", "thinking_steps": 0},
])
def test_invalid_settings_are_rejected(settings):
    with pytest.raises(KyvenError):
        FillSettings(**settings).validate()


def test_catalog_is_lazy_pinned_and_commercial(tmp_path):
    spec = ModelCatalog.builtin().get(PROVIDER_ID)
    assert spec.sha256 == manifest_checksum()
    assert spec.size_bytes == sum(item["size_bytes"] for item in manifest()["files"])
    assert spec.commercial_use and spec.supports_cpu
    assert all(len(item["sha256"]) == 64 and len(item["revision"]) == 40 for item in manifest()["files"])
    provider = ModelCatalog.builtin().registry(tmp_path, "cpu").get(PROVIDER_ID)
    assert provider._process is None
    assert provider.capabilities.minimum_vram_mb == 8192
    assert "GPL" in provider.capabilities.license_name
    assert "Klein" in provider.capabilities.display_name
    assert FillSettings().steps == 4
    assert working_dimensions(4000, 2000, 768) == (768, 384)


def test_saved_z_image_payload_migrates_without_mutating_input(case):
    _, _, request, _, _ = case
    payload = {"source": str(request.source), "mask": str(request.mask),
               "output": str(request.output), "model_id": "z-image-turbo-q4-lanpaint",
               "fill": {"prompt": "Remove the person", "steps": 9, "seed": 123}}
    converted = JobManager.generative_fill_request_from_payload(payload)
    assert converted.provider_id == PROVIDER_ID
    assert converted.fill.steps == 4
    assert converted.fill.seed == 123
    assert payload["fill"]["steps"] == 9


def test_provider_transport_cancellation_reaps_private_worker(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text("import sys,time\nfor line in sys.stdin:\n time.sleep(60)\n")
    provider = KleinLanPaintProvider(tmp_path, "cpu")
    provider.python = Path(sys.executable)
    provider.worker = worker
    provider.runtime = tmp_path / "runtime"
    provider._verified = True
    token = CancellationToken()
    timer = threading.Timer(0.2, token.cancel)
    timer.start()
    try:
        with pytest.raises(KyvenError) as exc:
            provider._exchange({"action": "enhance"}, token, 30)
        assert exc.value.code is ErrorCode.CANCELLED
        assert provider._process is None
    finally:
        timer.join()
        provider.unload()


def test_http_fill_and_prompt_improvement_share_scheduler(case, tmp_path):
    _, _, request, _, registry = case
    manager = JobManager(SegmentService(registry), inpaint_service=InpaintService(registry))
    server = KyvenServer(ServerConfig(token="x" * 32, models_dir=tmp_path, port=0),
                         manager, registry, ModelCatalog.builtin())
    server.start()
    try:
        client = KyvenClient(f"http://127.0.0.1:{server.port}", "x" * 32)
        payload = {"source": str(request.source), "mask": str(request.mask),
                   "output": str(request.output), "fill": asdict(request.fill), "context_padding": 12}
        for job_id in (client.submit_generative_fill(payload),
                       client.submit_prompt_enhance({"prompt": "red vase"})):
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                job = client.job(job_id)
                if job["status"] in {"succeeded", "failed", "cancelled"}:
                    break
                time.sleep(0.01)
            assert job["status"] == "succeeded", job
        assert job["result"]["original_prompt"] == "red vase"
        assert "ceramic" in job["result"]["prompt"]
    finally:
        server.shutdown()


def test_host_import_does_not_import_inference_packages():
    root = Path(__file__).parents[1]
    code = (f"import sys; sys.path.insert(0, {str(root / 'hosts/nuke')!r}); "
            "import kyven_nuke.generative_fill_node; "
            "assert not any(x in sys.modules for x in ('torch','diffusers','transformers','LanPaint'))")
    subprocess.run([sys.executable, "-c", code], check=True, timeout=15)


def test_prompt_request_validation_and_serialization():
    request = EnhancePromptRequest(prompt="красная ваза")
    request.validate()
    assert request.canonical()["prompt"] == "красная ваза"
    with pytest.raises(KyvenError):
        replace(request, temperature=float("nan")).validate()


def _worker_module():
    path = Path(__file__).parents[1] / "src/kyven/resources/generative_fill/worker.py"
    spec = importlib.util.spec_from_file_location("test_fill_worker_text", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("touch_bottom", [False, True])
def test_seamless_boundary_never_reintroduces_the_removed_subject(tmp_path, touch_bottom):
    pytest.importorskip("cv2")
    source = np.full((96, 96, 3), 120, np.uint8)
    mask = np.zeros((96, 96), np.uint8)
    mask[16:96 if touch_bottom else 80, 24:72] = 255
    source[mask > 0] = [5, 20, 200]
    generated = np.full_like(source, 140)
    for name, array in (("source", source), ("generated", generated), ("mask", mask)):
        Image.fromarray(array).save(tmp_path / f"{name}.png")
    payload = {name: str(tmp_path / f"{name}.png")
               for name in ("source", "generated", "mask", "output")}
    result = _worker_module().Runtime.stitch(payload)
    pixels = np.array(Image.open(payload["output"]))
    assert result["metadata"]["applied"]
    np.testing.assert_array_equal(pixels[mask == 0], source[mask == 0])
    assert np.max(abs(pixels[mask > 0].astype(float) - 120)) <= 3


def test_seamless_blend_preserves_an_enclosed_new_object(tmp_path):
    pytest.importorskip("cv2")
    source = np.full((96, 96, 3), 120, np.uint8)
    generated = source.copy()
    generated[35:60, 36:60] = [20, 60, 180]
    mask = np.zeros((96, 96), np.uint8)
    mask[16:80, 16:80] = 255
    for name, array in (("source", source), ("generated", generated), ("mask", mask)):
        Image.fromarray(array).save(tmp_path / f"{name}.png")
    payload = {name: str(tmp_path / f"{name}.png")
               for name in ("source", "generated", "mask", "output")}
    _worker_module().Runtime.stitch(payload)
    pixels = np.array(Image.open(payload["output"]))
    np.testing.assert_allclose(pixels[48, 48], [20, 60, 180], atol=3)


def test_english_decoding_blocks_cyrillic_chinese_and_partial_byte_tokens():
    class Tokenizer:
        eos_token_id = 6

        def get_vocab(self):
            return {"a": 0, " cat": 1, "ж": 2, "中文": 3, "\ufffd": 4, "<think>": 5, "<eos>": 6}

        def convert_tokens_to_string(self, tokens):
            return "".join(tokens)

        def convert_tokens_to_ids(self, token):
            return self.get_vocab().get(token)

    assert _worker_module().english_suppression_ids(Tokenizer(), 8) == [2, 3, 4, 5, 7]


def test_russian_generation_prompt_is_translated_without_improve_button():
    module = _worker_module()
    runtime = module.Runtime.__new__(module.Runtime)
    calls = []

    def enhance(settings):
        calls.append(settings)
        return {"prompt": "a woman wearing black sunglasses"}

    runtime.enhance = enhance
    assert runtime.image_prompt("женщина в чёрных очках", 42) == (
        "a woman wearing black sunglasses", True)
    assert calls[0]["temperature"] == 0.0
    assert "Translate" in calls[0]["system_prompt"]
    assert runtime.image_prompt("a red vase", 42) == ("a red vase", False)
    assert len(calls) == 1
