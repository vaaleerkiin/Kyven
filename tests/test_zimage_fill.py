"""Native learned inpainting stays isolated and preserves existing providers/caches."""
import hashlib
import io
import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from kyven.generative_fill import assets
from kyven.generative_fill.models import (
    PROVIDER_ID,
    ZIMAGE_PROVIDER_ID,
    FillSettings,
    GenerativeFillRequest,
)
from kyven.generative_fill.zimage_provider import ZImageInpaintProvider
from kyven.models.catalog import ModelCatalog
from kyven.server.jobs import JobManager


def test_native_bundle_has_trained_lite_control_and_correct_vae():
    data = assets.manifest(ZIMAGE_PROVIDER_ID)
    spec = ModelCatalog.builtin().get(ZIMAGE_PROVIDER_ID)
    assert spec.sha256 == assets.manifest_checksum(ZIMAGE_PROVIDER_ID)
    assert spec.size_bytes == sum(f["size_bytes"] for f in data["files"])
    assert spec.commercial_use and spec.redistribution and spec.runtime == "zimage-fill"
    files = {f["path"]: f for f in data["files"]}
    assert "lite" in files["controlnet/model.safetensors"]["url"]
    assert files["vae/config.json"]["repo"] == "Tongyi-MAI/Z-Image"
    assert files["vae/diffusion_pytorch_model.safetensors"]["repo"] == "Tongyi-MAI/Z-Image"
    config = json.loads((assets.manifest_path().parent/"zimage_control_config.json").read_text())
    assert config["control_in_dim"] == 33
    assert config["control_layers_places"] == [0, 10, 20]


def test_native_request_defaults_and_relevant_cache_fields(tmp_path):
    from PIL import Image
    for name in ("source.png", "mask.png"):
        Image.new("RGB", (32, 32)).save(tmp_path/name)
    request = JobManager.generative_fill_request_from_payload({"source": str(tmp_path/"source.png"),
        "mask": str(tmp_path/"mask.png"), "output": str(tmp_path/"out.png"),
        "model_id": ZIMAGE_PROVIDER_ID, "fill": {"prompt": "Remove the person"}})
    assert request.fill.steps == 28 and request.fill.working_size == 768
    assert request.fill.mask_overlap == 64 and request.preserve_input_mask
    assert request.canonical() == replace(request, fill=replace(request.fill, thinking_steps=8)).canonical()
    assert request.canonical() != replace(request, fill=replace(request.fill, zimage_control_scale=0.7)).canonical()
    klein = GenerativeFillRequest(tmp_path/"a", tmp_path/"b", tmp_path/"c", fill=FillSettings(prompt="Remove"))
    assert klein.provider_id == PROVIDER_ID
    assert klein.canonical() == replace(klein, fill=replace(klein.fill, zimage_guidance=7)).canonical()


def test_native_transport_is_lazy_and_separate(tmp_path):
    provider = ZImageInpaintProvider(tmp_path)
    assert provider.worker.name == "zimage_worker.py"
    assert provider.runtime.name == "zimage-fill"
    assert provider.capabilities.supports_cpu and provider.capabilities.minimum_vram_mb == 8192
    result = subprocess.run([sys.executable, "-I", "-c",
        "import sys;from kyven.generative_fill.zimage_provider import ZImageInpaintProvider;assert 'torch' not in sys.modules"],
        capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 0, result.stderr


def test_matching_shared_weights_reuse_a_verified_hard_link(tmp_path, monkeypatch):
    data = b"verified shared Qwen weights"
    source = tmp_path/"klein/text_encoder/model.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(data)
    monkeypatch.setattr(assets, "manifest", lambda: {"files": [{"path": "text_encoder/model.bin",
        "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "url": "https://unused.invalid/model"}]})
    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda *a, **k: pytest.fail("Shared weights must not download"))
    root = assets.download_bundle(tmp_path/"native")
    assert (root/"text_encoder/model.bin").read_bytes() == data
    assert os.path.samefile(source, root/"text_encoder/model.bin")


def test_same_size_corrupt_peer_is_never_reused(tmp_path, monkeypatch):
    data, bad = b"good", b"evil"
    source = tmp_path/"peer/weights.bin"
    source.parent.mkdir()
    source.write_bytes(bad)
    monkeypatch.setattr(assets, "manifest", lambda: {"files": [{"path": "weights.bin", "size_bytes": 4,
        "sha256": hashlib.sha256(data).hexdigest(), "url": "https://test.invalid/weights"}]})

    class Response(io.BytesIO):
        status = 200

    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda *a, **k: Response(data))
    root = assets.download_bundle(tmp_path/"native")
    assert (root/"weights.bin").read_bytes() == data and source.read_bytes() == bad


def test_native_worker_unicode_protocol(tmp_path):
    resources = assets.manifest_path().parent
    script = tmp_path/"echo.py"
    script.write_text("import sys\nsys.path.insert(0, "+ascii(str(resources))+''')
import zimage_worker
class Echo:
    def fill(self, payload): return {"metadata": {"echo": payload["settings"]["prompt"]}}
zimage_worker.Runtime = lambda *args: Echo()
zimage_worker.main()
''', encoding="utf-8")
    prompt = "\u0443\u0431\u0435\u0440\u0438 \u0447\u0435\u043b\u043e\u0432\u0435\u043a\u0430"
    result = subprocess.run([sys.executable, "-I", str(script), "--models", str(tmp_path)],
        input=json.dumps({"action": "fill", "settings": {"prompt": prompt}}, ensure_ascii=False)+"\n",
        capture_output=True, encoding="utf-8", timeout=20, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["metadata"]["echo"] == prompt


def test_native_planner_keeps_english_edit_instruction_separate_from_target_caption(monkeypatch):
    monkeypatch.syspath_prepend(str(assets.manifest_path().parent))
    import zimage_worker
    monkeypatch.setattr(zimage_worker, "release", lambda: None)
    monkeypatch.setattr(zimage_worker, "emit", lambda *a, **k: None)
    runtime = zimage_worker.Runtime.__new__(zimage_worker.Runtime)
    plan = {"instruction": "Remove the person and restore the background.",
        "caption": "A continuation of the surrounding storefront and pavement.",
        "negative": "person, people", "operation": "remove"}
    calls = []
    runtime.enhance = lambda *a, **k: calls.append(True) or {"prompt": json.dumps(plan)}
    runtime.image_prompt = lambda *a: (plan["instruction"], True)
    runtime.plans = {}
    runtime.conditions = {(plan["caption"], plan["negative"]): ["positive", "negative"]}
    runtime.lm = runtime.tokenizer = object()
    original = "\u0443\u0431\u0435\u0440\u0438 \u0447\u0435\u043b\u043e\u0432\u0435\u043a\u0430"
    prepared, embeddings, hit = runtime.prepare_text(original, "")
    assert prepared["instruction"].startswith("Remove") and "person" not in prepared["caption"]
    assert hit and embeddings == ["positive", "negative"]
    runtime.prepare_text(original, "")
    assert calls == [True] and runtime.lm is None and runtime.tokenizer is None


def test_addition_never_automatically_negates_the_requested_subject(monkeypatch):
    monkeypatch.syspath_prepend(str(assets.manifest_path().parent))
    import zimage_worker
    monkeypatch.setattr(zimage_worker, "release", lambda: None)
    monkeypatch.setattr(zimage_worker, "emit", lambda *a, **k: None)
    runtime = zimage_worker.Runtime.__new__(zimage_worker.Runtime)
    plan = {"instruction": "Add a red vase on the pavement.", "caption": "A red vase standing on the pavement.",
        "negative": "No red vase", "operation": "add"}
    runtime.enhance = lambda *a, **k: {"prompt": json.dumps(plan)}
    runtime.plans = {}
    caption = plan["caption"]+" Integrated into the surrounding scene, matching its perspective, lighting and focus."
    runtime.conditions = {(caption, ""): ["positive", "negative"]}
    runtime.lm = runtime.tokenizer = None
    prepared, _, hit = runtime.prepare_text("Add a red vase", "")
    assert hit and prepared["negative"] == "" and "red vase" in prepared["caption"]


@pytest.mark.parametrize("mode,native,expected", [("auto", True, True), ("off", True, False),
                                                 ("on", True, True), ("auto", False, False)])
def test_native_auto_blending_also_handles_additions(tmp_path, mode, native, expected):
    import numpy as np
    from PIL import Image

    from kyven.inpaint.models import InpaintCapabilities, InpaintPrediction
    from kyven.inpaint.service import InpaintService
    from kyven.segment.providers.registry import ProviderRegistry
    source = np.full((64, 64, 3), 70, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:44, 20:44] = 255
    Image.fromarray(source).save(tmp_path/"source.png")
    Image.fromarray(mask).save(tmp_path/"mask.png")
    model_id = ZIMAGE_PROVIDER_ID if native else PROVIDER_ID

    class Stub:
        capabilities = InpaintCapabilities(model_id, "Synthetic", "1", "hash", "Apache-2.0", "", True, 8192)
        auto_stitch_all_edits = native
        stitches = 0

        def predict(self, request, token):
            pixels = np.array(Image.open(request.source).convert("RGB"))
            return InpaintPrediction(np.full_like(pixels, 120), {"edit_operation": "add"})

        def stitch(self, generated, original, selection, token):
            self.stitches += 1
            return generated, {"applied": True}

        def unload(self):
            pass

    provider = Stub()
    registry = ProviderRegistry()
    registry.register(model_id, lambda: provider)
    request = GenerativeFillRequest(tmp_path/"source.png", tmp_path/"mask.png", tmp_path/"out.png",
        provider_id=model_id, mask_grow=0, context_padding=8,
        fill=FillSettings(prompt="Add a vase", color_match=0, mask_overlap=0, seamless_blend=mode))
    result = InpaintService(registry).run(request)
    assert provider.stitches == int(expected)
    assert np.array_equal(np.array(Image.open(result.output))[mask == 0], source[mask == 0])


@pytest.mark.parametrize("radius", [0, 1, 4, 64])
def test_large_inference_overlap_matches_binary_dilation(radius):
    import numpy as np
    from PIL import Image, ImageFilter

    from kyven.inpaint.masks import expand_binary_mask
    mask = np.zeros((31, 47), dtype=np.uint8)
    mask[0, 0] = mask[-1, -1] = 255
    mask[12:15, 19:23] = 255
    expected = np.asarray(Image.fromarray(mask).filter(ImageFilter.MaxFilter(2*radius+1)))
    np.testing.assert_array_equal(expand_binary_mask(mask, radius), expected)
