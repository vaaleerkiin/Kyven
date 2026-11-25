"""Experimental MOSS remains opt-in and uses the existing masked job contract."""
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from kyven.errors import KyvenError
from kyven.generative_fill.assets import manifest, manifest_checksum
from kyven.generative_fill.models import (
    MOSS_PROVIDER_ID,
    PROVIDER_ID,
    EnhancePromptRequest,
    FillSettings,
    GenerativeFillRequest,
)
from kyven.generative_fill.moss_provider import MossLanPaintProvider
from kyven.generative_fill.provider import KleinLanPaintProvider
from kyven.models.catalog import ModelCatalog
from kyven.server.jobs import JobManager


def test_moss_is_an_additional_verified_experimental_bundle(tmp_path):
    catalog = ModelCatalog.builtin()
    spec = catalog.get(MOSS_PROVIDER_ID)
    data = manifest(MOSS_PROVIDER_ID)
    assert spec.sha256 == manifest_checksum(MOSS_PROVIDER_ID)
    assert spec.size_bytes == sum(f["size_bytes"] for f in data["files"])
    assert spec.size_bytes < 14_000_000_000
    assert spec.runtime == "moss-fill" and spec.license_acceptance_required
    assert not spec.commercial_use and not spec.redistribution
    assert catalog.get(PROVIDER_ID).runtime == "generative-fill"
    for item in data["files"]:
        assert len(item["sha256"]) == 64
        assert not Path(item["path"]).is_absolute() and ".." not in Path(item["path"]).parts


def test_model_choice_and_prompt_enhancement_are_routed_independently(tmp_path):
    from PIL import Image
    for name in ("source.png", "mask.png"):
        Image.new("RGB", (32, 32)).save(tmp_path/name)
    request = JobManager.generative_fill_request_from_payload({
        "source": str(tmp_path/"source.png"), "mask": str(tmp_path/"mask.png"),
        "output": str(tmp_path/"output.png"), "model_id": MOSS_PROVIDER_ID,
        "fill": {"prompt": "Remove the person"}})
    assert request.provider_id == MOSS_PROVIDER_ID
    assert request.fill.steps == 20
    EnhancePromptRequest("Remove the person", provider_id=MOSS_PROVIDER_ID).validate()
    with pytest.raises(KyvenError):
        replace(request, preserve_input_mask=False).validate()
    assert GenerativeFillRequest.__dataclass_fields__["provider_id"].default == PROVIDER_ID


def test_moss_and_klein_keep_separate_workers_and_dependencies(tmp_path):
    moss, klein = MossLanPaintProvider(tmp_path), KleinLanPaintProvider(tmp_path)
    assert moss.worker.name == "moss_worker.py" and klein.worker.name == "worker.py"
    assert moss.runtime.name == "moss-fill" and klein.runtime.name == "generative-fill"
    assert moss.capabilities.supports_cpu
    assert moss.capabilities.minimum_vram_mb == 8192
    assert moss.capabilities.model_checksum != klein.capabilities.model_checksum
    assert FillSettings(moss_guidance=4.0).moss_guidance == 4.0


def test_moss_guidance_does_not_invalidate_klein_cache(tmp_path):
    klein = GenerativeFillRequest(tmp_path/"source.png", tmp_path/"mask.png", tmp_path/"out.png",
                                 fill=FillSettings(prompt="Remove the person"))
    changed = replace(klein, fill=replace(klein.fill, moss_guidance=7.0))
    assert klein.canonical() == changed.canonical()
    moss = replace(klein, provider_id=MOSS_PROVIDER_ID)
    assert moss.canonical() != replace(moss, fill=changed.fill).canonical()


@pytest.mark.parametrize("accepted", [False, True])
def test_model_manager_shows_unconfirmed_edit_license_before_install(tmp_path, monkeypatch, accepted):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/"hosts/nuke"))
    from kyven_nuke import model_manager
    model = ModelCatalog.builtin().get(MOSS_PROVIDER_ID).snapshot(tmp_path)
    calls, messages = [], []

    class Panel:
        def addEnumerationPulldown(self, name, values):
            pass

        def addNotepad(self, *args):
            pass

        def show(self):
            return True

        def value(self, name):
            return "Install Selected" if name == "Action" else model_manager._choice_label(model)

    nuke = SimpleNamespace(Panel=lambda title: Panel(),
        ask=lambda message: messages.append(message) or accepted,
        message=lambda message: pytest.fail(message))
    monkeypatch.setattr(model_manager, "_nuke", lambda: nuke)
    monkeypatch.setattr(model_manager, "_manager_models", lambda: ([model], True))
    monkeypatch.setattr(model_manager, "_start_operation", lambda action, item:
                        calls.append((action, item["model_id"])))
    model_manager.show_model_manager()
    assert len(messages) == 1 and "not confirmed" in messages[0]
    assert calls == ([("download", MOSS_PROVIDER_ID)] if accepted else [])


def test_moss_isolated_worker_protocol_preserves_unicode(tmp_path):
    resources = Path(__file__).parents[1]/"src/kyven/resources/generative_fill"
    script = tmp_path/"echo.py"
    script.write_text("import sys\nsys.path.insert(0, "+ascii(str(resources))+''')
import moss_worker
class Echo:
    def fill(self, payload):
        return {"metadata": {"echo": payload["settings"]["prompt"]}}
moss_worker.Runtime = lambda *args: Echo()
moss_worker.main()
''', encoding="utf-8")
    prompt = "\u0443\u0431\u0435\u0440\u0438 \u0447\u0435\u043b\u043e\u0432\u0435\u043a\u0430"
    result = subprocess.run([sys.executable, "-I", str(script), "--models", str(tmp_path)],
        input=json.dumps({"action": "fill", "settings": {"prompt": prompt}}, ensure_ascii=False)+"\n",
        capture_output=True, encoding="utf-8", timeout=20, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["metadata"]["echo"] == prompt
