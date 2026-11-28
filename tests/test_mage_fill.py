from dataclasses import replace
from pathlib import Path

import numpy as np

from kyven.generative_fill.assets import manifest, manifest_checksum
from kyven.generative_fill.boundary import repair_boundary
from kyven.generative_fill.models import MAGE_PROVIDER_ID, FillSettings
from kyven.models.catalog import ModelCatalog
from kyven.server.jobs import JobManager


def test_mage_catalog_pins_complete_bundle_and_is_lazy(tmp_path):
    catalog = ModelCatalog.builtin()
    spec = catalog.get(MAGE_PROVIDER_ID)
    data = manifest(MAGE_PROVIDER_ID)
    assert spec.sha256 == manifest_checksum(MAGE_PROVIDER_ID)
    assert spec.size_bytes == sum(item["size_bytes"] for item in data["files"])
    assert spec.commercial_use and spec.redistribution
    assert spec.runtime == "mage-fill"
    assert all(len(item["sha256"]) == 64 and item["size_bytes"] > 0 for item in data["files"])
    provider = catalog.registry(tmp_path, "cpu").get(MAGE_PROVIDER_ID)
    assert provider._process is None
    assert provider.capabilities.provider_id == MAGE_PROVIDER_ID


def test_boundary_correction_removes_step_without_changing_subject_core():
    source = np.full((96, 96, 3), 100, np.uint8)
    generated = source.copy()
    generated[12:84, 12:84] = 140
    generated[40:56, 40:56] = [200, 15, 30]
    mask = np.zeros((96, 96), np.uint8)
    mask[12:84, 12:84] = 255
    result, metadata = repair_boundary(generated, source, mask, width=12)
    np.testing.assert_array_equal(result[mask == 0], source[mask == 0])
    np.testing.assert_array_equal(result[30:66, 30:66], generated[30:66, 30:66])
    assert np.mean(abs(result[12, 12:84].astype(float) - source[12, 12:84])) < 2
    assert metadata["applied"]


def test_boundary_empty_and_full_masks_do_not_create_false_edges():
    source = np.full((16, 16, 3), 10, np.uint8)
    generated = np.full_like(source, 200)
    empty, _ = repair_boundary(generated, source, np.zeros((16, 16), np.uint8))
    full, _ = repair_boundary(generated, source, np.full((16, 16), 255, np.uint8))
    np.testing.assert_array_equal(empty, source)
    np.testing.assert_array_equal(full, generated)


def test_boundary_repair_settings_validate_and_change_request_identity(tmp_path):
    from PIL import Image

    image = tmp_path / "source.png"
    Image.new("RGB", (32, 32)).save(image)
    request = JobManager.generative_fill_request_from_payload({
        "source": str(image), "mask": str(image), "output": str(tmp_path / "out.png"),
        "model_id": MAGE_PROVIDER_ID,
        "fill": {"prompt": "Add a vase", "seam_repair": "neural", "seam_width": 32},
    })
    assert request.provider_id == MAGE_PROVIDER_ID
    assert request.fill.steps == 4
    assert request.canonical() != replace(request, fill=replace(request.fill, seam_width=8)).canonical()
    assert request.canonical() != replace(request, fill=replace(request.fill, seam_repair="off")).canonical()
    import pytest

    from kyven.errors import KyvenError
    with pytest.raises(KyvenError):
        replace(FillSettings(prompt="edit"), seam_width=0).validate()


def test_mage_runtime_setup_is_separate_from_klein(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "hosts/nuke"))
    from kyven_nuke.runtime import _runtime_install_command, _runtime_probe
    assert "setup_mage_fill.py" in str(_runtime_install_command("mage-fill", Path("python")))
    assert "mage-fill" in _runtime_probe("mage-fill")


def test_mage_2048_keeps_full_source_and_limits_reference(tmp_path):
    import importlib.util

    from PIL import Image

    path = Path(__file__).parents[1] / "src/kyven/resources/generative_fill/mage_worker.py"
    spec = importlib.util.spec_from_file_location("mage_worker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime = module.Runtime.__new__(module.Runtime)
    runtime.cli = Path("sd-cli.exe")
    runtime.models = tmp_path
    runtime.device = "cuda"
    source = tmp_path / "source.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (2048, 1088)).save(source)
    Image.new("L", (2048, 1088), 255).save(mask)
    command = runtime.command(source, mask, tmp_path / "raw.png", "Remove a person", 42, 4)
    assert command[command.index("-i") + 1] == str(source)
    assert command[command.index("--mask") + 1] == str(mask)
    assert command[command.index("-W") + 1] == "2048"
    with Image.open(command[command.index("-r") + 1]) as reference:
        assert reference.size == (1024, 544)
    assert Image.open(source).size == (2048, 1088)
    FillSettings(prompt="Remove a person", working_size=2048).validate()


def test_2048_request_is_allowed_for_mage_only(tmp_path):
    import pytest
    from PIL import Image

    from kyven.errors import KyvenError
    from kyven.generative_fill.models import PROVIDER_ID, GenerativeFillRequest

    source = tmp_path / "source.png"
    Image.new("RGB", (16, 16)).save(source)
    request = GenerativeFillRequest(
        source=source, mask=source, output=tmp_path / "out.png",
        provider_id=MAGE_PROVIDER_ID, fill=FillSettings(prompt="Remove", working_size=2048),
    )
    request.validate()
    with pytest.raises(KyvenError, match="only for Mage"):
        replace(request, provider_id=PROVIDER_ID).validate()


def _mage_worker():
    import importlib.util
    path = Path(__file__).parents[1] / "src/kyven/resources/generative_fill/mage_worker.py"
    spec = importlib.util.spec_from_file_location("mage_memory_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mage_retries_exact_ram_graph_failure_without_changing_inputs(tmp_path, monkeypatch):
    from dataclasses import asdict

    from PIL import Image

    module = _mage_worker()
    runtime = module.Runtime.__new__(module.Runtime)
    runtime.cli, runtime.models, runtime.device = Path("sd-cli.exe"), tmp_path, "cuda"
    source, mask = tmp_path / "source.png", tmp_path / "mask.png"
    Image.new("RGB", (1024, 768), (75, 100, 125)).save(source)
    Image.new("L", (1024, 768), 255).save(mask)
    attempts, events = [], []
    failure = ("RAM graph capacity exceeded. Requested 56027248 bytes, 0 bytes available, "
               "3319795200 bytes resident, pending device commit is 2085214208 bytes")

    def run(command, timeout):
        attempts.append(command)
        assert "--disable-prefetch" in command
        if len(attempts) == 1:
            raise RuntimeError(failure)
        Image.open(command[command.index("-i") + 1]).save(command[command.index("-o") + 1])

    runtime.run = run
    monkeypatch.setattr(module, "emit", lambda event, **values: events.append(values))
    payload = {"source": str(source), "mask": str(mask), "output": str(tmp_path / "output.png"),
                   "settings": asdict(FillSettings(prompt="Remove a person", working_size=1024))}
    original = {p: p.read_bytes() for p in (source, mask)}
    result = runtime.fill(payload)
    assert len(attempts) == 2
    assert [c[c.index("-W") + 1] for c in attempts] == ["1024", "512"]
    assert result["metadata"]["oom_retried"]
    assert Image.open(payload["output"]).size == (1024, 768)
    assert all(p.read_bytes() == before for p, before in original.items())
    assert any("RAM/VRAM" in event.get("message", "") for event in events)


def test_mage_memory_failure_is_readable_and_retry_can_be_disabled(tmp_path):
    from dataclasses import asdict

    import pytest
    from PIL import Image

    module = _mage_worker()
    runtime = module.Runtime.__new__(module.Runtime)
    runtime.cli, runtime.models, runtime.device = Path("sd-cli.exe"), tmp_path, "cpu"
    source = tmp_path / "source.png"
    Image.new("RGB", (512, 512)).save(source)
    attempts = []

    def run(*args):
        attempts.append(args)
        raise RuntimeError("mage_flow segment failed during workspace allocation")

    runtime.run = run
    with pytest.raises(MemoryError, match="Processing Resolution 512 or 256"):
        runtime.fill({"source": str(source), "mask": str(source), "output": str(tmp_path / "out.png"),
                          "settings": asdict(FillSettings(prompt="Edit", working_size=512, oom_retry=False))})
    assert len(attempts) == 1
    assert not (tmp_path / "out.png").exists()
    assert not module.is_memory_error("model checksum verification failed")
    assert module.is_memory_error("CUDA out of memory")
    assert module.is_memory_error("RAM weight capacity exceeded")
