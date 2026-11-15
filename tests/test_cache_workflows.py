"""Exercise complete service cache workflows without Nuke or model weights."""
from dataclasses import replace
from unittest.mock import Mock

import pytest
from PIL import Image
from test_inpaint import FakeInpaintProvider
from test_refine import SyntheticRefinementProvider
from test_segment_service import SyntheticProvider

from kyven.inpaint.models import InpaintRequest
from kyven.inpaint.service import InpaintService
from kyven.refine.models import RefineRequest
from kyven.refine.service import RefineService
from kyven.segment.models import PointPrompt, SegmentRequest
from kyven.segment.providers.registry import ProviderRegistry
from kyven.segment.service import SegmentService


@pytest.mark.parametrize("kind", ["segment", "refine", "inpaint"])
def test_sequence_reuse_repair_revision_restart_and_input_invalidation(tmp_path, kind):
    source, mask = tmp_path / "source.png", tmp_path / "mask.png"
    Image.new("RGB", (2, 2), "white").save(source)
    Image.new("L", (2, 2), 255).save(mask)
    if kind == "segment":
        provider, service_type = SyntheticProvider(), SegmentService
        base = SegmentRequest(source, tmp_path / "1001.png", provider_id="test",
                              points=(PointPrompt(1, 1),), fill_holes=True)
        settings = {"max_hole_area": 10}
    elif kind == "refine":
        provider, service_type = SyntheticRefinementProvider(), RefineService
        base = RefineRequest(source, mask, tmp_path / "1001.png", provider_id="test")
        settings = {"foreground_radius": 2}
    else:
        provider, service_type = FakeInpaintProvider(), InpaintService
        base = InpaintRequest(source, mask, tmp_path / "1001.png", provider_id="test",
                              crop_mode="full")
        settings = {"edge_softness": 1.0}
    provider.predict = Mock(wraps=provider.predict)
    registry = ProviderRegistry()
    registry.register("test", lambda: provider)
    service = service_type(registry)
    requests = [replace(base, output=tmp_path / f"{frame}.png") for frame in range(1001, 1004)]
    for request in requests:
        assert not service.run(request).metadata["cache_hit"]
    assert provider.predict.call_count == 3
    timestamps = [r.output.stat().st_mtime_ns for r in requests]
    original = [r.output.read_bytes() for r in requests]
    for request in requests:
        assert service.run(request).metadata["cache_hit"]
    assert [r.output.stat().st_mtime_ns for r in requests] == timestamps
    assert provider.predict.call_count == 3
    requests[1].output.unlink()
    assert service.run(requests[1]).metadata["cache_hit"]
    assert requests[1].output.read_bytes() == original[1]
    assert requests[0].output.stat().st_mtime_ns == timestamps[0]
    assert requests[2].output.stat().st_mtime_ns == timestamps[2]
    assert provider.predict.call_count == 3
    assert not service.run(replace(requests[0], **settings)).metadata["cache_hit"]
    assert provider.predict.call_count == 4
    assert service.run(requests[0]).metadata["cache_hit"]
    assert requests[0].output.read_bytes() == original[0]
    # A new service instance must recover using disk state alone.
    service = service_type(registry)
    assert service.run(requests[0]).metadata["cache_hit"]
    assert provider.predict.call_count == 4
    Image.new("RGB", (2, 2), "black").save(source)
    assert not service.run(requests[0]).metadata["cache_hit"]
    assert provider.predict.call_count == 5
