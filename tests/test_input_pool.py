"""Pool identity and selection tests without Nuke or SAM weights."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from test_input_cache import Node

from kyven import input_index
from kyven.cancellation import CancellationToken
from kyven.errors import KyvenError
from kyven.segment.models import PointPrompt
from kyven.segment.output import write_mask_png_atomic
from kyven.segment.providers.registry import ProviderRegistry
from kyven.segment.video import (
    VideoDirection,
    VideoSegmentRequest,
    VideoSegmentResult,
    VideoSegmentService,
)

sys.path.insert(0, str(Path(__file__).parents[1] / "hosts" / "nuke"))
from kyven_nuke import input_cache
from kyven_nuke import node as segment


def test_tracking_paths_do_not_depend_on_selected_range(tmp_path, monkeypatch):
    monkeypatch.setattr(segment, "_cache_work_root", lambda node: tmp_path / "_kyven_work")
    monkeypatch.setattr(segment, "_uses_artist_cache", lambda node: True)
    monkeypatch.setattr(segment, "_public_matte_pattern", lambda node: tmp_path / "result.%04d.png")
    first = segment._video_cache_paths(Node(), 1, 20)
    larger = segment._video_cache_paths(Node(), 1, 40)
    assert first[:2] == larger[:2]
    assert not (tmp_path / "_kyven_work/source/1_20").exists()


@pytest.mark.parametrize("fallback", [False, True])
def test_tracking_provider_sees_only_selected_pool_frames(tmp_path, monkeypatch, fallback):
    pool = tmp_path / "tracking_pool"
    pool.mkdir()
    for frame in range(1, 5):
        Image.new("RGB", (2, 2), (frame * 20, 0, 0)).save(pool / f"{frame:05d}.jpg")
    request = VideoSegmentRequest(pool, tmp_path / "matte.%04d.png", 2, 3, 2,
                                  VideoDirection.FORWARD, points=(PointPrompt(1, 1),),
                                  provider_id="test")
    key = request.cache_key("1", "model")
    Image.new("RGB", (2, 2), "blue").save(pool / "00001.jpg")
    assert request.cache_key("1", "model") == key
    captured = []
    class Provider:
        capabilities = SimpleNamespace(provider_version="1", model_checksum="model")
        def propagate_video(self, prepared, token):
            frames = sorted(prepared.frames_dir.glob("*.jpg"))
            assert len(frames) == 2
            if not fallback:
                assert frames[0].samefile(pool / "00002.jpg")
            assert [p.read_bytes() for p in frames] == [(pool / f"{n:05d}.jpg").read_bytes() for n in (2, 3)]
            captured.append(prepared.frames_dir)
            outputs = tuple(prepared.output_for_index(i) for i in range(2))
            for output in outputs:
                write_mask_png_atomic(output, np.ones((2, 2), dtype=np.uint8))
            return VideoSegmentResult(outputs, 2, 3, 2, VideoDirection.FORWARD, {})
    if fallback:
        monkeypatch.setattr("kyven.segment.video.os.link", lambda *args: (_ for _ in ()).throw(OSError("unsupported")))
    registry = ProviderRegistry()
    registry.register("test", Provider)
    service = VideoSegmentService(registry)
    service.run(request, CancellationToken())
    assert not captured[0].exists()
    assert len(list(pool.glob("*.jpg"))) == 4
    assert service.run(request).metadata["cache_hit"]
    (pool / "00003.jpg").unlink()
    with pytest.raises(KyvenError, match="missing frame 00003"):
        request.cache_key("1", "model")


def test_inventory_uses_one_database_and_preserves_legacy_receipts(tmp_path):
    import json
    node = Node()
    paths = [tmp_path / "_kyven_work/source" / f"{n}.png" for n in range(10)]
    paths[0].parent.mkdir(parents=True)
    with input_index.batch():
        for frame, path in enumerate(paths):
            path.write_bytes(b"pixels")
            input_index.write(path, {"sha256": input_cache.cached_file_sha256(path),
                                     "state": input_cache._state(node, frame), "frame": frame})
            assert input_cache.ready(node, path, frame)
    assert len(list(tmp_path.rglob("inputs.sqlite3"))) == 1
    assert not list(tmp_path.rglob("*.input.json"))
    input_index.forget(paths[0])
    input_cache._receipt(paths[0]).write_text(json.dumps({
        "sha256": input_cache.cached_file_sha256(paths[0]), "state": input_cache._state(node, 0), "frame": 0}))
    assert input_cache.ready(node, paths[0], 0)


def test_old_range_jpeg_is_adopted_without_nuke_export(tmp_path):
    from unittest.mock import Mock
    node = Node()
    legacy = tmp_path / "_kyven_work/source/1_20/00002.jpg"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"verified old jpeg")
    input_index.write(legacy, {"sha256": input_cache.cached_file_sha256(legacy),
                               "state": input_cache._state(node, 2), "frame": 2})
    target = legacy.parent.parent / "tracking_pool/00002.jpg"
    nuke = SimpleNamespace(execute=Mock(side_effect=AssertionError("must not render")))
    assert not input_cache.export_input(nuke, node, {}, 2, target)
    assert input_cache.ready(node, target, 2)
    assert target.read_bytes() == legacy.read_bytes()
    nuke.execute.assert_not_called()
