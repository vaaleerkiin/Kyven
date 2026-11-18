"""Input preparation without launching Nuke, a server, or model providers."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "hosts" / "nuke"))

from kyven_nuke import input_cache


class Knob:
    def __init__(self, value):
        self.current = value

    def value(self):
        return self.current

    def setValue(self, value):
        self.current = value

    def setEnabled(self, value):
        self.enabled = value


class Node:
    def __init__(self):
        self.values = {"mask_grow": Knob(3)}

    def knobs(self):
        return self.values

    def __getitem__(self, name):
        return self.values[name]

    def fullName(self):
        return "TestCache"


@pytest.fixture
def context(tmp_path):
    node = Node()
    writer = {"file": Knob("")}
    def render(writer, first, last):
        Path(writer["file"].value()).write_bytes(f"pixels:{first}".encode())
    nuke = SimpleNamespace(execute=Mock(side_effect=render))
    return node, writer, nuke, tmp_path


def test_prepare_reuses_files_and_only_exports_missing_or_corrupt_inputs(context):
    node, writer, nuke, root = context
    paths = [root / f"source.{frame}.png" for frame in range(1, 4)]
    for frame, path in enumerate(paths, 1):
        assert input_cache.export_input(nuke, node, writer, frame, path)
    assert nuke.execute.call_count == 3
    timestamps = [p.stat().st_mtime_ns for p in paths]
    for frame, path in enumerate(paths, 1):
        assert not input_cache.export_input(nuke, node, writer, frame, path)
    assert nuke.execute.call_count == 3
    assert [p.stat().st_mtime_ns for p in paths] == timestamps
    paths[1].unlink()
    paths[2].write_bytes(b"changed")
    for frame, path in enumerate(paths, 1):
        input_cache.export_input(nuke, node, writer, frame, path)
    assert nuke.execute.call_count == 5
    assert paths[0].stat().st_mtime_ns == timestamps[0]


def test_process_only_checks_snapshot_and_never_exports(context):
    node, writer, nuke, root = context
    path = root / "source.1.png"
    writer["file"].setValue(str(path))
    with pytest.raises(RuntimeError, match="Cache Frames"):
        input_cache.use_input(nuke, node, writer, 1)
    assert nuke.execute.call_count == 0
    input_cache.export_input(nuke, node, writer, 1, path)
    timestamp = path.stat().st_mtime_ns
    input_cache.use_input(nuke, node, writer, 1)
    assert nuke.execute.call_count == 1
    assert path.stat().st_mtime_ns == timestamp
    node["mask_grow"].setValue(5)
    with pytest.raises(RuntimeError, match="outdated"):
        input_cache.use_input(nuke, node, writer, 1)
    assert nuke.execute.call_count == 1


def test_refresh_exports_again_and_failed_refresh_invalidates_snapshot(context):
    node, writer, nuke, root = context
    path = root / "source.png"
    input_cache.export_input(nuke, node, writer, 1, path)
    input_cache.export_input(nuke, node, writer, 1, path, refresh=True)
    assert nuke.execute.call_count == 2
    nuke.execute.side_effect = RuntimeError("upstream read failed")
    with pytest.raises(RuntimeError, match="upstream read failed"):
        input_cache.export_input(nuke, node, writer, 1, path, refresh=True)
    assert not input_cache.ready(node, path)


def test_animated_mask_settings_are_verified_at_the_cached_frame(context):
    node, writer, nuke, root = context
    knob = node["mask_grow"]
    knob.valueAt = lambda frame: frame * 2
    path = root / "source.1002.png"
    input_cache.export_input(nuke, node, writer, 1002, path)
    # Moving the timeline must not invalidate the recipe recorded for 1002.
    knob.setValue(999)
    assert input_cache.ready(node, path, 1002)
    assert not input_cache.ready(node, path, 1003)
    knob.valueAt = lambda frame: frame * 3
    assert not input_cache.ready(node, path, 1002)


def test_corrupt_index_can_be_refreshed_without_deleting_images(context):
    node, writer, nuke, root = context
    path = root / 'source.tif'
    input_cache.export_input(nuke, node, writer, 1, path)
    from kyven.cache_layout import work_root
    (work_root(path) / 'inputs.sqlite3').write_bytes(b'broken database')
    assert not input_cache.ready(node, path, 1)
    input_cache.export_input(nuke, node, writer, 1, path, refresh=True)
    assert input_cache.ready(node, path, 1)
    assert nuke.execute.call_count == 2


@pytest.mark.parametrize('kind', ['refine', 'inpaint'])
def test_alpha_only_upgrade_invalidates_legacy_input_snapshots(context, kind):
    node, writer, nuke, root = context
    path = root / 'mask.tif'
    input_cache.export_input(nuke, node, writer, 1, path)
    node.values['kyven_kind'] = Knob(kind)
    assert not input_cache.ready(node, path, 1)
    input_cache.export_input(nuke, node, writer, 1, path)
    assert input_cache.ready(node, path, 1)


def test_generated_vs_manual_guidance_requires_new_snapshot(context):
    node, writer, nuke, root = context
    node.values['kyven_kind'] = Knob('refine')
    node.values['generate_trimap'] = Knob(True)
    path = root/'mask.tif'
    input_cache.export_input(nuke, node, writer, 1, path)
    node['generate_trimap'].setValue(False)
    assert not input_cache.ready(node, path, 1)


def test_cattery_recipe_invalidates_positive_alpha_snapshots(context, monkeypatch):
    node, writer, nuke, root = context
    node.values['kyven_kind'] = Knob('refine')
    path = root / 'mask.tif'
    current_state = input_cache._state

    def old_state(node, frame):
        state = current_state(node, frame)
        state['mask_input_contract'] = 'alpha-only-positive-v1'
        return state

    with monkeypatch.context() as patch:
        patch.setattr(input_cache, '_state', old_state)
        input_cache.export_input(nuke, node, writer, 1, path)
    assert not input_cache.ready(node, path, 1)
    input_cache.export_input(nuke, node, writer, 1, path)
    assert input_cache.ready(node, path, 1)
    assert nuke.execute.call_count == 2
