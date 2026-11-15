"""Run Cache Frames and actual Process entry points using a fake Nuke API."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "hosts" / "nuke"))

from kyven_nuke import dlss_node as dlss
from kyven_nuke import inpaint_node as inpaint
from kyven_nuke import input_cache
from kyven_nuke import node as segment
from kyven_nuke import refine_node as refine
from test_input_cache import Knob, Node


class AdapterNode(Node):
    def __init__(self, kind):
        super().__init__()
        self.values = {name: Knob(value) for name, value in {
            "kyven_kind": kind, "range_first": 1001, "range_last": 1003,
            "first_frame": 1001, "last_frame": 1003, "live_mode": True,
            "input_cache_scope": "Frame Range", "kyven_status": "", "kyven_live_frame": 0,
        }.items()}
        self.source = SimpleNamespace(width=lambda: 2, height=lambda: 2)

    def input(self, index):
        return self.source


class FakeNuke:
    def __init__(self, node):
        self.node = node
        self.current = 1001
        self.message = Mock()
        self.execute = Mock(side_effect=self.render)
        self.ProgressTask = Mock(return_value=SimpleNamespace(
            isCancelled=lambda: False, setMessage=Mock(), setProgress=Mock()))

    def thisNode(self):
        return self.node

    def frame(self, value=None):
        if value is not None:
            self.current = value
        return self.current

    def render(self, writer, first, last):
        pattern = str(writer["file"].value())
        for frame in range(first, last + 1):
            path = Path(pattern % frame if "%" in pattern else pattern)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"input {frame}".encode())


@pytest.fixture(params=["segment", "refine", "inpaint", "dlss"])
def adapter_context(request, tmp_path, monkeypatch):
    kind = request.param
    module = {"segment": segment, "refine": refine, "inpaint": inpaint, "dlss": dlss}[kind]
    node = AdapterNode(kind)
    nuke = FakeNuke(node)
    writers = {}
    def inside(current, name):
        if name == "KyvenInpaintCombined":
            return None
        return writers.setdefault(name, {"file": Knob(""), "which": Knob(0)})
    starts = Mock()
    monkeypatch.setattr(module.threading, "Thread", lambda **kwargs: SimpleNamespace(start=starts))
    for current in {module, segment}:
        monkeypatch.setattr(current, "_nuke", lambda: nuke)
        monkeypatch.setattr(current, "_inside", inside)
        for name in ("_set_busy", "_start_progress", "_finish_progress", "_update_progress"):
            monkeypatch.setattr(current, name, Mock())
        monkeypatch.setattr(current, "_begin_node_operation", lambda node: True)
    monkeypatch.setattr(segment, "_set_status", lambda name, message: node["kyven_status"].setValue(message))
    if module is not segment:
        monkeypatch.setattr(module, "_set_status", segment._set_status)
        if kind != "dlss":
            monkeypatch.setattr(module, "_ensure_input_order", Mock())
            monkeypatch.setattr(module, "_export_cancelled", lambda name: False)
    monkeypatch.setattr(module, "_progress_cancelled", lambda name: False)
    def patterns():
        return ((tmp_path / "video/%05d.jpg" if kind == "segment" else tmp_path / "source.%04d.tif"), tmp_path / "mask.%04d.tif",
                tmp_path / "result.%04d.png", tmp_path / "trimap.%04d.png")
    if kind == "segment":
        monkeypatch.setattr(segment, "_has_prompts", lambda node: True)
        monkeypatch.setattr(segment, "_range_cancelled", lambda name: False)
        monkeypatch.setattr(segment, "_cache_patterns", lambda node: patterns()[:3])
        monkeypatch.setattr(segment, "_cache_paths", lambda node, frame:
                            tuple(Path(str(p) % frame) for p in patterns()[:3]))
        monkeypatch.setattr(segment, "_video_cache_paths", lambda node, first, last:
                            (tmp_path / "video", tmp_path / "video/%05d.jpg", *patterns()[1:3]))
        monkeypatch.setattr(segment, "_payload_for_paths", lambda *args: {})
    elif kind == "refine":
        monkeypatch.setattr(refine, "_cache_patterns", lambda node: patterns())
        monkeypatch.setattr(refine, "_cache_paths", lambda node, frame:
                            tuple(Path(str(p) % frame) for p in patterns()))
        monkeypatch.setattr(refine, "_configure_export_writers", Mock())
        monkeypatch.setattr(refine, "_payload", lambda *args: {})
    elif kind == "inpaint":
        monkeypatch.setattr(inpaint, "_configure_color_io", Mock())
        monkeypatch.setattr(inpaint, "_cache_work_root", lambda node: tmp_path)
        monkeypatch.setattr(inpaint, "_uses_artist_cache", lambda node: True)
        monkeypatch.setattr(inpaint, "_inpaint_public_pattern", lambda node: patterns()[2])
        # The default argument in the Process function binds the original builder.
        monkeypatch.setattr(inpaint, "_payload", lambda *args: {})
    else:
        worker = tmp_path / "worker.exe"
        worker.write_bytes(b"test worker identity")
        monkeypatch.setattr(dlss, "_request_settings", lambda node:
                            {"worker": str(worker), "temporal": False, "mask_source": "none"})
        monkeypatch.setattr(dlss, "_dlss_cache_root", lambda node: tmp_path)
        monkeypatch.setattr(dlss, "_public_pattern", lambda node: patterns()[2])
        monkeypatch.setattr(dlss, "_settings_signature", lambda node: "settings")
        monkeypatch.setattr(dlss, "_signature", lambda node: "frame")
        monkeypatch.setattr(dlss, "_identity", lambda node: "prepared-input-test")
    def process(current=False):
        if kind == "inpaint":
            function = (inpaint.process_current_frame_for_node if current
                        else inpaint.process_frame_range_for_node)
            function(node, payload_builder=lambda *args: {})
        elif current:
            module.process_current_frame(node)
        else:
            module.process_frame_range()
    return node, nuke, starts, process, tmp_path


def test_cache_then_current_and_range_process_do_not_export_or_rewrite(adapter_context):
    node, nuke, starts, process, root = adapter_context
    input_cache.cache_frames()
    assert "ready 1001-1003" in node["kyven_status"].value()
    assert nuke.execute.call_count > 0
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in root.rglob("*") if p.is_file()}
    nuke.execute.reset_mock()
    process()
    assert starts.call_count == 1, node["kyven_status"].value()
    process(current=True)
    assert starts.call_count == 2, node["kyven_status"].value()
    nuke.execute.assert_not_called()
    assert {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in before} == before
    input_cache.cache_frames()
    nuke.execute.assert_not_called()


def test_range_process_repairs_only_missing_middle_frame(adapter_context):
    _node, nuke, starts, process, root = adapter_context
    input_cache.cache_frames()
    source = next(p for p in root.rglob("*1002*") if p.is_file() and p.suffix != ".json")
    before = {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file() and p != source
              and p != input_cache._receipt(source) and not p.name.startswith("inputs.sqlite3")}
    source.unlink()
    nuke.execute.reset_mock()
    process()
    assert starts.call_count == 1
    assert nuke.execute.call_count == 1
    assert all(p.stat().st_mtime_ns == stamp for p, stamp in before.items())
    assert source.is_file()
    nuke.execute.reset_mock()
    process()
    assert starts.call_count == 2
    nuke.execute.assert_not_called()


def test_first_range_process_prepares_inputs_then_starts_work(adapter_context):
    node, nuke, starts, process, _root = adapter_context
    expected = len(input_cache._specs(node, 1001, 1001, 1003)) * 3
    process()
    assert starts.call_count == 1, node["kyven_status"].value()
    assert nuke.execute.call_count == expected
    nuke.execute.reset_mock()
    process()
    assert starts.call_count == 2
    nuke.execute.assert_not_called()


@pytest.mark.parametrize("damage", ["missing", "corrupt", "missing_receipt"])
def test_each_invalid_input_is_prepared_before_process(adapter_context, damage):
    node, nuke, starts, process, _root = adapter_context
    input_cache.cache_frames()
    specs = input_cache._specs(node, 1002, 1001, 1003)
    for writer, path in specs:
        starts.reset_mock()
        nuke.execute.reset_mock()
        if damage == "missing":
            path.unlink()
        elif damage == "corrupt":
            path.write_bytes(b"invalidated snapshot")
        else:
            input_cache.invalidate(path)
        process()
        assert starts.call_count == 1
        assert nuke.execute.call_count == 1
        assert input_cache.ready(node, path, 1002)
        nuke.execute.reset_mock()
        process()
        assert starts.call_count == 2
        nuke.execute.assert_not_called()


@pytest.fixture
def tracking_context(tmp_path, monkeypatch):
    return adapter_context.__wrapped__(SimpleNamespace(param="segment"), tmp_path, monkeypatch)


def test_tracking_reuses_ready_jpegs_and_prepares_missing_frame(tracking_context, monkeypatch):
    node, nuke, starts, _process, root = tracking_context
    node.values.update({"key_frame": Knob(1001), "box_enabled": Knob(False),
                        "profile": Knob("low_memory"), "model": Knob(0)})
    node["model"].getValue = lambda: 0
    monkeypatch.setattr(segment, "_correction_frames", lambda node: set())
    monkeypatch.setattr(segment, "_collect_points", lambda node, kind, frame: [(1, 1)] if kind == "positive" else [])
    monkeypatch.setattr(segment, "_bbox_at", lambda *args: (0, 0, 2, 2))
    monkeypatch.setattr(segment, "_cache_work_root", lambda node: root)
    monkeypatch.setattr(segment, "segment_video_payload", lambda **kwargs: {})
    input_cache.cache_frames()
    nuke.execute.reset_mock()
    segment.propagate_video("forward")
    assert starts.call_count == 1
    nuke.execute.assert_not_called()
    starts.reset_mock()
    (root / "video/01002.jpg").unlink()
    segment.propagate_video("forward")
    assert starts.call_count == 1
    assert nuke.execute.call_count == 1
    assert (root / "video/01002.jpg").is_file()


def test_extending_range_exports_only_new_frames(adapter_context):
    node, nuke, starts, process, root = adapter_context
    input_cache.cache_frames()
    existing = {p: p.stat().st_mtime_ns for p in root.rglob("*")
                if p.is_file() and not p.name.startswith("inputs.sqlite3")}
    node["range_last"].setValue(1005)
    node["last_frame"].setValue(1005)
    nuke.execute.reset_mock()
    expected_per_frame = len(input_cache._specs(node, 1004, 1001, 1005))
    input_cache.cache_frames()
    assert nuke.execute.call_count == expected_per_frame * 2
    assert all(p.stat().st_mtime_ns == stamp for p, stamp in existing.items())
    nuke.execute.reset_mock()
    process()
    assert starts.call_count == 1
    nuke.execute.assert_not_called()


def test_current_frame_populates_only_current_inputs_then_reuses_them(adapter_context):
    node, nuke, starts, process, root = adapter_context
    expected = len(input_cache._specs(node, 1001, 1001, 1001))
    process(current=True)
    assert starts.call_count == 1, node["kyven_status"].value()
    assert nuke.execute.call_count == expected
    assert not list(root.rglob("*1002*"))
    assert not list(root.rglob("*1003*"))
    nuke.execute.reset_mock()
    process(current=True)
    assert starts.call_count == 2
    nuke.execute.assert_not_called()
    specs = input_cache._specs(node, 1001, 1001, 1001)
    specs[-1][1].unlink()
    process(current=True)
    assert starts.call_count == 3
    assert nuke.execute.call_count == 1


@pytest.mark.parametrize("failure", ["cancel", "export_error"])
def test_range_preparation_failure_never_submits_inference(adapter_context, failure):
    node, nuke, starts, process, _root = adapter_context
    if failure == "cancel":
        nuke.ProgressTask.return_value.isCancelled = lambda: True
    else:
        def render_or_fail(writer, first, last):
            if first == 1002:
                raise RuntimeError("upstream Read failed at 1002")
            nuke.render(writer, first, last)
        nuke.execute.side_effect = render_or_fail
    process()
    starts.assert_not_called()
    if failure == "cancel":
        nuke.execute.assert_not_called()
    else:
        status = node["kyven_status"].value() + str(nuke.message.call_args)
        assert "1002" in status
    # Retrying resumes preparation and starts inference only after completion.
    nuke.ProgressTask.return_value.isCancelled = lambda: False
    nuke.execute.side_effect = nuke.render
    process()
    assert starts.call_count == 1


@pytest.mark.parametrize("seed_frames", [(), (1001,), (1001, 1003)])
def test_first_process_range_orders_missing_exports_before_worker_and_repeat_is_read_only(
        adapter_context, seed_frames):
    node, nuke, starts, process, _root = adapter_context
    specs = [(frame, writer, path) for frame in range(1001, 1004)
             for writer, path in input_cache._specs(node, frame, 1001, 1003)]
    for frame, writer, path in specs:
        if frame in seed_frames:
            input_cache.export_input(nuke, node, writer, frame, path)
    seeded = {path: (path.stat().st_mtime_ns, path.read_bytes())
              for frame, _writer, path in specs if frame in seed_frames}
    missing = {path for frame, _writer, path in specs if frame not in seed_frames}
    events = []
    def export(writer, first, last):
        # Each export must finish before any processing worker is allowed to start.
        assert not any(event[0] == "worker" for event in events)
        pattern = str(writer["file"].value())
        path = Path(pattern % first if "%" in pattern else pattern)
        assert path in missing
        nuke.render(writer, first, last)
        events.append(("export", path))
    def start():
        assert all(input_cache.ready(node, path, frame) for frame, _writer, path in specs)
        assert {path for event, path in events if event == "export"} == missing
        events.append(("worker", None))
    nuke.execute.reset_mock()
    nuke.execute.side_effect = export
    starts.side_effect = start
    process()
    assert starts.call_count == 1, node["kyven_status"].value()
    assert nuke.execute.call_count == len(missing)
    assert events[-1] == ("worker", None)
    assert all((path.stat().st_mtime_ns, path.read_bytes()) == before
               for path, before in seeded.items())
    complete = {path: (path.stat().st_mtime_ns, path.read_bytes())
                for _frame, _writer, path in specs}
    # Repeating Process Range may submit work, but must not export or change input files.
    nuke.execute.reset_mock()
    starts.side_effect = lambda: events.append(("repeat_worker", None))
    process()
    assert starts.call_count == 2
    nuke.execute.assert_not_called()
    assert all((path.stat().st_mtime_ns, path.read_bytes()) == before
               for path, before in complete.items())


@pytest.mark.parametrize("current", [False, True])
def test_manual_processing_disables_live_and_completion_does_not_reenable_live_mode(adapter_context, current):
    node, _nuke, starts, process, _root = adapter_context
    node["live_mode"].setValue(True)
    process(current=current)
    assert starts.call_count == 1
    assert node["live_mode"].value() is False
    assert node["live_mode"].enabled is False
    # Completion/error/cancel restores control availability without checking Live again.
    segment._set_operation_controls(node, False)
    assert node["live_mode"].enabled is True
    assert node["live_mode"].value() is False
