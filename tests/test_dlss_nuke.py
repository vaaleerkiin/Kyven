"""Graph and publication tests without a Nuke license or GPU."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "hosts" / "nuke"))

from kyven_nuke import dlss_node
from kyven_nuke.dlss_live import DlssLiveState

_schedule_live = dlss_node._schedule_live


class Knob:
    def __init__(self, name, label="", choices=None):
        self._name, self._value = name, ""
        self.choices = choices if isinstance(choices, list) else None
        self.command = choices if isinstance(choices, str) else ""
        self.owner = None
        self.visible = True
        self.enabled = True

    def name(self):
        return self._name

    def value(self):
        return self._value

    def getValue(self):
        return self.choices.index(self._value) if self.choices else self._value

    def setValue(self, value):
        self._value = self.choices[value] if self.choices and isinstance(value, int) else value
        if self.owner is not None and self.owner.nuke.auto_notify:
            owner = self.owner if self.owner.kind == "Group" else self.owner.parent_group
            if owner is not None:
                self.owner.nuke.notify(owner, self._name if owner is self.owner else "inputChange")

    def setVisible(self, value):
        self.visible = value

    def setEnabled(self, value):
        self.enabled = value

    def setCommand(self, value):
        self.command = value

    def values(self):
        return self.choices or []

    def setValues(self, values):
        self.choices = values

    def setExpression(self, value):
        self.expression = value

    def __getattr__(self, name):
        if name in {"setRange", "setTooltip", "setFlag", "clearFlag", "setVisible",
                    "setLabel", "setEnabled", "execute"}:
            return lambda *args: None
        raise AttributeError(name)


class Node:
    def __init__(self, nuke, kind, name):
        self.nuke, self.kind, self._name = nuke, kind, name
        self.inputs = {}
        self.parent_group = None
        self._knobs = {k: Knob(k) for k in (
            "tile_color", "knobChanged", "number", "file", "file_type", "channels",
            "datatype", "autocrop", "raw", "colorspace", "on_error", "which", "reload",
            "first", "last", "origfirst", "origlast", "first_frame", "last_frame",
            "from0", "from1", "from2", "from3", "to0", "to1", "to2", "to3",
        )}
        self._knobs["which"].setValue(0)
        self._knobs["first_frame"].setValue(1)
        self._knobs["last_frame"].setValue(10)
        self.hashes = [1234]
        self.pending_hashes = None
        self.children = {}
        for knob in self._knobs.values():
            knob.owner = self

    def __getitem__(self, name):
        return self._knobs[name]

    def knobs(self):
        return self._knobs

    def addKnob(self, knob):
        knob.owner = self
        self._knobs[knob.name()] = knob

    def removeKnob(self, knob):
        del self._knobs[knob.name()]

    def input(self, i):
        return self.inputs.get(i)

    def setInput(self, i, node):
        self.inputs[i] = node

    def fullName(self):
        return self._name

    def opHashes(self):
        return self.hashes

    def forceValidate(self):
        if self.pending_hashes is not None:
            self.hashes, self.pending_hashes = self.pending_hashes, None

    def width(self):
        return 1920

    def height(self):
        return 1080

    def begin(self):
        self.nuke.stack.append(self)

    def end(self):
        self.nuke.stack.pop()


class FakeNuke:
    READ_ONLY, STARTLINE = 1, 2

    def __init__(self):
        self.stack, self.top = [], {}
        self.auto_notify = False
        self._frame = 1
        self.callback_node = self.callback_knob = None
        self.source = Node(self, "Constant", "Original")
        self.root_node = Node(self, "Root", "Root")
        self.nodes = SimpleNamespace(**{kind: self.factory(kind)
                                       for kind in ("Group", "Input", "Write", "Read",
                                                    "Copy", "Switch", "Output")})

    def factory(self, kind):
        def create(name):
            node = Node(self, kind, name)
            if self.stack and self.stack[-1].kind == "Group":
                node.parent_group = self.stack[-1]
            (self.stack[-1].children if self.stack else self.top)[name] = node
            return node
        return create

    def __getattr__(self, name):
        if name.endswith("_Knob"):
            return Knob
        raise AttributeError(name)

    def selectedNodes(self):
        return [self.source]

    def root(self):
        return self.root_node

    def frame(self):
        return self._frame

    def executing(self):
        return False

    def toNode(self, name):
        return (self.stack[-1].children if self.stack and self.stack[-1].kind != "Root" else self.top).get(name)

    def allNodes(self, kind=None, recurseGroups=False):
        values = list((self.stack[-1].children if self.stack and self.stack[-1].kind != "Root" else self.top).values())
        return [node for node in values if kind is None or node.kind == kind]

    def thisNode(self):
        return self.callback_node

    def thisKnob(self):
        return self.callback_knob

    def notify(self, node, name):
        if not node["knobChanged"].value():
            return
        previous = self.callback_node, self.callback_knob
        self.callback_node = node
        self.callback_knob = node.knobs().get(name, Knob(name))
        try:
            dlss_node.knob_changed()
        finally:
            self.callback_node, self.callback_knob = previous

    def executeInMainThread(self, function, args=()):
        # Nuke restores the UI frame after the dispatched callback returns, not
        # before execute(Write) returns inside that callback.
        frame = self._frame
        try:
            return function(*args)
        finally:
            self._frame = frame

    executeInMainThreadWithResult = executeInMainThread


@pytest.fixture
def graph(tmp_path, monkeypatch):
    nuke = FakeNuke()
    monkeypatch.setitem(sys.modules, "nuke", nuke)
    monkeypatch.setattr(dlss_node, "_schedule_live", lambda *args: None)
    group = dlss_node.create_dlss_node()
    group["cache_root"].setValue(str(tmp_path))
    yield nuke, group
    dlss_node._set_busy(group.fullName(), False)
    dlss_node._states.clear()
    dlss_node._operations.clear()
    dlss_node._mutations.clear()
    dlss_node._polling.clear()
    for timer in dlss_node._live_timers.values():
        timer.cancel()
    dlss_node._live_timers.clear()


@pytest.fixture
def live_clock(graph, monkeypatch):
    """Advance real Live wake-ups without depending on another Nuke UI event."""
    nuke, _ = graph
    now = [100.0]
    pending = []

    class Timer:
        def __init__(self, delay, callback):
            self.due, self.callback = now[0] + delay, callback
            self.cancelled = False

        def start(self):
            pending.append(self)

        def cancel(self):
            self.cancelled = True

    def advance(seconds=0.5):
        end = now[0] + seconds
        while True:
            due = [timer for timer in pending if not timer.cancelled and timer.due <= end]
            if not due:
                break
            timer = min(due, key=lambda timer: timer.due)
            pending.remove(timer)
            now[0] = timer.due
            timer.callback()
        now[0] = end

    monkeypatch.setattr(dlss_node.threading, "Timer", Timer)
    monkeypatch.setattr(dlss_node.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(dlss_node, "_schedule_live", _schedule_live)
    renders = []

    def render(node, live):
        assert live
        state, cancellation = _running_state(nuke, node)
        node["kyven_busy"].setValue(True)
        renders.append((dlss_node._signature(node), cancellation))
        return state

    monkeypatch.setattr(dlss_node, "process_current_frame", render)
    monkeypatch.setattr(dlss_node, "_operation_node", lambda name: nuke.toNode(name))
    return advance, renders


def test_live_renders_once_after_a_single_edit_without_further_ui_events(graph, live_clock):
    nuke, group = graph
    advance, renders = live_clock
    group["intensity"].setValue(0.7)
    nuke.notify(group, "intensity")
    advance(0.2)
    assert not renders
    advance(0.3)
    assert len(renders) == 1
    advance(1)
    assert len(renders) == 1


@pytest.mark.parametrize("name,value", [
    ("intensity", 0.6), ("local_tone", 0.6), ("local_structure", 0.6),
    ("skin_structure", 0.5), ("style", "Natural"), ("preset", "Preset 1"),
    ("processing_scale", "0.5"), ("temporal", True), ("temporal_strength", 0.6),
    ("rejection", 0.2), ("scene_cut", 0.6), ("motion_x", "backward.u"),
    ("motion_y", "backward.v"), ("motion_units", "UV"), ("motion_y_up", False),
    ("motion_scale_x", 2), ("motion_scale_y", 2), ("depth_channel", "R"),
    ("cache_version", "v002"), ("reset_epoch", 1),
    ("hdr_rolloff", False), ("rolloff_knee", 0.7), ("mask_source", "Source Alpha"),
])
def test_live_replaces_active_work_after_processing_edit(graph, live_clock, name, value):
    nuke, group = graph
    advance, renders = live_clock
    old_state, old_cancellation = _running_state(nuke, group)
    group["kyven_busy"].setValue(True)
    group[name].setValue(value)
    nuke.notify(group, name)
    advance()
    assert old_cancellation.is_set()
    assert not renders
    dlss_node._finish(group.fullName(), dlss_node._identity(group), old_state.active,
                      "", old_cancellation)
    advance()
    assert len(renders) == 1
    assert renders[0][0] == dlss_node._signature(group)
    assert not renders[0][1].is_set()


def test_live_ignores_upstream_changes_but_renders_on_frame_change(graph, live_clock, monkeypatch):
    nuke, group = graph
    advance, renders = live_clock
    state, cancellation = _running_state(nuke, group)
    dlss_node._finish(group.fullName(), dlss_node._identity(group), state.active, "", cancellation)
    advance()
    assert not renders
    # The upstream node does not emit this Group's knobChanged callback.
    nuke.source.pending_hashes = [5678]
    advance()
    assert not renders
    monkeypatch.setattr(nuke, "frame", lambda: 2)
    advance()
    assert len(renders) == 1


def test_live_coalesces_edits_and_stops_watching_when_disabled(graph, live_clock):
    nuke, group = graph
    advance, renders = live_clock
    for value in (0.2, 0.7, 0.9):
        group["intensity"].setValue(value)
        nuke.notify(group, "intensity")
        advance(0.1)
    advance()
    assert len(renders) == 1
    assert renders[0][0] == dlss_node._signature(group)
    group["live_mode"].setValue(False)
    nuke.notify(group, "live_mode")
    advance()
    assert renders[0][1].is_set()
    assert not dlss_node._live_timers


def test_six_external_updates_cannot_restart_a_completed_frame(graph, live_clock, monkeypatch):
    nuke, group = graph
    advance, renders = live_clock
    state, cancellation = _running_state(nuke, group)
    signature = state.active
    dlss_node._finish(group.fullName(), dlss_node._identity(group), signature, "", cancellation)
    nuke.auto_notify = True

    def forbidden():
        raise AssertionError("Live polling must not evaluate or hash upstream inputs")

    monkeypatch.setattr(nuke.source, "forceValidate", forbidden)
    monkeypatch.setattr(nuke.source, "opHashes", forbidden)
    monkeypatch.setattr(dlss_node, "resolve_worker", lambda *args: Path(sys.executable))
    for index in range(6):
        # Repeated source dirtiness, connection notifications, Read reloads, status
        # updates and a runtime refresh all leave the same frame/settings intent.
        nuke.source.hashes = [100 + index]
        nuke.notify(group, "inputChange")
        group.children["KyvenDlssRead"]["file"].setValue(f"cache.{index}.exr")
        group["kyven_status"].setValue(f"Service update {index}")
        dlss_node.refresh_runtime(group)
        dlss_node.poll_live(group)
        advance(2)
        assert not renders
        assert state.attempted == signature
        assert not cancellation.is_set()
    group.setInput(0, Node(nuke, "Constant", "Replacement"))
    nuke.notify(group, "inputChange")
    advance(2)
    assert not renders
    # Only an actual own processing setting now starts a replacement.
    group["intensity"].setValue(0.6)
    advance()
    assert len(renders) == 1


def test_mask_channel_is_always_alpha_even_for_older_saved_nodes(graph, monkeypatch):
    from kyven_nuke.dlss_interface import restyle_dlss_node
    _, group = graph
    monkeypatch.setattr(dlss_node, "resolve_worker", lambda *args: Path(sys.executable))
    group["mask_channel"].setValue("R")
    assert dlss_node._request_settings(group)["mask_channel"] == "A"
    restyle_dlss_node(group, upgrade=True)
    assert group["mask_channel"].value() == "A"
    assert not group["mask_channel"].visible


def test_export_last_plus_one_context_does_not_cancel_or_restart_live(graph, monkeypatch):
    nuke, group = graph
    nuke._frame = 400
    state, cancellation = _running_state(nuke, group)
    signature = state.active

    def export(writer, first, last):
        nuke._frame = last + 1
        # Reentrant notifications inside the private render context are internal.
        nuke.notify(group, "inputChange")
        dlss_node.poll_live(group)

    monkeypatch.setattr(nuke, "execute", export, raising=False)
    _, _, validated = nuke.executeInMainThreadWithResult(
        dlss_node._export, (group.fullName(), 400, "context-check",
                           {"temporal": False, "mask_source": "none"}, False, signature))
    assert nuke.frame() == 400
    assert validated == signature
    assert state.desired == signature
    assert not cancellation.is_set()
    assert state.accepts(dlss_node._signature(group))
    # A persistent UI change after export is still a new artist request.
    nuke._frame = 402
    dlss_node.poll_live(group)
    assert cancellation.is_set()
    assert state.desired == dlss_node._signature(group)


def test_completed_live_result_stays_idle_through_redraw_and_timer_updates(
        graph, live_clock, tmp_path, monkeypatch):
    nuke, group = graph
    advance, renders = live_clock
    group["intensity"].setValue(0.8)
    nuke.notify(group, "intensity")
    advance()
    assert len(renders) == 1
    signature, cancellation = renders[0]
    nuke.auto_notify = True

    def clear_global_cache():
        raise AssertionError("Publishing DLSS must not invalidate the global upstream RAM cache")

    monkeypatch.setattr(nuke, "clearRAMCache", clear_global_cache, raising=False)
    monkeypatch.setattr(nuke, "redraw", lambda: dlss_node.poll_live(group), raising=False)
    staging = tmp_path / "staged.exr"
    staging.write_bytes(b"verified result")
    job = {"result": {"output": str(staging), "metadata": {}}}
    identity = dlss_node._identity(group)
    assert dlss_node._publish(group.fullName(), job, str(tmp_path / "output.%04d.exr"),
                              1, identity, signature, True, cancellation)
    dlss_node._finish(group.fullName(), identity, signature, "", cancellation)
    advance(30)
    assert len(renders) == 1
    assert not group["kyven_busy"].value()
    assert not cancellation.is_set()
    # A real edit still produces exactly one replacement.
    group["local_tone"].setValue(0.7)
    advance()
    assert len(renders) == 2


def test_live_scan_does_not_reenter_when_group_context_emits_update_ui(graph, monkeypatch):
    from kyven_nuke import live
    nuke, _ = graph
    entries = []
    begin = nuke.root_node.begin

    def recursive_begin():
        entries.append(1)
        if len(entries) > 2:
            raise AssertionError("Root context change recursively reentered the Live scan")
        begin()
        live.update_live_nodes()

    monkeypatch.setattr(nuke.root_node, "begin", recursive_begin)
    monkeypatch.setattr(live, "_last_scan", -100)
    # Reentry can occur after the normal scan throttle has expired.
    ticks = iter(range(100, 1000))
    monkeypatch.setattr(live.time, "monotonic", lambda: next(ticks))
    live.update_live_nodes()
    assert entries == [1]
    assert not live._scan_active


@pytest.mark.parametrize("scale,expected", [("1", 1920), ("0.5", 960),
                                          ("0.25", 480), ("0.125", 240)])
def test_processing_size_menu_scales_source_dimensions(graph, monkeypatch, scale, expected):
    _, group = graph
    assert group["processing_scale"].value() == "1"
    assert group["processing_scale"].values() == ["1", "0.5", "0.25", "0.125"]
    assert not group["max_dimension"].visible
    monkeypatch.setattr(dlss_node, "resolve_worker", lambda *args: Path(sys.executable))
    group["processing_scale"].setValue(scale)
    assert dlss_node._request_settings(group)["max_dimension"] == expected


def test_node_keeps_source_alpha_and_aovs_and_exports_float_linear_guides(graph):
    nuke, group = graph
    assert group.input(0) is nuke.source
    assert group["kyven_kind"].value() == "dlss"
    assert group["mask_source"].value() == "Full Frame"
    assert group["live_mode"].value()
    children = group.children
    for index, name in enumerate(("Source", "Motion", "Depth", "Mask")):
        assert children[name]["number"].value() == index
        writer = children["KyvenDlss" + name + "Write"]
        assert writer.input(0) is children[name]
        assert writer["file_type"].value() == "exr"
        assert writer["datatype"].value() == "32 bit float"
        assert writer["raw"].value()
        assert not writer["autocrop"].value()
    copy = children["KyvenDlssSourceChannels"]
    assert copy.input(0) is children["Source"]
    assert copy.input(1) is children["KyvenDlssRead"]
    assert {copy[f"to{i}"].value() for i in range(3)} == {
        "rgba.red", "rgba.green", "rgba.blue"}
    assert all(node.kind not in {"Premult", "Unpremult"} for node in children.values())
    assert not nuke.stack


def test_mask_interface_selects_neural_guide_and_shows_only_relevant_controls(graph):
    nuke, group = graph
    assert group["mask_source"].enabled
    assert not group["mask_channel"].visible
    assert not group["protect_outside"].visible
    assert "guides DLSS inference" in group["mask_help"].value()
    assert group["mask_source"].value() == "Full Frame"
    group["mask_source"].setValue("Source Alpha")
    nuke.notify(group, "mask_source")
    assert not group["mask_channel"].visible
    assert group["protect_outside"].visible
    group["mask_source"].setValue("Mask Input")
    nuke.notify(group, "mask_source")
    assert not group["mask_channel"].visible
    assert group["mask_channel"].value() == "A"


def _running_state(nuke, group):
    assert dlss_node._begin_node_operation(group)
    state = DlssLiveState()
    state.settings = dlss_node._settings_signature(group)
    signature = dlss_node._signature(group)
    state.observe(signature, 0)
    cancellation = state.begin(signature)
    dlss_node._states[dlss_node._identity(group)] = state
    dlss_node._operations[dlss_node._identity(group)] = cancellation
    return state, cancellation


def test_input_change_notification_with_same_external_connections_does_not_cancel(graph):
    nuke, group = graph
    _, cancellation = _running_state(nuke, group)
    before = group["reset_epoch"].value()
    nuke.notify(group, "inputChange")
    assert not cancellation.is_set()
    assert group["reset_epoch"].value() == before


def test_noop_setting_notifications_do_not_cancel_but_artist_edit_does(graph):
    nuke, group = graph
    _, cancellation = _running_state(nuke, group)
    nuke.notify(group, "intensity")
    nuke.notify(group, "live_mode")
    assert not cancellation.is_set()
    group["intensity"].setValue(0.5)
    nuke.notify(group, "intensity")
    assert cancellation.is_set()
    assert cancellation.reason == "settings_changed"


def test_actual_rewire_is_not_a_live_trigger_and_does_not_cancel(graph):
    nuke, group = graph
    _, cancellation = _running_state(nuke, group)
    group.setInput(0, Node(nuke, "Constant", "OtherSource"))
    nuke.notify(group, "inputChange")
    dlss_node.poll_live(group)
    assert not cancellation.is_set()
    assert group["reset_epoch"].value() == 0


def test_live_scan_uses_root_even_when_node_graph_is_inside_group(graph):
    from kyven_nuke.live import _root_groups
    nuke, group = graph
    state, cancellation = _running_state(nuke, group)
    group.begin()
    try:
        nodes = _root_groups(nuke)
        assert group in nodes
        dlss_node.prune_live_nodes(nodes)
    finally:
        group.end()
    assert not cancellation.is_set()
    assert state.active


@pytest.mark.parametrize("render_context_advance", [False, True])
def test_source_to_live_result_survives_nuke_export_and_read_notifications(
        graph, tmp_path, monkeypatch, render_context_advance):
    pytest.importorskip("OpenEXR")
    import numpy as np

    from kyven.client import KyvenClient
    from kyven.dlss.images import read_channels, write_exr
    from kyven.dlss.service import DlssService
    from kyven.models.catalog import ModelCatalog
    from kyven.segment.providers.registry import ProviderRegistry
    from kyven.segment.service import SegmentService
    from kyven.server.app import KyvenServer, ServerConfig
    from kyven.server.jobs import JobManager

    nuke, group = graph
    worker = tmp_path / "synthetic-worker.exe"
    worker.write_bytes(b"fixture; not executed")
    monkeypatch.setattr(dlss_node, "resolve_worker", lambda *args: worker)
    calls = []
    class Provider:
        def predict(self, request, rgba, motion, mask, depth, reset, cancellation):
            cancellation.raise_if_cancelled()
            calls.append(request.frame)
            result = rgba.copy()
            result[..., :3] += 0.2
            return result
        def unload(self):
            pass
    registry = ProviderRegistry()
    registry.register("dlss-external", Provider)
    manager = JobManager(SegmentService(registry), dlss_service=DlssService(registry))
    server = KyvenServer(ServerConfig("x" * 32, tmp_path / "models", port=0), manager,
                         registry, ModelCatalog.builtin())
    server.start()
    client = KyvenClient(f"http://127.0.0.1:{server.port}", "x" * 32)
    monkeypatch.setattr(dlss_node, "ensure_server", lambda **kwargs: client)
    nuke.auto_notify = True
    def export(writer, first, last):
        pixels = np.full((4, 8, 4), 0.3, dtype=np.float32)
        pixels[..., 3] = 0.7
        write_exr(Path(writer["file"].value()), pixels)
        # Simulate first validation warming the source hash and Group notifications
        # emitted while Nuke evaluates an internal Write.
        if render_context_advance:
            nuke._frame = last + 1
        else:
            nuke.source.hashes = [9999]
        nuke.notify(group, "inputChange")
        dlss_node.poll_live(group)
    nuke.execute = export
    try:
        dlss_node.process_current_frame(group, live=True)
        deadline = time.monotonic() + 5
        while group["kyven_busy"].value() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not group["kyven_busy"].value(), group["kyven_status"].value()
        assert calls == [1], group["kyven_status"].value()
        assert "complete" in group["kyven_status"].value(), group["kyven_status"].value()
        assert group.children["KyvenDlssResultSwitch"]["which"].value() == 1
        output = Path(str(dlss_node._public_pattern(group)) % 1)
        np.testing.assert_allclose(read_channels(output)["R"], 0.5)
        state = dlss_node._states[dlss_node._identity(group)]
        assert not state.cancel.is_set()
        assert not state.active
        assert state.attempted == dlss_node._signature(group)
        dlss_node.poll_live(group)
        assert calls == [1]
    finally:
        server.shutdown()


def test_host_payload_contains_external_guide_conventions_without_engine_imports(graph, monkeypatch):
    _, group = graph
    group["worker"].setValue(str(Path(sys.executable)))
    monkeypatch.setattr(dlss_node, "resolve_worker", lambda *args: Path(sys.executable))
    group["temporal"].setValue(True)
    group["mask_source"].setValue("Mask Input")
    payload = dlss_node._request_settings(group)
    assert payload["mask_source"] == "input"
    assert payload["motion_units"] == "pixels"
    assert payload["motion_y_up"]
    assert payload["max_dimension"] == 1920
    assert "processing_scale" not in payload
    assert payload["style"] == 0
    assert payload["session"].startswith(group["kyven_uuid"].value())
    assert Path(payload["worker"]).is_absolute()


def test_upstream_same_frame_edit_does_not_change_live_signature(graph):
    nuke, group = graph
    before = dlss_node._signature(group)
    nuke.source.hashes = [5678]
    assert dlss_node._signature(group) == before


def test_blank_cache_folder_has_an_automatic_isolated_cache(graph):
    _, group = graph
    group["cache_root"].setValue("")
    path = dlss_node._public_pattern(group)
    assert path.is_absolute()
    assert group["kyven_uuid"].value() in path.parts
    assert path.name == "dlss_v001.%04d.exr"


def test_layout_matches_kyven_sections_and_moves_expert_controls_to_advanced(graph):
    _, group = graph
    names = list(group.knobs())
    assert names.index("kyven_title") < names.index("model_section")
    assert names.index("model_section") < names.index("processing_section")
    assert names.index("processing_section") < names.index("cache_section")
    assert names.index("status_section") < names.index("dlss_advanced")
    assert names.index("dlss_advanced") < names.index("motion_x")
    assert "open_model_manager" in names
    assert "KYVEN / DLSS" in group["kyven_title"].value()


def test_upgrading_layout_preserves_uuid_inputs_and_settings(graph):
    from kyven_nuke.dlss_interface import restyle_dlss_node
    nuke, group = graph
    identity = group["kyven_uuid"].value()
    group["dlss_ui_version"].setValue(0)
    group["live_mode"].setValue(False)
    group["intensity"].setValue(0.7)
    restyle_dlss_node(group, upgrade=True)
    assert group["live_mode"].value()
    assert group["kyven_uuid"].value() == identity
    assert group.input(0) is nuke.source
    assert group["intensity"].value() == 0.7


def test_scale_upgrade_defaults_to_full_size_and_preserves_later_selection(graph):
    from kyven_nuke.dlss_interface import restyle_dlss_node
    _, group = graph
    identity = group["kyven_uuid"].value()
    group.removeKnob(group["processing_scale"])
    group["dlss_ui_version"].setValue(3)
    group["live_mode"].setValue(False)
    restyle_dlss_node(group, upgrade=True)
    assert group["processing_scale"].value() == "1"
    assert not group["live_mode"].value()
    group["processing_scale"].setValue("0.25")
    restyle_dlss_node(group, upgrade=True)
    assert group["processing_scale"].value() == "0.25"
    assert group["kyven_uuid"].value() == identity


def test_stale_live_result_cannot_overwrite_public_sequence(graph, tmp_path, monkeypatch):
    _, group = graph
    monkeypatch.setattr(dlss_node, "_operation_node", lambda name: group)
    group["live_mode"].setValue(True)
    identity = dlss_node._identity(group)
    signature = dlss_node._signature(group)
    state = DlssLiveState()
    state.observe(signature, 0)
    state.begin(signature)
    dlss_node._states[identity] = state
    public = tmp_path / "public.0001.exr"
    public.write_bytes(b"previous valid result")
    staging = tmp_path / "staged.exr"
    staging.write_bytes(b"stale pixels")
    job = {"result": {"output": str(staging), "metadata": {}}}
    group["intensity"].setValue(0.8)
    assert not dlss_node._publish(group.fullName(), job, str(tmp_path / "public.%04d.exr"),
                                  1, identity, signature, True)
    assert public.read_bytes() == b"previous valid result"
    assert staging.exists()


def test_staging_cleanup_only_removes_known_request_files(tmp_path):
    root = tmp_path / "request"
    root.mkdir()
    source, output = root / "source.exr", root / "output.exr"
    protected = tmp_path / "outside.exr"
    for path in (source, output, protected):
        path.write_bytes(b"pixels")
    dlss_node._discard_request({"source": str(source), "output": str(output),
                                "mask": str(protected)})
    assert not root.exists()
    assert protected.read_bytes() == b"pixels"


def test_accepted_result_and_manifest_are_published_to_artist_cache(graph, tmp_path, monkeypatch):
    _, group = graph
    monkeypatch.setattr(dlss_node, "_operation_node", lambda name: group)
    monkeypatch.setattr(dlss_node, "_set_status", lambda *args: None)
    staging = tmp_path / "staged.exr"
    staging.write_bytes(b"finished pixels")
    manifest = staging.with_suffix(".exr.kyven.json")
    manifest.write_text('{"cache_key":"verified"}')
    cancellation = threading.Event()
    identity = dlss_node._identity(group)
    dlss_node._operations[identity] = cancellation
    pattern = str(tmp_path / "dlss_v001.%04d.exr")
    job = {"result": {"output": str(staging), "metadata": {"history_reason": "continuous"}}}
    assert dlss_node._publish(group.fullName(), job, pattern, 1, identity, "manual", False,
                              cancellation)
    assert (tmp_path / "dlss_v001.0001.exr").read_bytes() == b"finished pixels"
    assert (tmp_path / "_kyven_work/manifests/dlss_v001.0001.exr.kyven.json").is_file()
    assert not staging.exists()
    assert group.children["KyvenDlssResultSwitch"]["which"].value() == 1


def test_old_operation_cannot_publish_or_finish_after_script_reload(graph, tmp_path, monkeypatch):
    _, group = graph
    monkeypatch.setattr(dlss_node, "_operation_node", lambda name: group)
    identity = dlss_node._identity(group)
    old, new = threading.Event(), threading.Event()
    dlss_node._operations[identity] = new
    staged = tmp_path / "stale.exr"
    staged.write_bytes(b"stale")
    job = {"result": {"output": str(staged), "metadata": {}}}
    assert not dlss_node._publish(group.fullName(), job, str(tmp_path / "public.%04d.exr"),
                                  1, identity, "same-signature", False, old)
    dlss_node._finish(group.fullName(), identity, "same-signature", "obsolete completion", old)
    assert dlss_node._operations[identity] is new
    assert staged.exists()


def test_host_adapter_import_does_not_load_inference_or_exr_libraries():
    script = ("import sys,json; import kyven_nuke; "
              "print(json.dumps([n for n in ('torch','numpy','OpenEXR','kyven.dlss.provider') "
              "if n in sys.modules]))")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / "hosts" / "nuke"))
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True,
                            text=True, check=True)
    assert json.loads(result.stdout) == []
