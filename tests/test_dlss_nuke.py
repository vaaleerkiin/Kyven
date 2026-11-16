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
from unittest.mock import Mock

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

    def setExpression(self, value, *args):
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
            "disable", "operation", "box", "reformat",
            "compression", "interleave",
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

    def channels(self):
        return ["rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"]

    def begin(self):
        self.nuke.stack.append(self)

    def end(self):
        self.nuke.stack.pop()


class FakeNuke:
    READ_ONLY, STARTLINE = 1, 2

    def __init__(self):
        self.stack, self.top = [], {}
        self._layers = {}
        self.auto_notify = False
        self._frame = 1
        self.callback_node = self.callback_knob = None
        self.source = Node(self, "Constant", "Original")
        self.root_node = Node(self, "Root", "Root")
        self.nodes = SimpleNamespace(**{kind: self.factory(kind)
                                       for kind in ("Group", "Input", "Write", "Read",
                                                    "Copy", "Switch", "Output", "Remove", "Crop")})

    def layers(self):
        return list(self._layers)

    def Layer(self, name, channels):
        self._layers[name] = channels

    def delete(self, node):
        (node.parent_group.children if node.parent_group else self.top).pop(node._name)

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
    ("rejection", 0.2), ("scene_cut", 0.6), ("motion_x", "R"),
    ("motion_y", "G"), ("motion_units", "UV"), ("motion_y_up", False),
    ("motion_scale_x", 2), ("motion_scale_y", 2), ("depth_channel", "R"),
    ("cache_version", "v002"), ("reset_epoch", 1),
    ("hdr_rolloff", False), ("rolloff_knee", 0.7), ("mask_source", "Source Alpha"),
    ("history_frames", "4"), ("history_budget_mb", 256), ("history_clamp", 1),
    ("motion_consistency", True), ("forward_x", "B"), ("forward_y", "A"),
    ("consistency_threshold", 2), ("depth_rejection", 0.2),
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


def test_new_dlss_node_uses_backward_motion_without_rgb(graph):
    from kyven.dlss.images import channel

    _, group = graph
    vectors = {"backward.u": object(), "backward.v": object()}
    assert channel(vectors, group["motion_x"].value()) is vectors["backward.u"]
    assert channel(vectors, group["motion_y"].value()) is vectors["backward.v"]
    assert group["motion_units"].value() == "Pixels"
    assert group["motion_y_up"].value()


@pytest.mark.parametrize("mapping, expected", [
    (("R", "G"), ("R", "G")),
    (("custom.x", "custom.y"), ("custom.x", "custom.y")),
])
def test_upgrade_preserves_existing_motion_mapping(graph, mapping, expected):
    from kyven_nuke.dlss_interface import restyle_dlss_node

    _, group = graph
    identity = group["kyven_uuid"].value()
    group["dlss_ui_version"].setValue(8)
    group["motion_x"].setValue(mapping[0])
    group["motion_y"].setValue(mapping[1])
    restyle_dlss_node(group, upgrade=True)
    assert (group["motion_x"].value(), group["motion_y"].value()) == expected
    assert group["kyven_uuid"].value() == identity
    group["motion_x"].setValue("R")
    group["motion_y"].setValue("G")
    restyle_dlss_node(group, upgrade=True)
    assert (group["motion_x"].value(), group["motion_y"].value()) == ("R", "G")


def test_nuke_vector_button_maps_channels_without_creating_a_generator(graph):
    nuke, group = graph
    nuke.auto_notify = True
    nuke.callback_node = group
    before = dict(group.children)
    dlss_node.use_nuke_vectors()
    assert group["motion_x"].value() == "backward.u"
    assert group["motion_y"].value() == "backward.v"
    assert group["forward_x"].value() == "forward.u"
    assert group["forward_y"].value() == "forward.v"
    assert group["motion_units"].value() == "Pixels"
    assert group["motion_y_up"].value()
    assert group.children == before


def test_prepared_input_check_does_not_render_or_change_live_intent(graph, monkeypatch, tmp_path):
    from kyven_nuke.input_cache import export_input
    nuke, group = graph
    nuke._frame = 400
    state, cancellation = _running_state(nuke, group)
    signature = state.active
    monkeypatch.setattr(dlss_node, "_dlss_cache_root", lambda node: tmp_path)
    writer = dlss_node._inside(group, "KyvenDlssInputWrite")
    path = tmp_path / "_kyven_work/inputs/input.0400.exr"
    def export(writer, first, last):
        Path(writer["file"].value()).write_bytes(b"prepared pixels")
    monkeypatch.setattr(nuke, "execute", export, raising=False)
    export_input(nuke, group, writer, 400, path)
    monkeypatch.setattr(nuke, "execute", lambda *args: pytest.fail("Process must not render inputs"))
    _, _, validated = dlss_node._export(group.fullName(), 400, "check",
                                      {"temporal": False, "mask_source": "none"}, False, signature)
    assert validated == signature
    assert not cancellation.is_set()


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
    writers = [child for child in children.values() if child.kind == "Write"]
    assert len(writers) == 1
    writer = writers[0]
    assert writer is children["KyvenDlssInputWrite"]
    assert writer.input(0) is children["KyvenDlssInputBounds"]
    assert writer["channels"].value() == "all"
    assert writer["file_type"].value() == "exr"
    assert writer["datatype"].value() == "32 bit float"
    assert writer["compression"].value() == "Zip (16 scanlines)"
    assert writer["raw"].value()
    assert not writer["autocrop"].value()
    copy = children["KyvenDlssSourceChannels"]
    assert copy.input(0) is children["Source"]
    assert copy.input(1) is children["KyvenDlssRead"]
    assert {copy[f"to{i}"].value() for i in range(3)} == {
        "rgba.red", "rgba.green", "rgba.blue"}
    assert all(node.kind not in {"Premult", "Unpremult"} for node in children.values())
    assert not nuke.stack


def test_packed_input_export_remaps_guides_and_reads_one_cached_snapshot(graph, monkeypatch, tmp_path):
    from kyven_nuke import dlss_inputs, input_cache

    nuke, group = graph
    guides = nuke.source
    guides.channels = lambda: ["rgba.red", "rgba.green", "rgba.blue", "rgba.alpha", "depth.Z"]
    for index in (1, 2, 3):
        group.setInput(index, guides)
    settings = {"temporal": True, "motion_consistency": True, "motion_x": "R", "motion_y": "G",
                "forward_x": "B", "forward_y": "A", "depth_channel": "depth.Z", "mask_source": "input"}
    monkeypatch.setattr(dlss_node, "_request_settings", lambda node: settings)
    monkeypatch.setattr(dlss_node, "_operation_node", lambda name: group)
    group["motion_x"].setValue("R")
    group["motion_y"].setValue("G")
    group["mask_source"].setValue("Mask Input")
    specs = input_cache._specs(group, 1, 1, 1)
    assert len(specs) == 1
    writer, path = specs[0]
    assert path.name == "input.0001.exr"
    pack = group.children["KyvenDlssPackMotion"]
    assert not pack["disable"].value()
    assert [pack[f"from{i}"].value() for i in range(4)] == [
        "rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"]
    assert [pack[f"to{i}"].value() for i in range(4)] == list(dlss_inputs.MOTION)
    nuke.execute = Mock(side_effect=lambda writer, *_: Path(writer["file"].value()).write_bytes(b"snapshot"))
    assert input_cache.export_input(nuke, group, writer, 1, path)
    assert not input_cache.export_input(nuke, group, writer, 1, path)
    group["intensity"].setValue(1.5)
    assert input_cache.ready(group, path, 1)
    payload, _, _ = dlss_node._export(group.fullName(), 1, "packed", settings, False)
    assert {payload[field] for field in ("source", "motion", "depth", "mask")} == {str(path)}
    assert payload["mask_channel"] == dlss_inputs.MASK
    assert payload["depth_channel"] == dlss_inputs.DEPTH
    assert payload["motion_x"] == dlss_inputs.MOTION[0]
    assert nuke.execute.call_count == 1
    group["motion_x"].setValue("B")
    assert not input_cache.ready(group, path, 1)
    group["motion_x"].setValue("R")
    assert input_cache.ready(group, path, 1)
    group.setInput(2, None)
    assert not input_cache.ready(group, path, 1)


@pytest.mark.parametrize("failure", ["size", "channel"])
def test_pack_rejects_invalid_guides_instead_of_writing_black(graph, failure):
    nuke, group = graph
    guide = Node(nuke, "Constant", "BadGuide")
    group.setInput(1, guide)
    if failure == "size":
        guide.width = lambda: 123
    with pytest.raises(RuntimeError, match="match Source|missing channels"):
        dlss_node._packed_input_writer(group, {"mask_source": "none", "temporal": False})


def test_packed_input_ignores_unused_guides_and_upgrades_legacy_writers(graph):
    from kyven_nuke import dlss_inputs

    nuke, group = graph
    identity = group["kyven_uuid"].value()
    node_inputs = dict(group.inputs)
    group.begin()
    try:
        for name in ("Source", "Motion", "Depth", "Mask"):
            nuke.nodes.Write(name="KyvenDlss" + name + "Write")
    finally:
        group.end()
    group.setInput(3, Node(nuke, "Constant", "IgnoredMask"))
    dlss_node._packed_input_writer(group, {"mask_source": "none", "temporal": False})
    assert group["kyven_uuid"].value() == identity
    assert group.input(0) == node_inputs[0]
    assert sum(child.kind == "Write" for child in group.children.values()) == 1
    for name in ("Motion", "Depth", "Mask"):
        assert group.children["KyvenDlssPack" + name]["disable"].value()
    assert dlss_inputs.payload(group, {"mask_source": "source_alpha"}, Path("test.exr")) == {
        "source": "test.exr"}


def test_packed_range_writes_one_file_per_frame_and_only_adds_missing_frames(graph, monkeypatch):
    from kyven_nuke import input_cache

    nuke, group = graph
    monkeypatch.setattr(dlss_node, "_request_settings", lambda node: {"temporal": False, "mask_source": "none"})
    progress = Mock()
    progress.isCancelled.return_value = False
    monkeypatch.setattr(nuke, "ProgressTask", lambda *_: progress, raising=False)
    monkeypatch.setattr(nuke, "frame", lambda frame=None: 1)
    nuke.execute = Mock(side_effect=lambda writer, first, last:
                        Path(writer["file"].value()).write_bytes(f"pixels:{first}".encode()))
    assert input_cache.prepare_range(nuke, group, 1, 3)
    assert nuke.execute.call_count == 3
    assert input_cache.prepare_range(nuke, group, 2, 4)
    assert nuke.execute.call_count == 4
    group["intensity"].setValue(0.5)
    assert input_cache.prepare_range(nuke, group, 1, 4)
    assert nuke.execute.call_count == 4
    assert input_cache.prepare_range(nuke, group, 2, 3, refresh=True)
    assert nuke.execute.call_count == 6
    folder = dlss_node._dlss_cache_root(group) / "_kyven_work/inputs"
    assert {path.name for path in folder.glob("*.exr")} == {
        f"input.{frame:04d}.exr" for frame in range(1, 5)}


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


@pytest.mark.parametrize("bridge_discards_result", [False, True])
@pytest.mark.parametrize("render_context_advance", [False, True])
def test_source_to_live_result_survives_nuke_export_and_read_notifications(
        graph, tmp_path, monkeypatch, render_context_advance, bridge_discards_result):
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
    if bridge_discards_result:
        def bridge(function):
            nuke.executeInMainThread(function)
        monkeypatch.setattr(nuke, "executeInMainThreadWithResult", bridge)
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
        from kyven_nuke.input_cache import export_input
        input_path = dlss_node._dlss_cache_root(group) / "_kyven_work/inputs/input.0001.exr"
        export_input(nuke, group, dlss_node._inside(group, "KyvenDlssInputWrite"), 1, input_path)
        nuke._frame = 1
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


def test_blank_cache_folder_is_required_and_never_uses_automatic_cache(graph, monkeypatch):
    _, group = graph
    group["cache_root"].setValue(" ")
    monkeypatch.setattr(dlss_node.config, "cache_dir", lambda: pytest.fail("No automatic cache allowed"))
    with pytest.raises(RuntimeError, match="Choose Cache Folder"):
        dlss_node._public_pattern(group)


@pytest.mark.parametrize("action", ["current", "range", "live", "cache", "refresh"])
def test_empty_cache_folder_blocks_every_dlss_entry_before_export_or_server(graph, monkeypatch, action):
    from kyven_nuke import input_cache
    nuke, group = graph
    group["cache_root"].setValue("")
    monkeypatch.setattr(nuke, "thisNode", lambda: group, raising=False)
    popup = Mock()
    monkeypatch.setattr(nuke, "message", popup, raising=False)
    monkeypatch.setattr(nuke, "execute", lambda *args: pytest.fail("Must not export"), raising=False)
    monkeypatch.setattr(dlss_node, "ensure_server", lambda **kwargs: pytest.fail("Must not start server"))
    monkeypatch.setattr(dlss_node.config, "cache_dir", lambda: pytest.fail("Must not use fallback"))
    if action == "current":
        dlss_node.process_current_frame(group)
    elif action == "live":
        dlss_node.process_current_frame(group, live=True)
    elif action == "range":
        dlss_node.process_frame_range()
    else:
        input_cache.cache_frames(refresh=action == "refresh")
    assert "Choose Cache Folder" in group["kyven_status"].value()
    if action == "live":
        popup.assert_not_called()
    else:
        popup.assert_called_once()
    assert not group["kyven_busy"].value()


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
    signature = dlss_node._signature(group)
    job = {"result": {"output": str(staging), "metadata": {
        "history_reason": "continuous", "temporal_samples_used": 3, "history_capacity": 3,
        "temporal_reused_fraction": 0.66, "temporal_mean_weight": 0.17}}}
    assert dlss_node._publish(group.fullName(), job, pattern, 1, identity, "manual", False,
                              cancellation)
    assert (tmp_path / "dlss_v001.0001.exr").read_bytes() == b"finished pixels"
    assert (tmp_path / "_kyven_work/manifests/dlss_v001.0001.exr.kyven.json").is_file()
    assert staging.exists()
    target = tmp_path / "dlss_v001.0001.exr"
    previous_mtime = target.stat().st_mtime_ns
    assert dlss_node._publish(group.fullName(), job, pattern, 1, identity, "manual", False,
                              cancellation)
    assert target.stat().st_mtime_ns == previous_mtime
    assert group.children["KyvenDlssResultSwitch"]["which"].value() == 1
    assert "3/3 previous frames used" in group["history_status"].value()
    assert "reuse 66%" in group["history_status"].value()
    assert dlss_node._signature(group) == signature


def test_main_thread_callback_error_survives_nuke_discarding_return(graph, monkeypatch):
    nuke, _ = graph
    def bridge(function):
        function()
    monkeypatch.setattr(nuke, "executeInMainThreadWithResult", bridge)
    def export_failure():
        raise RuntimeError("Could not export source at frame 42: missing input file")
    with pytest.raises(RuntimeError, match="source at frame 42: missing input file"):
        dlss_node._main_thread_result(export_failure)


def test_missing_vectors_rejected_before_background_job(graph, monkeypatch):
    _, group = graph
    monkeypatch.setattr(dlss_node, "resolve_worker", lambda *args: Path(sys.executable))
    group["temporal"].setValue(True)
    dlss_node.process_current_frame(group)
    assert not group["kyven_busy"].value()
    assert "Motion" in group["kyven_status"].value()


def test_process_without_prepared_inputs_preserves_public_result(graph, monkeypatch, tmp_path):
    nuke, group = graph
    monkeypatch.setattr(dlss_node, "_operation_node", lambda name: group)
    monkeypatch.setattr(dlss_node, "_dlss_cache_root", lambda node: tmp_path)
    public = Path(str(dlss_node._public_pattern(group)) % 1)
    public.parent.mkdir(parents=True, exist_ok=True)
    public.write_bytes(b"previous completed result")
    monkeypatch.setattr(nuke, "execute", lambda *args: pytest.fail("Process must not export"), raising=False)
    with pytest.raises(RuntimeError, match="Cache Frames"):
        dlss_node._export(group.fullName(), 1, "missing", {"temporal": False, "mask_source": "none"}, False)
    assert public.read_bytes() == b"previous completed result"


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
