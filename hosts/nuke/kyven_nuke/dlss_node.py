"""Thin Nuke DLSS adapter: float export, asynchronous jobs, durable Read results.

Only graph export and graph mutations run on the Nuke main thread. Model/NGX,
reprojection and image-file decoding run in the server, never in this module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from functools import wraps
from pathlib import Path
from typing import Any

from kyven.atomic_io import replace_with_retry
from kyven.dlss.runtime import WORKER_NAME, resolve_worker
from kyven_nuke import config
from kyven_nuke.color_management import set_data_io
from kyven_nuke.dlss_live import DlssCancellation, DlssLiveState
from kyven_nuke.node import (
    _add_knob,
    _artist_cache_name,
    _begin_node_operation,
    _cache_path_component,
    _cache_root_path,
    _create_group_at_cursor,
    _ensure_artist_cache_fields,
    _ensure_live_controls,
    _ensure_server_controls,
    _finish_progress,
    _inside,
    _job_error_text,
    _nuke,
    _nuke_file_path,
    _operation_node,
    _progress_cancelled,
    _sequence_frame_range,
    _set_busy,
    _set_status,
    _start_progress,
    _update_progress,
)
from kyven_nuke.runtime import ensure_server

_states: dict[str, DlssLiveState] = {}
_operations: dict[str, threading.Event] = {}
_mutations: dict[str, int] = {}
_live_timers: dict[str, threading.Timer] = {}
_polling: set[str] = set()
_logger = logging.getLogger(__name__)
_SETTINGS = (
    "worker", "temporal", "mask_source", "mask_channel", "motion_x", "motion_y",
    "motion_units", "motion_y_up", "motion_scale_x", "motion_scale_y", "depth_channel",
    "intensity", "local_tone", "local_structure", "skin_structure", "style", "preset",
    "temporal_strength", "rejection", "scene_cut", "protect_outside", "max_dimension",
    "reset_epoch", "cache_root", "dlss_name", "cache_version", "processing_scale",
    "hdr_rolloff", "rolloff_knee",
    "history_frames", "history_budget_mb", "history_clamp", "depth_rejection",
    "motion_consistency", "forward_x", "forward_y", "consistency_threshold",
)
_LIVE_SETTINGS = tuple(name for name in _SETTINGS
                       if name not in {"worker", "max_dimension", "mask_channel"})


def _main_thread_result(function: Any, *args: Any) -> Any:
    """Carry results/errors ourselves: Nuke can return None after a callback exception."""
    outcome: dict[str, Any] = {}

    def invoke() -> None:
        try:
            outcome["value"] = function(*args)
        except Exception as exc:
            outcome["error"] = f"{function.__name__}: {exc}"
            _logger.exception("DLSS main-thread callback failed: %s", function.__name__)

    _nuke().executeInMainThreadWithResult(invoke)
    if "error" in outcome:
        raise RuntimeError(outcome["error"])
    if "value" not in outcome:
        raise RuntimeError(f"Nuke did not execute the DLSS {function.__name__} callback.")
    return outcome["value"]


def _validate_inputs(node: Any, settings: dict[str, Any]) -> None:
    if node.input(0) is None:
        raise RuntimeError("Connect Source to Kyven DLSS.")
    if settings["temporal"] and node.input(1) is None:
        raise RuntimeError("Temporal requires external backward vectors on Motion. Connect them or disable Temporal.")
    if settings["mask_source"] == "input" and node.input(3) is None:
        raise RuntimeError("Connect the alpha/control guide to Mask (input 3).")


def _schedule_live(node: Any, delay: float = 0.1) -> None:
    """Keep watching edits and finish debounce even when Nuke emits no more UpdateUI calls."""
    identity = _identity(node)
    if not node["live_mode"].value():
        timer = _live_timers.pop(identity, None)
        if timer is not None:
            timer.cancel()
        return
    if identity in _live_timers:
        return
    name = str(node.fullName())

    def dispatch() -> None:
        _nuke().executeInMainThread(_live_tick, args=(name, identity, timer))

    timer = threading.Timer(delay, dispatch)
    timer.daemon = True
    _live_timers[identity] = timer
    timer.start()


def _live_tick(name: str, identity: str, timer: threading.Timer) -> None:
    if _live_timers.get(identity) is not timer:
        return
    _live_timers.pop(identity, None)
    nuke = _nuke()
    nuke.root().begin()
    try:
        node = nuke.toNode(name)
    finally:
        nuke.root().end()
    if node is None or "kyven_uuid" not in node.knobs() or _identity(node) != identity:
        state = _states.pop(identity, None)
        if state is not None:
            state.cancel.cancel("node_deleted")
        return
    try:
        if not nuke.executing():
            poll_live(node)
    except (RuntimeError, ValueError) as exc:
        # An upstream graph can be temporarily invalid while the artist is editing it.
        node["kyven_status"].setValue(f"DLSS waiting for valid inputs: {exc}")
    finally:
        _schedule_live(node)


def _suppress_node_callbacks(function: Any) -> Any:
    """Nuke can emit Group notifications while we export, reload Reads or switch output."""
    @wraps(function)
    def guarded(node: Any = None, *args: Any, **kwargs: Any) -> Any:
        target = _operation_node(node) if isinstance(node, str) else node
        if target is None and not isinstance(node, str):
            target = _nuke().thisNode()
        if target is None:
            return function(node, *args, **kwargs)
        name = str(target.fullName())
        _mutations[name] = _mutations.get(name, 0) + 1
        try:
            return function(node, *args, **kwargs)
        finally:
            depth = _mutations[name] - 1
            if depth:
                _mutations[name] = depth
            else:
                _mutations.pop(name, None)
    return guarded


def _settings_signature(node: Any) -> str:
    return hashlib.sha256(json.dumps({name: node[name].value() for name in _LIVE_SETTINGS},
                                    sort_keys=True).encode("utf-8")).hexdigest()


def _cancel(cancellation: threading.Event, reason: str) -> None:
    if isinstance(cancellation, DlssCancellation):
        cancellation.cancel(reason)
    else:
        cancellation.set()


def prune_live_nodes(nodes: list[Any]) -> None:
    identities = {_identity(node) for node in nodes
                  if "kyven_kind" in node.knobs() and node["kyven_kind"].value() == "dlss"}
    for identity in tuple(_states):
        if identity not in identities:
            # Inventories taken from a nested Group do not prove that a node was deleted.
            if _nuke().toNode(identity.rpartition(":")[0]) is not None:
                continue
            _states.pop(identity).cancel.cancel("node_deleted")
            timer = _live_timers.pop(identity, None)
            if timer is not None:
                timer.cancel()
            cancellation = _operations.get(identity)
            if cancellation is not None:
                _cancel(cancellation, "node_deleted")


def _identity(node: Any) -> str:
    return str(node.fullName()) + ":" + str(node["kyven_uuid"].value())


def _signature(node: Any) -> str:
    """Only the current frame and the node's own artist settings can request Live."""
    values = {name: node[name].value() for name in _LIVE_SETTINGS}
    values["frame"] = int(_nuke().frame())
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()


def poll_live(node: Any) -> None:
    """Debounce current-frame and own-setting edits only; never inspect upstream hashes."""
    _schedule_live(node)
    identity = _identity(node)
    if identity in _polling or _mutations.get(str(node.fullName()), 0):
        return
    _polling.add(identity)
    try:
        _poll_live(node, identity)
    finally:
        _polling.discard(identity)


def _poll_live(node: Any, identity: str) -> None:
    state = _states.setdefault(identity, DlssLiveState())
    settings = _settings_signature(node)
    if state.active and state.settings and settings != state.settings:
        _cancel(state.cancel, "settings_changed")
    state.settings = settings
    enabled = bool(node["live_mode"].value())
    if not enabled:
        state.observe("", time.monotonic(), False)
        return
    signature = _signature(node)
    now = time.monotonic()
    state.observe(signature, now)
    if (state.ready(now, float(node["live_delay"].value()))
            and not node["kyven_busy"].value() and node.input(0) is not None):
        process_current_frame(node, live=True)


def knob_changed() -> None:
    nuke = _nuke()
    node, name = nuke.thisNode(), nuke.thisKnob().name()
    if _mutations.get(str(node.fullName()), 0):
        return
    if name == "mask_source":
        node["mask_channel"].setVisible(False)
        node["protect_outside"].setVisible(node["mask_source"].value() != "Full Frame")
    if name in _LIVE_SETTINGS or name == "live_mode":
        _schedule_live(node)
        state = _states.setdefault(_identity(node), DlssLiveState())
        if name == "live_mode" and state.active and bool(node["live_mode"].value()) and state.live_active:
            return
        if name in _LIVE_SETTINGS and _settings_signature(node) == state.settings:
            return
        state.settings = _settings_signature(node)
        if name in {"cache_root", "dlss_name", "cache_version"}:
            _inside(node, "KyvenDlssResultSwitch")["which"].setValue(0)
        # An active range is a snapshot. Edits invalidate it before another frame is exported.
        cancellation = _operations.get(_identity(node))
        if cancellation is not None and (name != "live_mode" or (
                _states.get(_identity(node)) is not None
                and _states[_identity(node)].live_active)):
            reason = "live_disabled" if name == "live_mode" else "settings_changed"
            _cancel(cancellation, reason)
        if bool(node["live_mode"].value()):
            state = _states.setdefault(_identity(node), DlssLiveState())
            state.observe(_signature(node), time.monotonic())
        else:
            state = _states.get(_identity(node))
            if state is not None:
                state.observe("", time.monotonic(), False)


def reset_history() -> None:
    node = _nuke().thisNode()
    node["reset_epoch"].setValue(int(node["reset_epoch"].value()) + 1)
    _set_status(str(node.fullName()), "History reset. Process this frame or enable Live.")


@_suppress_node_callbacks
def _set_nuke_vector_channels(node: Any) -> None:
    for name, value in {"motion_x": "backward.u", "motion_y": "backward.v",
                        "forward_x": "forward.u", "forward_y": "forward.v",
                        "motion_units": "Pixels", "motion_y_up": True,
                        "motion_scale_x": 1.0, "motion_scale_y": 1.0}.items():
        node[name].setValue(value)


def use_nuke_vectors() -> None:
    """Explicit channel mapping only; never create an optical-flow node."""
    node = _nuke().thisNode()
    _set_nuke_vector_channels(node)
    poll_live(node)


def _public_pattern(node: Any) -> Path:
    return _dlss_cache_root(node) / (
        f"{_artist_cache_name(node)}_{str(node['cache_version'].value()).strip()}.%04d.exr"
    )


def _dlss_cache_root(node: Any) -> Path:
    if str(node["cache_root"].value()).strip():
        return _cache_root_path(node)
    identity = _cache_path_component(node["kyven_uuid"].value(), "UUID")
    version = _cache_path_component(node["cache_version"].value(), "Version")
    return config.cache_dir().resolve() / identity / _artist_cache_name(node) / version


def _request_settings(node: Any) -> dict[str, Any]:
    payload = {name: node[name].value() for name in _SETTINGS if name not in {
        "cache_root", "dlss_name", "cache_version", "reset_epoch", "processing_scale",
    }}
    worker = resolve_worker(config.runtime_dir())
    if worker is None:
        raise RuntimeError("DLSS is not ready. Open Model Manager > Kyven DLSS Runtime to install/check components.")
    payload["worker"] = str(worker)
    choice = str(node["mask_source"].value())
    payload["mask_source"] = {"Full Frame": "none", "None": "none", "Source Alpha": "source_alpha", "Mask Input": "input",
                              "Auto": "input" if node.input(3) is not None else "none"}[choice]
    payload["mask_channel"] = "A"
    payload["motion_units"] = ("pixels", "uv")[int(node["motion_units"].getValue())]
    payload["style"] = int(node["style"].getValue())
    payload["preset"] = int(node["preset"].getValue())
    source = node.input(0)
    if source is None:
        raise RuntimeError("Connect Source to Kyven DLSS.")
    source.forceValidate()
    payload["max_dimension"] = max(64, min(8192, round(
        max(source.width(), source.height()) * float(node["processing_scale"].value()))))
    payload["history_frames"] = int(node["history_frames"].value())
    payload["history_budget_mb"] = int(node["history_budget_mb"].value())
    payload["session"] = str(node["kyven_uuid"].value()) + ":" + str(node["reset_epoch"].value())
    return payload


@_suppress_node_callbacks
def _export(node_name: str, frame: int, request_id: str,
            settings: dict[str, Any], reset: bool, signature: str = "") -> tuple[dict[str, Any], str, str]:
    """Nuke graph evaluation must run on the main thread. No model code is imported."""
    nuke = _nuke()
    node = _operation_node(node_name)
    if node is None:
        raise RuntimeError("DLSS node was deleted.")
    _validate_inputs(node, settings)
    root = _dlss_cache_root(node) / "_kyven_work" / "requests" / request_id
    root.mkdir(parents=True, exist_ok=True)
    payload = dict(settings, frame=frame, reset=reset,
                   output=str(root / f"result.{frame:04d}.exr"))
    _set_status(node_name, f"Exporting DLSS frame {frame}...")
    try:
        for index, field in enumerate(("source", "motion", "depth", "mask")):
            if node.input(index) is None:
                continue
            if field == "mask" and settings["mask_source"] != "input":
                continue
            writer = _inside(node, "KyvenDlss" + field.title() + "Write")
            path = root / f"{field}.{frame:04d}.exr"
            payload[field] = str(path)
            writer["file"].setValue(_nuke_file_path(path))
            try:
                nuke.execute(writer, frame, frame)
            except Exception as exc:
                raise RuntimeError(f"Could not export {field} at frame {frame}: {exc}") from exc
    except Exception:
        _discard_request(payload)
        raise
    # Do not inspect Nuke's private export context (often last_frame + 1).
    # The original frame/settings intent remains immutable throughout this job.
    return payload, str(_public_pattern(node)), signature


@_suppress_node_callbacks
def _publish(node_name: str, job: dict[str, Any], pattern: str, frame: int,
             identity: str, signature: str, live: bool,
             expected_cancellation: threading.Event | None = None) -> bool:
    node = _operation_node(node_name)
    if node is None:
        return False
    if live:
        state = _states.get(identity)
        latest = _signature(node)
        if state is not None and latest != signature:
            state.observe(latest, time.monotonic())
        if not node["live_mode"].value() or state is None or not state.accepts(signature):
            return False
    cancellation = _operations.get(identity)
    if (cancellation is None or cancellation.is_set()
            or cancellation is not expected_cancellation):
        return False
    source = Path(job["result"]["output"])
    target = Path(pattern % frame)
    target.parent.mkdir(parents=True, exist_ok=True)
    replace_with_retry(source, target)
    manifest_source = source.with_suffix(source.suffix + ".kyven.json")
    manifest_target = target.parent / "_kyven_work" / "manifests" / (target.name + ".kyven.json")
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    if manifest_source.is_file():
        replace_with_retry(manifest_source, manifest_target)
    read = _inside(node, "KyvenDlssRead")
    read["file"].setValue(_nuke_file_path(Path(pattern)))
    frame_range = _sequence_frame_range(Path(pattern)) or (frame, frame)
    for name, value in zip(("first", "last", "origfirst", "origlast"), frame_range * 2):
        read[name].setValue(value)
    read["reload"].execute()
    _inside(node, "KyvenDlssResultSwitch")["which"].setValue(1)
    node["kyven_live_frame"].setValue(frame)
    metadata = job["result"]["metadata"]
    reason = metadata.get("history_reason", "")
    if "history_status" in node.knobs():
        samples = int(metadata.get("temporal_samples_used", 0))
        capacity = int(metadata.get("history_capacity", 0))
        reused = float(metadata.get("temporal_reused_fraction", 0))
        weight = float(metadata.get("temporal_mean_weight", 0))
        node["history_status"].setValue(
            f"{samples}/{capacity} previous frames used | "
            f"reuse {reused:.0%} | mean blend {weight:.0%} | {reason}")
    _set_status(str(node.fullName()), f"DLSS frame {frame} complete | {reason}")
    # Read.reload() above invalidates our result and its downstream consumers.
    # Clearing Nuke's global RAM cache also invalidates unrelated upstream graphs,
    # introducing feedback into Live signatures and other nodes' update callbacks.
    redraw = getattr(_nuke(), "redraw", None)
    if callable(redraw):
        redraw()
    return True


def load_cache() -> None:
    node = _nuke().thisNode()
    if node["kyven_busy"].value():
        _set_status(str(node.fullName()), "Cancel active DLSS processing before loading another cache.")
        return
    try:
        pattern = _public_pattern(node)
        frames = _sequence_frame_range(pattern)
        if frames is None:
            raise RuntimeError("No DLSS EXR sequence in this cache version.")
        read = _inside(node, "KyvenDlssRead")
        read["file"].setValue(_nuke_file_path(pattern))
        for name, value in zip(("first", "last", "origfirst", "origlast"), frames * 2):
            read[name].setValue(value)
        read["reload"].execute()
        _inside(node, "KyvenDlssResultSwitch")["which"].setValue(1)
        _set_status(str(node.fullName()), f"Loaded DLSS cache {frames[0]}–{frames[1]}")
    except RuntimeError as exc:
        _set_status(str(node.fullName()), str(exc))


@_suppress_node_callbacks
def _finish(node_name: str, identity: str, signature: str, message: str,
            cancellation: threading.Event) -> None:
    if _operations.get(identity) is not cancellation:
        return
    _operations.pop(identity, None)
    state = _states.get(identity)
    if state is not None:
        state.finish(state.active)
    _set_busy(node_name, False)
    _finish_progress(node_name)
    if message:
        _set_status(node_name, message)
    node = _operation_node(node_name)
    if node is not None:
        _schedule_live(node)


def _discard_request(payload: dict[str, Any]) -> None:
    """Remove only this request's known staging files after the server is finished."""
    root = Path(payload["output"]).parent
    for field in ("source", "motion", "depth", "mask", "output"):
        if field not in payload:
            continue
        path = Path(payload[field])
        if path.parent != root:
            continue
        for item in (path, path.with_suffix(path.suffix + ".kyven.json")):
            try:
                item.unlink(missing_ok=True)
            except OSError:
                pass
    try:
        root.rmdir()
    except OSError:
        pass


@_suppress_node_callbacks
def _start(node: Any, frames: range, live: bool) -> None:
    nuke = _nuke()
    node_name = str(node.fullName())
    if live:
        state = _states.setdefault(_identity(node), DlssLiveState())
        state.attempted = _signature(node)
    try:
        _public_pattern(node)
        settings = _request_settings(node)
        _validate_inputs(node, settings)
        if not Path(settings["worker"]).is_file():
            raise RuntimeError("Choose the local DLSS worker executable first.")
        if len(frames) > 100000:
            raise RuntimeError("DLSS range is limited to 100000 frames per job.")
    except (RuntimeError, ValueError) as exc:
        # No operation identity has been registered yet; use the supplied node.
        node["kyven_status"].setValue(str(exc))
        return
    if not _begin_node_operation(node):
        return
    identity = _identity(node)
    signature = _signature(node)
    state = _states.setdefault(identity, DlssLiveState())
    state.settings = _settings_signature(node)
    if live:
        state.observe(signature, time.monotonic())
    cancellation = state.begin(signature, live=live)
    _operations[identity] = cancellation
    request_id = uuid.uuid4().hex
    _inside(node, "KyvenDlssResultSwitch")["which"].setValue(0)
    if not live:
        _start_progress(node_name, "Kyven DLSS", "Preparing local DLSS inputs")
    _set_status(node_name, "Preparing DLSS...")

    def work() -> None:
        nonlocal signature
        message, client, job_id = "", None, ""
        try:
            client = ensure_server(progress_callback=lambda message: nuke.executeInMainThread(
                _set_status, args=(node_name, str(message))))
            for index, frame in enumerate(frames):
                if cancellation.is_set():
                    break
                exported = _main_thread_result(
                    _export, node_name, frame, request_id, settings, not live and index == 0, signature)
                if not isinstance(exported, tuple) or len(exported) != 3:
                    raise RuntimeError("DLSS export returned no usable inputs. Check the Nuke Script Editor for an export error.")
                payload, pattern, signature = exported
                if cancellation.is_set():
                    _discard_request(payload)
                    break
                job_id = client.submit_dlss(payload)
                deadline = time.monotonic() + 1800
                cancel_sent = False
                last_stage = ""
                while True:
                    if not live:
                        cancelled = _main_thread_result(_progress_cancelled, node_name)
                        if cancelled:
                            cancellation.cancel("user_cancelled")
                    if cancellation.is_set() and not cancel_sent:
                        client.cancel(job_id)
                        cancel_sent = True
                    job = client.job(job_id)
                    if job["status"] in {"succeeded", "failed", "cancelled"}:
                        break
                    if time.monotonic() >= deadline:
                        client.cancel(job_id)
                        raise RuntimeError("DLSS job timed out.")
                    if live:
                        stage = str(job.get("progress_message") or job.get("status") or "Working")
                        if stage != last_stage:
                            nuke.executeInMainThread(_set_status, args=(node_name, f"DLSS frame {frame} | {stage}"))
                            last_stage = stage
                    if not live:
                        progress = (index + float(job.get("progress", 0))) / len(frames)
                        nuke.executeInMainThread(_update_progress, args=(
                            node_name, int(progress * 100), f"DLSS frame {frame}"))
                    time.sleep(0.1)
                if cancellation.is_set() or job["status"] == "cancelled":
                    if not cancellation.is_set():
                        message = "DLSS was stopped by the server. Retry the current frame."
                    _discard_request(payload)
                    break
                if job["status"] != "succeeded":
                    _discard_request(payload)
                    raise RuntimeError(_job_error_text(job))
                accepted = _main_thread_result(
                    _publish, node_name, job, pattern, frame, identity, signature, live, cancellation)
                _discard_request(payload)
                if not accepted:
                    message = "Frame changed; waiting for the latest Live result." if live else "DLSS result was superseded."
                    break
                job_id = ""
            if cancellation.is_set():
                if cancellation.reason in {"superseded", "settings_changed", "inputs_changed"}:
                    message = ("Updating the latest Live frame..." if live else
                               "Inputs or settings changed; restart the frame range.")
                elif cancellation.reason == "node_deleted":
                    message = ""
                else:
                    message = "DLSS stopped. Completed cache frames are preserved."
        except Exception as exc:  # noqa: BLE001 - host worker boundary
            if client is not None and job_id:
                try:
                    client.cancel(job_id)
                except RuntimeError:
                    pass
            message = f"DLSS failed: {exc}"
        finally:
            nuke.executeInMainThread(_finish, args=(node_name, identity, signature, message,
                                                   cancellation))

    threading.Thread(target=work, name="kyven-dlss-host", daemon=True).start()


def process_current_frame(node: Any | None = None, live: bool = False) -> None:
    nuke = _nuke()
    frame = int(nuke.frame())
    _start(node or nuke.thisNode(), range(frame, frame + 1), live)


def process_frame_range() -> None:
    node = _nuke().thisNode()
    first, last = int(node["first_frame"].value()), int(node["last_frame"].value())
    if last < first:
        _set_status(str(node.fullName()), "Last frame must be at least First frame.")
        return
    _start(node, range(first, last + 1), False)


def cancel_current_job() -> None:
    node = _nuke().thisNode()
    if node["live_mode"].value():
        node["live_mode"].setValue(False)
    cancellation = _operations.get(_identity(node))
    if cancellation is not None:
        _cancel(cancellation, "user_cancelled")


def repair_dlss_nodes() -> None:
    """Reset transient work on script load without changing UUIDs or cached graph outputs."""
    for cancellation in tuple(_operations.values()):
        cancellation.set()
    _operations.clear()
    _states.clear()
    _polling.clear()
    for timer in tuple(_live_timers.values()):
        timer.cancel()
    _live_timers.clear()
    nuke = _nuke()
    for node in nuke.allNodes("Group", recurseGroups=True):
        if "kyven_kind" in node.knobs() and node["kyven_kind"].value() == "dlss":
            from kyven_nuke.dlss_interface import restyle_dlss_node

            restyle_dlss_node(node, upgrade=True)
            _ensure_output_graph(node)
            _set_busy(str(node.fullName()), False)
            node["kyven_live_frame"].setValue(-2147483647)
            refresh_runtime(node)
            _schedule_live(node)


@_suppress_node_callbacks
def refresh_runtime(node: Any | None = None) -> None:
    node = node or _nuke().thisNode()
    worker = resolve_worker(config.runtime_dir())
    if worker is None:
        own = config.runtime_dir() / "dlss" / "bin" / WORKER_NAME
        text = ("NVIDIA NR component required | Model Manager" if own.is_file()
                else "Install Kyven DLSS Runtime in Model Manager")
        node["model_help"].setValue(text)
        node["kyven_status"].setValue(text)
    else:
        node["worker"].setValue(str(worker))
        node["model_help"].setValue("Kyven runtime ready | External motion supported")
        node["kyven_status"].setValue("Ready | Live processes the connected Source")


@_suppress_node_callbacks
def _ensure_output_graph(node: Any) -> None:
    nuke = _nuke()
    node.begin()
    try:
        output = nuke.toNode("KyvenDlssOutputSwitch")
        if output is None:
            output = nuke.nodes.Switch(name="KyvenDlssOutputSwitch")
        output.setInput(0, nuke.toNode("KyvenDlssResultSwitch"))
        output.setInput(1, nuke.toNode("Source"))
        output["which"].setExpression("parent.output_mode")
        nuke.toNode("Output").setInput(0, output)
    finally:
        node.end()


def create_result_read() -> None:
    node = _nuke().thisNode()
    pattern = _public_pattern(node)
    frames = _sequence_frame_range(pattern)
    if frames is None:
        _set_status(str(node.fullName()), "Process a frame first.")
        return
    read = _nuke().nodes.Read(file=_nuke_file_path(pattern))
    set_data_io(read)
    for name, value in zip(("first", "last", "origfirst", "origlast"), frames * 2):
        read[name].setValue(value)


def new_cache_version() -> None:
    import re

    node = _nuke().thisNode()
    if node["kyven_busy"].value():
        return
    old = str(node["cache_version"].value())
    match = re.fullmatch(r"(.*?)(\d+)", old)
    prefix, value = (match[1], int(match[2]) + 1) if match else (old + "_", 1)
    while True:
        candidate = f"{prefix}{value:03d}"
        parent = _dlss_cache_root(node).parent
        if not (parent / candidate).exists():
            break
        value += 1
    node["cache_version"].setValue(candidate)


def upgrade_selected_dlss_node() -> None:
    from kyven_nuke.dlss_interface import restyle_dlss_node

    for node in _nuke().selectedNodes():
        if "kyven_kind" in node.knobs() and node["kyven_kind"].value() == "dlss":
            restyle_dlss_node(node, upgrade=True)
            _ensure_output_graph(node)
            refresh_runtime(node)


def create_dlss_node() -> Any:
    nuke = _nuke()
    selected = nuke.selectedNodes()
    source = selected[-1] if selected else None
    node = _create_group_at_cursor(nuke, "KyvenDLSS")
    node["tile_color"].setValue(0x485D6FFF)
    _add_knob(nuke, node, nuke.Tab_Knob("kyven", "Kyven DLSS"))

    def add(kind: str, name: str, label: str, value: Any, choices=None,
            limits=None, tooltip: str = "", hidden: bool = False) -> Any:
        knob = (getattr(nuke, kind)(name, label, choices) if choices is not None
                else getattr(nuke, kind)(name, label))
        knob.setValue(value)
        if limits is not None:
            knob.setRange(*limits)
        knob.setTooltip(tooltip)
        knob.setVisible(not hidden)
        _add_knob(nuke, node, knob)
        return knob

    add("String_Knob", "kyven_uuid", "UUID", uuid.uuid4().hex, hidden=True)
    add("Int_Knob", "reset_epoch", "History generation", 0, hidden=True)
    default_worker = config.runtime_dir() / "dlss" / "bin" / WORKER_NAME
    add("File_Knob", "worker", "DLSS Worker", str(default_worker),
        tooltip="Managed by Model Manager. Kyven uses its own worker inside the project.")
    _ensure_artist_cache_fields(node, name_field="dlss_name", name_label="Name", name_default="dlss")
    add("Text_Knob", "guide_help", "", "Source / Motion / Depth / Mask. Alpha is a DLSS control guide.")
    add("Boolean_Knob", "temporal", "Temporal (external vectors required)", False)
    _ensure_live_controls(node, "dlss")
    node["live_mode"].setValue(True)
    add("Double_Knob", "live_delay", "Live Debounce (s)", 0.25, limits=(0.1, 2))
    add("Enumeration_Knob", "mask_source", "Processing Area", "Full Frame",
        choices=["Full Frame", "Mask Input", "Source Alpha"],
        tooltip="Choose the neural control guide. White allows changes, black protects. Source alpha is preserved.")
    add("String_Knob", "mask_channel", "Mask Channel", "A")
    add("Boolean_Knob", "protect_outside", "Protect RGB Outside Mask", True)
    add("String_Knob", "motion_x", "Motion X Channel", "R")
    add("String_Knob", "motion_y", "Motion Y Channel", "G")
    add("Enumeration_Knob", "motion_units", "Motion Units", "Pixels", choices=["Pixels", "UV"])
    add("Boolean_Knob", "motion_y_up", "Motion Y Points Up (Nuke)", True,
        tooltip="Vectors must map current pixels to previous frame. Forward flow cannot be fixed by negation.")
    for axis in ("x", "y"):
        add("Double_Knob", "motion_scale_" + axis, "Motion Scale " + axis.upper(), 1.0,
            limits=(-10, 10))
    add("String_Knob", "depth_channel", "Depth Channel", "depth.Z")
    for name, label, value, limits in (
        ("intensity", "Intensity", 1.0, (0, 2)),
        ("local_tone", "Local Tone", 1.0, (0, 2)),
        ("local_structure", "Local Structure", 1.0, (0, 2)),
        ("skin_structure", "Skin Structure (-1 = inherit)", -1.0, (-1, 2)),
        ("temporal_strength", "Residual Stability", 0.35, (0, 0.95)),
        ("rejection", "History Rejection Threshold", 0.1, (0.001, 1)),
        ("scene_cut", "Scene Cut Threshold", 0.45, (0.01, 2)),
    ):
        add("Double_Knob", name, label, value, limits=limits)
    add("Enumeration_Knob", "style", "NR Style", "Default",
        choices=["Default", "Natural", "Cinematic"])
    add("Enumeration_Knob", "preset", "NR Preset", "Default",
        choices=["Default", "Preset 1", "Preset 2", "Preset 3"])
    add("Int_Knob", "max_dimension", "Processing Limit (px)", 1024, limits=(64, 8192),
        tooltip="Default 1024 limits GPU use. Neural residual is restored at Source resolution. "
                "4 GB VRAM compatibility depends on your runtime and must be tested.")
    add("Int_Knob", "first_frame", "First Frame", int(nuke.root()["first_frame"].value()))
    add("Int_Knob", "last_frame", "Last Frame", int(nuke.root()["last_frame"].value()))
    for name, label, function in (
        ("process_frame", "Current Frame", "process_current_frame"),
        ("process_range", "Frame Range", "process_frame_range"),
        ("cancel", "Cancel", "cancel_current_job"),
        ("reset_history", "Reset History", "reset_history"),
        ("load_cache", "Load Cache", "load_cache"),
    ):
        _add_knob(nuke, node, nuke.PyScript_Knob(
            name, label, f"kyven_nuke.dlss_node.{function}()"))
    add("String_Knob", "kyven_status", "Status", "Connect Source and choose a local DLSS worker.")
    node["kyven_status"].setFlag(nuke.READ_ONLY)
    _ensure_server_controls(node)
    node.begin()
    try:
        inputs = []
        for i, name in enumerate(("Source", "Motion", "Depth", "Mask")):
            inp = nuke.nodes.Input(name=name)
            inp["number"].setValue(i)
            inputs.append(inp)
            writer = nuke.nodes.Write(name="KyvenDlss" + name + "Write")
            writer.setInput(0, inp)
            writer["file_type"].setValue("exr")
            writer["channels"].setValue("rgba" if i == 0 else "all")
            writer["datatype"].setValue("32 bit float")
            if "autocrop" in writer.knobs():
                writer["autocrop"].setValue(False)
            set_data_io(writer)
        read = nuke.nodes.Read(name="KyvenDlssRead")
        read["on_error"].setValue("black")
        set_data_io(read)
        # Copy only the generated RGB. Alpha and auxiliary channels stay with Source.
        copy = nuke.nodes.Copy(name="KyvenDlssSourceChannels")
        copy.setInput(0, inputs[0])
        copy.setInput(1, read)
        for index, component in enumerate(("red", "green", "blue")):
            copy[f"from{index}"].setValue("rgba." + component)
            copy[f"to{index}"].setValue("rgba." + component)
        switch = nuke.nodes.Switch(name="KyvenDlssResultSwitch")
        switch.setInput(0, inputs[0])
        switch.setInput(1, copy)
        output = nuke.nodes.Output(name="Output")
        output.setInput(0, switch)
    finally:
        node.end()
    if source is not None:
        node.setInput(0, source)
    node["knobChanged"].setValue("kyven_nuke.dlss_node.knob_changed()")
    from kyven_nuke.dlss_interface import restyle_dlss_node

    restyle_dlss_node(node)
    _ensure_output_graph(node)
    refresh_runtime(node)
    _schedule_live(node)
    return node
