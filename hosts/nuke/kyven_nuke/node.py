"""Kyven Segment Group node and background job orchestration."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any

from kyven_nuke import config
from kyven_nuke.payload import MODEL_LABELS, segment_payload
from kyven_nuke.runtime import ensure_server

MAX_PROMPT_POINTS = 32
_range_cancellations: set[str] = set()
_range_cancel_lock = threading.RLock()


def _nuke() -> Any:
    import nuke

    return nuke


def _inside(group: Any, name: str) -> Any:
    group.begin()
    try:
        return _nuke().toNode(name)
    finally:
        group.end()


def _nuke_file_path(path: Path) -> str:
    """Return an absolute path using the separators expected by Nuke file knobs."""
    return path.resolve().as_posix()


def _point_knob_name(kind: str, index: int) -> str:
    if kind not in {"positive", "negative"}:
        raise ValueError(f"Unsupported point kind: {kind}")
    return f"{kind}_point" if index == 1 else f"{kind}_point_{index}"


def _point_knob_names(kind: str, count: int) -> list[str]:
    return [_point_knob_name(kind, index) for index in range(1, count + 1)]


def _prompt_defaults(width: float, height: float) -> tuple[list[float], list[float]]:
    return [width / 2.0, height / 2.0], [0.0, 0.0, width, height]


def _point_count(node: Any, kind: str) -> int:
    return int(node[f"{kind}_point_count"].value())


def _collect_points(node: Any, kind: str) -> list[tuple[float, float]]:
    if not bool(node[f"{kind}_enabled"].value()):
        return []
    return [tuple(node[name].value()) for name in _point_knob_names(kind, _point_count(node, kind))]


def sync_prompt_visibility(node: Any | None = None) -> None:
    """Hide disabled prompt controls and their Viewer handles."""
    nuke = _nuke()
    node = node or nuke.thisNode()
    for kind in ("positive", "negative"):
        visible = bool(node[f"{kind}_enabled"].value())
        for name in _point_knob_names(kind, _point_count(node, kind)):
            if name in node.knobs():
                node[name].setVisible(visible)
    node["prompt_box"].setVisible(bool(node["box_enabled"].value()))


def prompt_knob_changed() -> None:
    nuke = _nuke()
    if nuke.thisKnob().name() in {"positive_enabled", "negative_enabled", "box_enabled"}:
        sync_prompt_visibility(nuke.thisNode())


def add_point(kind: str) -> None:
    nuke = _nuke()
    node = nuke.thisNode()
    count = _point_count(node, kind)
    if count >= MAX_PROMPT_POINTS:
        node["kyven_status"].setValue(f"Maximum {MAX_PROMPT_POINTS} {kind} points reached.")
        return
    new_count = count + 1
    label = f"{kind.title()} Point {new_count}"
    knob = nuke.XY_Knob(_point_knob_name(kind, new_count), label)
    previous = node[_point_knob_name(kind, count)].value()
    knob.setValue([float(previous[0]) + 20.0, float(previous[1]) + 20.0])
    node.addKnob(knob)
    node[f"{kind}_point_count"].setValue(new_count)
    sync_prompt_visibility(node)
    node["kyven_status"].setValue(f"Added {kind} point {new_count}.")


def remove_point(kind: str) -> None:
    nuke = _nuke()
    node = nuke.thisNode()
    count = _point_count(node, kind)
    if count <= 1:
        node["kyven_status"].setValue(f"Keep at least one {kind} point control.")
        return
    node.removeKnob(node[_point_knob_name(kind, count)])
    node[f"{kind}_point_count"].setValue(count - 1)
    node["kyven_status"].setValue(f"Removed {kind} point {count}.")


def _set_status(node_name: str, status: str) -> None:
    node = _nuke().toNode(node_name)
    if node is not None:
        node["kyven_status"].setValue(status)


def _set_matte_read(node: Any, output: str, first: int | None = None, last: int | None = None) -> None:
    nuke = _nuke()
    node.begin()
    try:
        matte = nuke.toNode("KyvenMatteRead")
        if matte is None:
            matte = nuke.nodes.Read(name="KyvenMatteRead", file=output)
            nuke.toNode("KyvenMatteSwitch").setInput(1, matte)
        else:
            matte["file"].setValue(output)
        if first is not None and last is not None:
            for knob_name, value in (
                ("first", first),
                ("last", last),
                ("origfirst", first),
                ("origlast", last),
            ):
                if knob_name in matte.knobs():
                    matte[knob_name].setValue(value)
        if "reload" in matte.knobs():
            matte["reload"].execute()
        nuke.toNode("KyvenMatteSwitch")["which"].setValue(1)
    finally:
        node.end()


def _apply_result(node_name: str, job: dict[str, Any]) -> None:
    nuke = _nuke()
    node = nuke.toNode(node_name)
    if node is None:
        return
    if job["status"] != "succeeded":
        error = job.get("error") or {}
        node["kyven_status"].setValue(f"Failed: {error.get('message', job['status'])}")
        return
    output = _nuke_file_path(Path(job["result"]["output"]))
    _set_matte_read(node, output)
    node["kyven_status"].setValue(f"Ready - score {float(job['result']['score']):.3f}")


def _apply_range_result(
    node_name: str,
    output_pattern: str,
    first: int,
    last: int,
    average_score: float,
) -> None:
    node = _nuke().toNode(node_name)
    if node is None:
        return
    _set_matte_read(node, output_pattern, first, last)
    node["kyven_status"].setValue(
        f"Range {first}-{last} ready - average score {average_score:.3f}"
    )


def _submit_and_wait(node_name: str, payload: dict[str, Any]) -> None:
    nuke = _nuke()
    try:
        client = ensure_server()
        job_id = client.submit_segment(payload)
        nuke.executeInMainThread(_set_job_id, args=(node_name, job_id))
        job = client.wait(job_id)
        nuke.executeInMainThread(_apply_result, args=(node_name, job))
    except Exception as exc:  # noqa: BLE001 - background boundary must report to the host UI
        nuke.executeInMainThread(_set_status, args=(node_name, f"Failed: {exc}"))


def _range_cancelled(node_name: str) -> bool:
    with _range_cancel_lock:
        return node_name in _range_cancellations


def _submit_range_and_wait(
    node_name: str,
    payloads: list[tuple[int, dict[str, Any]]],
    output_pattern: str,
    first: int,
    last: int,
) -> None:
    nuke = _nuke()
    scores: list[float] = []
    try:
        client = ensure_server()
        total = len(payloads)
        for position, (frame, payload) in enumerate(payloads, start=1):
            if _range_cancelled(node_name):
                nuke.executeInMainThread(
                    _set_status,
                    args=(node_name, f"Range cancelled before frame {frame}."),
                )
                return
            job_id = client.submit_segment(payload)
            nuke.executeInMainThread(_set_job_id, args=(node_name, job_id))
            nuke.executeInMainThread(
                _set_status,
                args=(node_name, f"Segmenting frame {frame} ({position}/{total})..."),
            )
            job = client.wait(job_id)
            if job["status"] != "succeeded":
                error = job.get("error") or {}
                message = error.get("message", job["status"])
                nuke.executeInMainThread(
                    _set_status,
                    args=(node_name, f"Frame {frame} failed: {message}"),
                )
                return
            scores.append(float(job["result"]["score"]))
        average_score = sum(scores) / len(scores)
        nuke.executeInMainThread(
            _apply_range_result,
            args=(node_name, output_pattern, first, last, average_score),
        )
    except Exception as exc:  # noqa: BLE001 - background boundary must report to the host UI
        nuke.executeInMainThread(_set_status, args=(node_name, f"Range failed: {exc}"))
    finally:
        with _range_cancel_lock:
            _range_cancellations.discard(node_name)


def _set_job_id(node_name: str, job_id: str) -> None:
    node = _nuke().toNode(node_name)
    if node is not None:
        node["kyven_job_id"].setValue(job_id)
        node["kyven_status"].setValue("Segmenting...")


def _cache_root(node: Any) -> Path:
    root = config.cache_dir() / node["kyven_uuid"].value()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cache_paths(node: Any, frame: int) -> tuple[Path, Path]:
    root = _cache_root(node)
    return root / f"source.{frame:04d}.png", root / f"matte.{frame:04d}.png"


def _cache_patterns(node: Any) -> tuple[Path, Path]:
    root = _cache_root(node)
    return root / "source.%04d.png", root / "matte.%04d.png"


def _path_for_frame(pattern: Path, frame: int) -> Path:
    return Path(str(pattern) % frame)


def _has_prompts(node: Any) -> bool:
    return bool(
        node["positive_enabled"].value()
        or node["negative_enabled"].value()
        or node["box_enabled"].value()
    )


def _payload_for_paths(node: Any, source: Any, source_path: Path, matte_path: Path) -> dict[str, Any]:
    return segment_payload(
        source=str(source_path.resolve()),
        output=str(matte_path.resolve()),
        model_index=int(node["model"].getValue()),
        profile=str(node["profile"].value()),
        image_height=int(source.height()),
        positive_points=_collect_points(node, "positive"),
        negative_points=_collect_points(node, "negative"),
        box_enabled=bool(node["box_enabled"].value()),
        box=tuple(node["prompt_box"].value()),
    )


def _reset_prompts(node: Any) -> bool:
    source = node.input(0)
    if source is None:
        return False
    width = float(source.width())
    height = float(source.height())
    center, prompt_box = _prompt_defaults(width, height)
    for kind in ("positive", "negative"):
        for index, name in enumerate(
            _point_knob_names(kind, _point_count(node, kind)),
            start=0,
        ):
            node[name].setValue([center[0] + index * 20.0, center[1] + index * 20.0])
    node["prompt_box"].setValue(prompt_box)
    node["kyven_status"].setValue(f"Prompts reset to input {int(width)}x{int(height)}.")
    return True


def reset_prompts_to_input() -> None:
    nuke = _nuke()
    if not _reset_prompts(nuke.thisNode()):
        nuke.message("Connect the Kyven Segment Source input first.")


def process_current_frame() -> None:
    nuke = _nuke()
    node = nuke.thisNode()
    source = node.input(0)
    if source is None:
        nuke.message("Kyven Segment requires a Source input.")
        return
    if not _has_prompts(node):
        nuke.message("Enable at least one positive point, negative point, or box prompt.")
        return

    frame = int(nuke.frame())
    source_path, matte_path = _cache_paths(node, frame)
    payload = _payload_for_paths(node, source, source_path, matte_path)

    writer = _inside(node, "KyvenSourceWrite")
    writer["file"].setValue(_nuke_file_path(source_path))
    node["kyven_status"].setValue("Exporting source frame...")
    try:
        nuke.execute(writer, frame, frame)
    except Exception as exc:  # noqa: BLE001 - host boundary must report useful context
        node["kyven_status"].setValue(f"Source export failed: {exc}")
        nuke.message(f"Kyven source export failed:\n{exc}")
        return
    if not source_path.is_file():
        node["kyven_status"].setValue("Source export failed: PNG was not created.")
        return
    node["kyven_status"].setValue("Starting Kyven server...")
    threading.Thread(
        target=_submit_and_wait,
        args=(node.fullName(), payload),
        name="kyven-nuke-submit",
        daemon=True,
    ).start()


def process_frame_range() -> None:
    nuke = _nuke()
    node = nuke.thisNode()
    source = node.input(0)
    if source is None:
        nuke.message("Kyven Segment requires a Source input.")
        return
    if not _has_prompts(node):
        nuke.message("Enable at least one positive point, negative point, or box prompt.")
        return
    first = int(node["range_first"].value())
    last = int(node["range_last"].value())
    if last < first:
        nuke.message("Kyven frame range requires Last to be greater than or equal to First.")
        return

    source_pattern, matte_pattern = _cache_patterns(node)
    writer = _inside(node, "KyvenSourceWrite")
    writer["file"].setValue(_nuke_file_path(source_pattern))
    total = last - first + 1
    node["kyven_status"].setValue(f"Exporting {total} source frames...")
    try:
        nuke.execute(writer, first, last)
    except Exception as exc:  # noqa: BLE001 - host boundary must report useful context
        node["kyven_status"].setValue(f"Range export failed: {exc}")
        nuke.message(f"Kyven range export failed:\n{exc}")
        return

    payloads = []
    for frame in range(first, last + 1):
        source_path = _path_for_frame(source_pattern, frame)
        if not source_path.is_file():
            node["kyven_status"].setValue(f"Range export failed: frame {frame} was not created.")
            return
        matte_path = _path_for_frame(matte_pattern, frame)
        payloads.append((frame, _payload_for_paths(node, source, source_path, matte_path)))

    node_name = node.fullName()
    with _range_cancel_lock:
        _range_cancellations.discard(node_name)
    node["kyven_status"].setValue(f"Starting range {first}-{last}...")
    threading.Thread(
        target=_submit_range_and_wait,
        args=(node_name, payloads, _nuke_file_path(matte_pattern), first, last),
        name="kyven-nuke-range",
        daemon=True,
    ).start()


def cancel_current_job() -> None:
    nuke = _nuke()
    node = nuke.thisNode()
    node_name = node.fullName()
    with _range_cancel_lock:
        _range_cancellations.add(node_name)
    job_id = node["kyven_job_id"].value()
    if not job_id:
        node["kyven_status"].setValue("Cancellation requested.")
        return

    def cancel() -> None:
        try:
            ensure_server().cancel(job_id)
            nuke.executeInMainThread(_set_status, args=(node_name, "Cancellation requested."))
        except Exception as exc:  # noqa: BLE001 - background boundary must report to the host UI
            nuke.executeInMainThread(_set_status, args=(node_name, f"Cancel failed: {exc}"))

    threading.Thread(target=cancel, name="kyven-nuke-cancel", daemon=True).start()


def refresh_models() -> None:
    nuke = _nuke()
    node = nuke.thisNode()
    node_name = node.fullName()

    def refresh() -> None:
        try:
            models = ensure_server().models()
            labels = []
            for model in models:
                suffix = "installed" if model["installed"] else "not installed"
                if model["compatible"] is False:
                    suffix += ", VRAM warning"
                labels.append(f"{model['display_name']} [{suffix}]")
            nuke.executeInMainThread(_apply_model_labels, args=(node_name, labels))
        except Exception as exc:  # noqa: BLE001 - background boundary must report to the host UI
            nuke.executeInMainThread(_set_status, args=(node_name, f"Model refresh failed: {exc}"))

    threading.Thread(target=refresh, name="kyven-model-refresh", daemon=True).start()


def _apply_model_labels(node_name: str, labels: list[str]) -> None:
    node = _nuke().toNode(node_name)
    if node is None or not labels:
        return
    previous = int(node["model"].getValue())
    node["model"].setValues(labels)
    node["model"].setValue(min(previous, len(labels) - 1))
    node["kyven_status"].setValue("Model list refreshed.")


def start_server() -> None:
    nuke = _nuke()

    def start() -> None:
        try:
            ensure_server()
            nuke.executeInMainThread(nuke.message, args=("Kyven Server is ready.",))
        except Exception as exc:  # noqa: BLE001 - background boundary must report to the host UI
            nuke.executeInMainThread(nuke.message, args=(f"Kyven Server failed: {exc}",))

    threading.Thread(target=start, name="kyven-server-start", daemon=True).start()


def create_segment_node() -> Any:
    nuke = _nuke()
    selected = nuke.selectedNode() if nuke.selectedNodes() else None
    node = nuke.nodes.Group(name="KyvenSegment")
    node.setInput(0, selected)
    node["label"].setValue("[value kyven_status]")
    node.addKnob(nuke.Tab_Knob("kyven", "Kyven Segment"))
    node.addKnob(nuke.Enumeration_Knob("model", "Model", list(MODEL_LABELS)))
    node["model"].setValue(1)
    node.addKnob(nuke.Enumeration_Knob("profile", "Execution Profile", ["low_memory", "balanced", "quality"]))
    node["profile"].setValue(1)
    node.addKnob(nuke.PyScript_Knob("refresh_models", "Refresh Models", "kyven_nuke.node.refresh_models()"))
    node.addKnob(nuke.Boolean_Knob("positive_enabled", "Show / Use Positive Points"))
    node["positive_enabled"].setValue(True)
    positive_count = nuke.Int_Knob("positive_point_count", "Positive Point Count")
    positive_count.setValue(1)
    positive_count.setVisible(False)
    node.addKnob(positive_count)
    node.addKnob(nuke.XY_Knob("positive_point", "Positive Point 1"))
    node.addKnob(
        nuke.PyScript_Knob(
            "add_positive_point",
            "Add Positive Point",
            "kyven_nuke.node.add_point('positive')",
        )
    )
    node.addKnob(
        nuke.PyScript_Knob(
            "remove_positive_point",
            "Remove Last Positive",
            "kyven_nuke.node.remove_point('positive')",
        )
    )
    node.addKnob(nuke.Boolean_Knob("negative_enabled", "Show / Use Negative Points"))
    negative_count = nuke.Int_Knob("negative_point_count", "Negative Point Count")
    negative_count.setValue(1)
    negative_count.setVisible(False)
    node.addKnob(negative_count)
    node.addKnob(nuke.XY_Knob("negative_point", "Negative Point 1"))
    node.addKnob(
        nuke.PyScript_Knob(
            "add_negative_point",
            "Add Negative Point",
            "kyven_nuke.node.add_point('negative')",
        )
    )
    node.addKnob(
        nuke.PyScript_Knob(
            "remove_negative_point",
            "Remove Last Negative",
            "kyven_nuke.node.remove_point('negative')",
        )
    )
    node.addKnob(nuke.Boolean_Knob("box_enabled", "Show / Use Box"))
    node.addKnob(nuke.BBox_Knob("prompt_box", "Prompt Box"))
    node.addKnob(
        nuke.PyScript_Knob(
            "reset_prompts",
            "Reset Prompts to Input",
            "kyven_nuke.node.reset_prompts_to_input()",
        )
    )
    node.addKnob(nuke.PyScript_Knob("process_frame", "Process Current Frame", "kyven_nuke.node.process_current_frame()"))
    first = nuke.Int_Knob("range_first", "Range First")
    first.setValue(int(nuke.root().firstFrame()))
    node.addKnob(first)
    last = nuke.Int_Knob("range_last", "Range Last")
    last.setValue(int(nuke.root().lastFrame()))
    node.addKnob(last)
    node.addKnob(
        nuke.PyScript_Knob(
            "process_range",
            "Process Frame Range",
            "kyven_nuke.node.process_frame_range()",
        )
    )
    node.addKnob(nuke.PyScript_Knob("cancel", "Cancel", "kyven_nuke.node.cancel_current_job()"))
    status = nuke.String_Knob("kyven_status", "Status")
    status.setFlag(nuke.READ_ONLY)
    status.setValue("Ready")
    node.addKnob(status)
    internal_id = nuke.String_Knob("kyven_uuid", "UUID")
    internal_id.setValue(uuid.uuid4().hex)
    internal_id.setVisible(False)
    node.addKnob(internal_id)
    job_id = nuke.String_Knob("kyven_job_id", "Job ID")
    job_id.setVisible(False)
    node.addKnob(job_id)
    node["knobChanged"].setValue("kyven_nuke.node.prompt_knob_changed()")

    node.begin()
    try:
        source = nuke.nodes.Input(name="Source")
        source["number"].setValue(0)
        black = nuke.nodes.Constant(name="KyvenEmptyMatte")
        black["color"].setValue([0.0, 0.0, 0.0, 0.0])
        switch = nuke.nodes.Switch(name="KyvenMatteSwitch")
        switch.setInput(0, black)
        switch["which"].setValue(0)
        output = nuke.nodes.Output(name="Output")
        output.setInput(0, switch)
        writer = nuke.nodes.Write(name="KyvenSourceWrite")
        writer.setInput(0, source)
        writer["file_type"].setValue("png")
        writer["channels"].setValue("rgb")
    finally:
        node.end()
    _reset_prompts(node)
    sync_prompt_visibility(node)
    return node
