"""Kyven Segment Group node and background job orchestration."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any

from kyven_nuke import config
from kyven_nuke.payload import MODEL_LABELS, segment_payload, segment_video_payload
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
        enabled = bool(node[f"{kind}_enabled"].value())
        count = _point_count(node, kind)
        for index, name in enumerate(_point_knob_names(kind, MAX_PROMPT_POINTS), start=1):
            if name in node.knobs():
                node[name].setVisible(enabled and index <= count)
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
    knob = node[_point_knob_name(kind, new_count)]
    previous = node[_point_knob_name(kind, count)].value()
    knob.setValue([float(previous[0]) + 20.0, float(previous[1]) + 20.0])
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
    node[f"{kind}_point_count"].setValue(count - 1)
    sync_prompt_visibility(node)
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


def _apply_video_result(node_name: str, job: dict[str, Any]) -> None:
    node = _nuke().toNode(node_name)
    if node is None:
        return
    if job["status"] != "succeeded":
        error = job.get("error") or {}
        node["kyven_status"].setValue(f"Propagation failed: {error.get('message', job['status'])}")
        return
    result = job["result"]
    output_pattern = _nuke_file_path(Path(result["output_pattern"]))
    first = int(result["first_frame"])
    last = int(result["last_frame"])
    _set_matte_read(node, output_pattern, first, last)
    node["kyven_status"].setValue(
        f"SAM 2 tracking ready: {first}-{last} ({result['direction']})"
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


def _submit_video_and_wait(node_name: str, payload: dict[str, Any]) -> None:
    nuke = _nuke()
    try:
        client = ensure_server()
        job_id = client.submit_video(payload)
        nuke.executeInMainThread(_set_job_id, args=(node_name, job_id))
        nuke.executeInMainThread(
            _set_status,
            args=(node_name, "SAM 2 is propagating the key-frame mask..."),
        )
        job = client.wait(job_id, timeout_seconds=3600.0)
        nuke.executeInMainThread(_apply_video_result, args=(node_name, job))
    except Exception as exc:  # noqa: BLE001 - background boundary must report to the host UI
        nuke.executeInMainThread(_set_status, args=(node_name, f"Propagation failed: {exc}"))


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


def _video_cache_paths(node: Any, first: int, last: int) -> tuple[Path, Path, Path]:
    root = _cache_root(node)
    frames_dir = root / f"sam2_video_{first}_{last}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("*.jpg"):
        stale.unlink(missing_ok=True)
    source_pattern = frames_dir / "%05d.jpg"
    output_pattern = root / "tracked_matte.%04d.png"
    return frames_dir, source_pattern, output_pattern


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


def set_key_frame_to_current() -> None:
    nuke = _nuke()
    node = nuke.thisNode()
    frame = int(nuke.frame())
    node["key_frame"].setValue(frame)
    node["kyven_status"].setValue(f"Key frame set to {frame}.")


def propagate_video(direction: str) -> None:
    nuke = _nuke()
    node = nuke.thisNode()
    source = node.input(0)
    if source is None:
        nuke.message("Kyven Segment requires a Source input.")
        return
    if not _has_prompts(node):
        nuke.message("Enable at least one point or box prompt on the key frame.")
        return
    if direction not in {"forward", "backward", "both"}:
        nuke.message(f"Unsupported SAM 2 propagation direction: {direction}")
        return
    first = int(node["range_first"].value())
    last = int(node["range_last"].value())
    key_frame = int(node["key_frame"].value())
    if last < first:
        nuke.message("Kyven frame range requires Last to be greater than or equal to First.")
        return
    if not first <= key_frame <= last:
        nuke.message("Key Frame must be inside Range First/Last.")
        return

    frames_dir, source_pattern, output_pattern = _video_cache_paths(node, first, last)
    writer = _inside(node, "KyvenVideoWrite")
    writer["file"].setValue(_nuke_file_path(source_pattern))
    total = last - first + 1
    node["kyven_status"].setValue(f"Exporting {total} tracking frames...")
    try:
        nuke.execute(writer, first, last)
    except Exception as exc:  # noqa: BLE001 - host boundary must report useful context
        node["kyven_status"].setValue(f"Tracking export failed: {exc}")
        nuke.message(f"Kyven tracking export failed:\n{exc}")
        return
    exported = list(frames_dir.glob("*.jpg"))
    if len(exported) != total:
        node["kyven_status"].setValue(
            f"Tracking export failed: expected {total} JPEGs, found {len(exported)}."
        )
        return

    payload = segment_video_payload(
        frames_dir=str(frames_dir.resolve()),
        output_pattern=str(output_pattern.resolve()),
        model_index=int(node["model"].getValue()),
        profile=str(node["profile"].value()),
        image_height=int(source.height()),
        positive_points=_collect_points(node, "positive"),
        negative_points=_collect_points(node, "negative"),
        box_enabled=bool(node["box_enabled"].value()),
        box=tuple(node["prompt_box"].value()),
        first_frame=first,
        last_frame=last,
        key_frame=key_frame,
        direction=direction,
    )
    node["kyven_status"].setValue("Starting SAM 2 video predictor...")
    threading.Thread(
        target=_submit_video_and_wait,
        args=(node.fullName(), payload),
        name="kyven-sam2-video",
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


def _add_knob(nuke: Any, node: Any, knob: Any, *, start_line: bool = True) -> Any:
    if start_line:
        knob.setFlag(nuke.STARTLINE)
    else:
        knob.clearFlag(nuke.STARTLINE)
    node.addKnob(knob)
    return knob


def _add_section(nuke: Any, node: Any, name: str, title: str) -> None:
    _add_knob(nuke, node, nuke.Text_Knob(name, "", f"<b>{title}</b>"))


def create_segment_node() -> Any:
    nuke = _nuke()
    selected = nuke.selectedNode() if nuke.selectedNodes() else None
    node = nuke.nodes.Group(name="KyvenSegment")
    node.setInput(0, selected)
    node["label"].setValue("[value kyven_status]")
    node.addKnob(nuke.Tab_Knob("kyven", "Kyven Segment"))

    _add_section(nuke, node, "model_section", "MODEL AND PERFORMANCE")
    _add_knob(nuke, node, nuke.Enumeration_Knob("model", "Model", list(MODEL_LABELS)))
    node["model"].setValue(1)
    _add_knob(
        nuke,
        node,
        nuke.Enumeration_Knob(
            "profile",
            "Memory Profile",
            ["low_memory", "balanced", "quality"],
        ),
    )
    node["profile"].setValue(1)
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "refresh_models",
            "Refresh Installed Models",
            "kyven_nuke.node.refresh_models()",
        ),
    )

    _add_section(nuke, node, "positive_section", "POSITIVE POINTS (object to keep)")
    _add_knob(nuke, node, nuke.Boolean_Knob("positive_enabled", "Enable Positive Points"))
    node["positive_enabled"].setValue(True)
    positive_count = nuke.Int_Knob("positive_point_count", "Positive Point Count")
    positive_count.setValue(1)
    positive_count.setVisible(False)
    _add_knob(nuke, node, positive_count)
    for index, name in enumerate(_point_knob_names("positive", MAX_PROMPT_POINTS), start=1):
        _add_knob(nuke, node, nuke.XY_Knob(name, f"Positive Point {index}"))
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "add_positive_point",
            "Add Positive Point",
            "kyven_nuke.node.add_point('positive')",
        ),
    )
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "remove_positive_point",
            "Remove Last Positive",
            "kyven_nuke.node.remove_point('positive')",
        ),
        start_line=False,
    )

    _add_section(nuke, node, "negative_section", "NEGATIVE POINTS (areas to remove)")
    _add_knob(nuke, node, nuke.Boolean_Knob("negative_enabled", "Enable Negative Points"))
    negative_count = nuke.Int_Knob("negative_point_count", "Negative Point Count")
    negative_count.setValue(1)
    negative_count.setVisible(False)
    _add_knob(nuke, node, negative_count)
    for index, name in enumerate(_point_knob_names("negative", MAX_PROMPT_POINTS), start=1):
        _add_knob(nuke, node, nuke.XY_Knob(name, f"Negative Point {index}"))
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "add_negative_point",
            "Add Negative Point",
            "kyven_nuke.node.add_point('negative')",
        ),
    )
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "remove_negative_point",
            "Remove Last Negative",
            "kyven_nuke.node.remove_point('negative')",
        ),
        start_line=False,
    )

    _add_section(nuke, node, "box_section", "PROMPT BOX (search area)")
    _add_knob(nuke, node, nuke.Boolean_Knob("box_enabled", "Enable Prompt Box"))
    _add_knob(nuke, node, nuke.BBox_Knob("prompt_box", "Prompt Box"))
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "reset_prompts",
            "Reset Points and Box to Input Size",
            "kyven_nuke.node.reset_prompts_to_input()",
        ),
    )

    _add_section(nuke, node, "processing_section", "INDEPENDENT FRAME PROCESSING")
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "process_frame",
            "Process Current Frame",
            "kyven_nuke.node.process_current_frame()",
        ),
    )
    first = nuke.Int_Knob("range_first", "Range First")
    first.setValue(int(nuke.root().firstFrame()))
    _add_knob(nuke, node, first)
    last = nuke.Int_Knob("range_last", "Range Last")
    last.setValue(int(nuke.root().lastFrame()))
    _add_knob(nuke, node, last)
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "process_range",
            "Process Range as Matte Sequence",
            "kyven_nuke.node.process_frame_range()",
        ),
    )
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob("cancel", "Cancel Processing", "kyven_nuke.node.cancel_current_job()"),
    )

    _add_section(nuke, node, "tracking_section", "SAM 2 VIDEO TRACKING")
    key_frame = nuke.Int_Knob("key_frame", "Key Frame")
    key_frame.setValue(int(nuke.frame()))
    _add_knob(nuke, node, key_frame)
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "set_key_frame",
            "Set Key Frame to Current",
            "kyven_nuke.node.set_key_frame_to_current()",
        ),
    )
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "propagate_forward",
            "Propagate Forward",
            "kyven_nuke.node.propagate_video('forward')",
        ),
    )
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "propagate_backward",
            "Propagate Backward",
            "kyven_nuke.node.propagate_video('backward')",
        ),
        start_line=False,
    )
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "propagate_both",
            "Propagate Both Directions",
            "kyven_nuke.node.propagate_video('both')",
        ),
    )
    _add_section(nuke, node, "status_section", "STATUS")
    status = nuke.String_Knob("kyven_status", "Status")
    status.setFlag(nuke.READ_ONLY)
    status.setValue("Ready")
    _add_knob(nuke, node, status)
    internal_id = nuke.String_Knob("kyven_uuid", "UUID")
    internal_id.setValue(uuid.uuid4().hex)
    internal_id.setVisible(False)
    _add_knob(nuke, node, internal_id)
    job_id = nuke.String_Knob("kyven_job_id", "Job ID")
    job_id.setVisible(False)
    _add_knob(nuke, node, job_id)
    node["knobChanged"].setValue("kyven_nuke.node.prompt_knob_changed()")

    node.begin()
    try:
        source = nuke.nodes.Input(name="Source")
        source["number"].setValue(0)
        black = nuke.nodes.Multiply(name="KyvenEmptyMatte", value=0.0)
        black.setInput(0, source)
        black["channels"].setValue("rgba")
        switch = nuke.nodes.Switch(name="KyvenMatteSwitch")
        switch.setInput(0, black)
        switch["which"].setValue(0)
        output = nuke.nodes.Output(name="Output")
        output.setInput(0, switch)
        writer = nuke.nodes.Write(name="KyvenSourceWrite")
        writer.setInput(0, source)
        writer["file_type"].setValue("png")
        writer["channels"].setValue("rgb")
        video_writer = nuke.nodes.Write(name="KyvenVideoWrite")
        video_writer.setInput(0, source)
        video_writer["file_type"].setValue("jpeg")
        video_writer["channels"].setValue("rgb")
        if "_jpeg_quality" in video_writer.knobs():
            video_writer["_jpeg_quality"].setValue(1.0)
    finally:
        node.end()
    _reset_prompts(node)
    sync_prompt_visibility(node)
    return node
