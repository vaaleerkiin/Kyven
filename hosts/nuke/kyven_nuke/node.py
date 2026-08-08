"""Kyven Segment Group node and background job orchestration."""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from kyven_nuke import config
from kyven_nuke.branding import add_node_branding
from kyven_nuke.color_management import set_data_io
from kyven_nuke.payload import MODEL_LABELS, segment_payload, segment_video_payload
from kyven_nuke.runtime import ensure_server, stop_server

MAX_PROMPT_POINTS = 32
MAX_CORRECTION_FRAMES = 32
OUTPUT_MODES = ("Matte", "Source + Alpha", "Cutout", "Source (Bypass)")
_range_cancellations: set[str] = set()
_range_cancel_lock = threading.RLock()
_progress_tasks: dict[str, Any] = {}
_progress_lock = threading.RLock()
_postprocess_lock = threading.RLock()
_postprocess_revisions: dict[str, int] = {}
_correction_sync_nodes: set[str] = set()


def _nuke() -> Any:
    import nuke

    return nuke


def _job_error_text(job: dict[str, Any]) -> str:
    """Expose unexpected technical detail instead of hiding it behind a generic worker error."""

    error = job.get("error") or {}
    message = str(error.get("message") or job.get("status") or "Unknown error")
    detail = str(error.get("technical_detail") or "").strip()
    if detail and detail not in message:
        return f"{message} | {detail}"
    return message


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


def _knob_value_at(knob: Any, frame: int, dimensions: int = 1) -> Any:
    """Read an animated knob without changing Nuke's current frame."""

    if dimensions == 1:
        try:
            return knob.valueAt(frame)
        except (AttributeError, TypeError):
            return knob.value()
    values = []
    for index in range(dimensions):
        try:
            values.append(float(knob.valueAt(frame, index)))
        except (AttributeError, TypeError):
            current = knob.value()
            return tuple(float(value) for value in current[:dimensions])
    return tuple(values)


def _collect_points(node: Any, kind: str, frame: int | None = None) -> list[tuple[float, float]]:
    enabled_knob = node[f"{kind}_enabled"]
    enabled = enabled_knob.value() if frame is None else _knob_value_at(enabled_knob, frame)
    if not bool(enabled):
        return []
    return [
        tuple(node[name].value())
        if frame is None
        else _knob_value_at(node[name], frame, dimensions=2)
        for name in _point_knob_names(kind, _point_count(node, kind))
    ]


def _bbox_at(node: Any, frame: int) -> tuple[float, float, float, float]:
    return _knob_value_at(node["prompt_box"], frame, dimensions=4)


def _start_progress(node_name: str, title: str, message: str) -> None:
    task = _nuke().ProgressTask(title)
    task.setMessage(message)
    task.setProgress(0)
    with _progress_lock:
        _progress_tasks[node_name] = task


def _update_progress(node_name: str, percent: int, message: str) -> None:
    with _progress_lock:
        task = _progress_tasks.get(node_name)
    if task is not None:
        task.setMessage(message)
        task.setProgress(max(0, min(100, int(percent))))


def _progress_cancelled(node_name: str) -> bool:
    with _progress_lock:
        task = _progress_tasks.get(node_name)
    return bool(task is not None and task.isCancelled())


def _finish_progress(node_name: str) -> None:
    with _progress_lock:
        _progress_tasks.pop(node_name, None)


def _format_eta(elapsed: float, progress: float) -> str:
    if progress <= 0.01:
        return "calculating..."
    remaining = max(0, round(elapsed * (1.0 - progress) / progress))
    minutes, seconds = divmod(remaining, 60)
    return f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"


def sync_prompt_visibility(node: Any | None = None) -> None:
    """Hide disabled prompt controls and their Viewer handles."""
    nuke = _nuke()
    node = node or nuke.thisNode()
    for kind in ("positive", "negative"):
        enabled = bool(node[f"{kind}_enabled"].value())
        count = _point_count(node, kind)
        for index, name in enumerate(_point_knob_names(kind, MAX_PROMPT_POINTS), start=1):
            if name in node.knobs():
                knob = node[name]
                visible = enabled and index <= count
                knob.setVisible(visible)
                if visible:
                    knob.setFlag(nuke.STARTLINE)
                else:
                    knob.clearFlag(nuke.STARTLINE)
            remove_name = f"remove_{kind}_point_{index}"
            if remove_name in node.knobs():
                remove = node[remove_name]
                if visible and not bool(remove.value()):
                    remove.setValue(True)
                remove.setVisible(visible)
        if f"add_{kind}_point" in node.knobs():
            node[f"add_{kind}_point"].setVisible(enabled)
        if f"remove_{kind}_point" in node.knobs():
            node[f"remove_{kind}_point"].setVisible(False)
    node["prompt_box"].setVisible(bool(node["box_enabled"].value()))
    if "max_hole_area" in node.knobs():
        node["max_hole_area"].setVisible(bool(node["fill_holes"].value()))


def prompt_knob_changed() -> None:
    nuke = _nuke()
    knob = nuke.thisKnob()
    knob_name = knob.name()
    for kind in ("positive", "negative"):
        prefix = f"remove_{kind}_point_"
        if knob_name.startswith(prefix):
            if knob.Class() == "Boolean_Knob" and not bool(knob.value()):
                _defer_ui_action(
                    remove_point,
                    kind,
                    int(knob_name.removeprefix(prefix)),
                    str(nuke.thisNode().fullName()),
                )
            return
    if knob_name.startswith("remove_correction_"):
        if knob.Class() == "Boolean_Knob" and not bool(knob.value()):
            _defer_ui_action(
                remove_correction,
                int(knob_name.removeprefix("remove_correction_")),
                str(nuke.thisNode().fullName()),
            )
        return
    if knob_name.startswith("correction_frame_") and knob_name != "correction_frame_count":
        node = nuke.thisNode()
        node_name = str(node.fullName())
        if node_name not in _correction_sync_nodes:
            _move_correction_frame(
                node,
                int(knob_name.removeprefix("correction_frame_")),
                int(knob.value()),
            )
        return
    if knob_name == "key_frame":
        node = nuke.thisNode()
        node_name = str(node.fullName())
        if node_name not in _correction_sync_nodes:
            _move_primary_key_frame(node, int(knob.value()))
        return
    if knob_name in {
        "positive_enabled",
        "negative_enabled",
        "box_enabled",
        "fill_holes",
    }:
        sync_prompt_visibility(nuke.thisNode())
    node = nuke.thisNode()
    if knob_name in {"fill_holes", "max_hole_area"}:
        request_mask_postprocess(node)
    from kyven_nuke.live import affects_live_result, request_live_update

    if "kyven_live_frame" in node.knobs() and affects_live_result(knob_name, "segment"):
        node["kyven_live_frame"].setValue(-2147483647)
        request_live_update(node)


def _defer_ui_action(function: Any, *args: Any) -> None:
    """Run structural knob updates after Nuke finishes the current knobChanged callback."""

    def dispatch() -> None:
        _nuke().executeInMainThread(function, args=args)

    timer = threading.Timer(0.01, dispatch)
    timer.daemon = True
    timer.start()


def request_mask_postprocess(node: Any, delay_seconds: float = 0.18) -> None:
    """Debounce post-process controls without submitting another SAM job."""

    node_name = str(node.fullName())
    with _postprocess_lock:
        revision = _postprocess_revisions.get(node_name, 0) + 1
        _postprocess_revisions[node_name] = revision
    timer = threading.Timer(
        delay_seconds,
        _dispatch_mask_postprocess,
        args=(node_name, revision),
    )
    timer.daemon = True
    timer.start()


def _dispatch_mask_postprocess(node_name: str, revision: int) -> None:
    _nuke().executeInMainThread(_start_mask_postprocess, args=(node_name, revision))


def _start_mask_postprocess(node_name: str, revision: int) -> None:
    nuke = _nuke()
    with _postprocess_lock:
        if _postprocess_revisions.get(node_name) != revision:
            return
    node = nuke.toNode(node_name)
    if node is None:
        return
    if nuke.executing() or bool(node["kyven_busy"].value()):
        timer = threading.Timer(0.2, _dispatch_mask_postprocess, args=(node_name, revision))
        timer.daemon = True
        timer.start()
        return
    frame = int(nuke.frame())
    _source_path, raw_matte_path, matte_path = _cache_paths(node, frame)
    if not raw_matte_path.is_file():
        tracked_raw = _cache_root(node) / f"raw_tracked_matte.{frame:04d}.png"
        tracked_matte = _cache_root(node) / f"tracked_matte.{frame:04d}.png"
        if tracked_raw.is_file():
            raw_matte_path = tracked_raw
            matte_path = tracked_matte
    if not raw_matte_path.is_file():
        node["kyven_status"].setValue(
            "Mask post-process is ready after SAM has created this frame once."
        )
        return
    preview_path = matte_path.with_name(f"postprocess_preview.{frame:04d}.{revision}.png")
    payload = {
        "source": str(raw_matte_path.resolve()),
        "output": str(preview_path.resolve()),
        "fill_holes": bool(node["fill_holes"].value()),
        "max_hole_area": int(node["max_hole_area"].value()),
    }
    threading.Thread(
        target=_mask_postprocess_worker,
        args=(node_name, revision, preview_path, matte_path, payload),
        name="kyven-mask-postprocess",
        daemon=True,
    ).start()


def _mask_postprocess_worker(
    node_name: str,
    revision: int,
    preview_path: Path,
    matte_path: Path,
    payload: dict[str, Any],
) -> None:
    try:
        result = ensure_server().preview_mask_postprocess(payload)
        _nuke().executeInMainThread(
            _apply_mask_postprocess,
            args=(node_name, revision, preview_path, matte_path, result),
        )
    except Exception as exc:  # noqa: BLE001
        _nuke().executeInMainThread(
            _set_status,
            args=(node_name, f"Mask post-process preview failed: {exc}"),
        )


def _apply_mask_postprocess(
    node_name: str,
    revision: int,
    preview_path: Path,
    matte_path: Path,
    result: dict[str, Any],
) -> None:
    with _postprocess_lock:
        current = _postprocess_revisions.get(node_name)
    if current != revision:
        preview_path.unlink(missing_ok=True)
        return
    node = _nuke().toNode(node_name)
    if node is None or not preview_path.is_file():
        preview_path.unlink(missing_ok=True)
        return
    preview_path.replace(matte_path)
    node.begin()
    try:
        matte = _nuke().toNode("KyvenMatteRead")
        if matte is not None and "reload" in matte.knobs():
            matte["reload"].execute()
    finally:
        node.end()
    filled = int(result.get("filled_holes", 0))
    node["kyven_status"].setValue(
        f"Mask post-process updated instantly | filled {filled} hole(s)"
    )


def _set_busy(node_name: str, busy: bool) -> None:
    node = _nuke().toNode(node_name)
    if node is not None and "kyven_busy" in node.knobs():
        node["kyven_busy"].setValue(busy)


def add_point(kind: str) -> None:
    nuke = _nuke()
    node = nuke.thisNode()
    count = _point_count(node, kind)
    if count >= MAX_PROMPT_POINTS:
        node["kyven_status"].setValue(f"Maximum {MAX_PROMPT_POINTS} {kind} points reached.")
        return
    new_count = count + 1
    knob = node[_point_knob_name(kind, new_count)]
    correction_frames = sorted(
        _correction_frames(node) | {int(node["key_frame"].value())}
    )
    if count:
        previous_knob = node[_point_knob_name(kind, count)]
        previous = previous_knob.value()
    else:
        previous_knob = None
        source = node.input(0)
        previous = (
            [float(source.width()) / 2.0, float(source.height()) / 2.0]
            if source is not None
            else [0.0, 0.0]
        )
    # Re-enabling a previously removed control must not destroy its animation curve.
    if correction_frames:
        for dimension in range(2):
            try:
                knob.clearAnimated(dimension)
            except (AttributeError, RuntimeError, TypeError):
                pass
            knob.setAnimated(dimension)
        for frame in correction_frames:
            if previous_knob is None:
                values = previous
            else:
                values = [
                    float(previous_knob.valueAt(frame, dimension))
                    for dimension in range(2)
                ]
            knob.setValueAt(float(values[0]) + 20.0, frame, 0)
            knob.setValueAt(float(values[1]) + 20.0, frame, 1)
    else:
        offset = 20.0 if count else 0.0
        knob.setValue([float(previous[0]) + offset, float(previous[1]) + offset])
    node[f"{kind}_point_count"].setValue(new_count)
    sync_prompt_visibility(node)
    node["kyven_status"].setValue(f"Added {kind} point {new_count}.")


def remove_point(
    kind: str,
    index: int | None = None,
    node_name: str | None = None,
) -> None:
    nuke = _nuke()
    node = nuke.toNode(node_name) if node_name else nuke.thisNode()
    if node is None:
        return
    count = _point_count(node, kind)
    if count <= 0:
        return
    index = count if index is None else int(index)
    if not 1 <= index <= count:
        return
    for position in range(index, count):
        target = node[_point_knob_name(kind, position)]
        source = node[_point_knob_name(kind, position + 1)]
        try:
            target.fromScript(source.toScript())
        except (AttributeError, TypeError):
            target.setValue(source.value())
    discarded = node[_point_knob_name(kind, count)]
    try:
        discarded.clearAnimated()
    except (AttributeError, TypeError):
        pass
    node[f"{kind}_point_count"].setValue(count - 1)
    sync_prompt_visibility(node)
    node["kyven_status"].setValue(f"Removed {kind} point {index}.")


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
        set_data_io(matte)
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
    if job["status"] == "cancelled":
        node["kyven_status"].setValue("Segmentation cancelled.")
        return
    if job["status"] != "succeeded":
        node["kyven_status"].setValue(f"Failed: {_job_error_text(job)}")
        return
    output = _nuke_file_path(Path(job["result"]["output"]))
    _set_matte_read(node, output)
    metadata = job["result"].get("metadata") or {}
    roi = metadata.get("processing_roi")
    suffix = f" | ROI {roi['width']}x{roi['height']}" if roi else ""
    postprocess = metadata.get("postprocess")
    if postprocess and int(postprocess.get("filled_holes", 0)):
        suffix += f" | filled {int(postprocess['filled_holes'])} hole(s)"
    node["kyven_status"].setValue(
        f"Ready - score {float(job['result']['score']):.3f}{suffix}"
    )


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
        node["kyven_status"].setValue(f"Propagation failed: {_job_error_text(job)}")
        return
    result = job["result"]
    output_pattern = _nuke_file_path(Path(result["output_pattern"]))
    first = int(result["first_frame"])
    last = int(result["last_frame"])
    _set_matte_read(node, output_pattern, first, last)
    roi = (result.get("metadata") or {}).get("processing_roi")
    suffix = f" | ROI {roi['width']}x{roi['height']}" if roi else ""
    postprocess = (result.get("metadata") or {}).get("postprocess")
    if postprocess and int(postprocess.get("filled_holes", 0)):
        suffix += f" | filled {int(postprocess['filled_holes'])} hole(s)"
    node["kyven_status"].setValue(
        f"SAM 2 tracking ready: {first}-{last} ({result['direction']}){suffix}"
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
    finally:
        nuke.executeInMainThread(_set_busy, args=(node_name, False))


def _submit_video_and_wait(node_name: str, payload: dict[str, Any]) -> None:
    nuke = _nuke()
    started = time.monotonic()
    cancellation_sent = False
    try:
        client = ensure_server()
        job_id = client.submit_video(payload)
        nuke.executeInMainThread(_set_job_id, args=(node_name, job_id))
        nuke.executeInMainThread(
            _set_status,
            args=(node_name, "SAM 2 is propagating the key-frame mask..."),
        )
        while True:
            job = client.job(job_id)
            server_progress = float(job.get("progress", 0.0))
            message = str(job.get("progress_message") or "SAM 2 video tracking")
            eta = _format_eta(time.monotonic() - started, server_progress)
            nuke.executeInMainThread(
                _update_progress,
                args=(node_name, 25 + round(server_progress * 74), f"{message} | ETA {eta}"),
            )
            cancelled = nuke.executeInMainThreadWithResult(
                _progress_cancelled,
                args=(node_name,),
            )
            if cancelled and not cancellation_sent:
                client.cancel(job_id)
                cancellation_sent = True
                nuke.executeInMainThread(
                    _set_status,
                    args=(node_name, "Cancelling SAM 2 propagation..."),
                )
            if job["status"] in ("succeeded", "failed", "cancelled"):
                break
            if time.monotonic() - started > 3600.0:
                raise RuntimeError(f"Timed out waiting for Kyven job {job_id}.")
            time.sleep(0.2)
        nuke.executeInMainThread(_apply_video_result, args=(node_name, job))
    except Exception as exc:  # noqa: BLE001 - background boundary must report to the host UI
        nuke.executeInMainThread(_set_status, args=(node_name, f"Propagation failed: {exc}"))
    finally:
        nuke.executeInMainThread(_finish_progress, args=(node_name,))
        nuke.executeInMainThread(_set_busy, args=(node_name, False))


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
    started = time.monotonic()
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
            cancellation_sent = False
            while True:
                job = client.job(job_id)
                job_progress = float(job.get("progress", 0.0))
                overall = (position - 1 + job_progress) / total
                eta = _format_eta(time.monotonic() - started, overall)
                nuke.executeInMainThread(
                    _update_progress,
                    args=(
                        node_name,
                        20 + round(overall * 79),
                        f"Frame {frame} ({position}/{total}) | Segmenting | ETA {eta}",
                    ),
                )
                cancelled = nuke.executeInMainThreadWithResult(
                    _progress_cancelled,
                    args=(node_name,),
                )
                if cancelled and not cancellation_sent:
                    with _range_cancel_lock:
                        _range_cancellations.add(node_name)
                    client.cancel(job_id)
                    cancellation_sent = True
                if job["status"] in ("succeeded", "failed", "cancelled"):
                    break
                time.sleep(0.2)
            if job["status"] == "cancelled":
                nuke.executeInMainThread(
                    _set_status,
                    args=(node_name, f"Range cancelled at frame {frame}."),
                )
                return
            if job["status"] != "succeeded":
                message = _job_error_text(job)
                nuke.executeInMainThread(
                    _set_status,
                    args=(node_name, f"Frame {frame} failed: {message}"),
                )
                return
            scores.append(float(job["result"]["score"]))
            nuke.executeInMainThread(
                _update_progress,
                args=(
                    node_name,
                    20 + round(position / total * 79),
                    f"Completed frame {frame} ({position}/{total})",
                ),
            )
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
        nuke.executeInMainThread(_finish_progress, args=(node_name,))
        nuke.executeInMainThread(_set_busy, args=(node_name, False))


def _set_job_id(node_name: str, job_id: str) -> None:
    node = _nuke().toNode(node_name)
    if node is not None:
        node["kyven_job_id"].setValue(job_id)
        node["kyven_status"].setValue("Segmenting...")


def _cache_root(node: Any) -> Path:
    root = _cache_root_path(node)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cache_root_path(node: Any) -> Path:
    _ensure_unique_cache_uuid(node)
    cache_root = config.cache_dir().resolve()
    node_id = str(node["kyven_uuid"].value()).strip()
    if not node_id:
        raise RuntimeError("Kyven node UUID is empty.")
    target = (cache_root / node_id).resolve()
    if target.parent != cache_root:
        raise RuntimeError("Kyven node cache path is outside the configured cache directory.")
    return target


def _ensure_unique_cache_uuid(node: Any) -> bool:
    """Give a copied Kyven Group its own cache before it reads or writes files."""

    if not hasattr(node, "knobs") or "kyven_uuid" not in node.knobs():
        return False
    node_id = str(node["kyven_uuid"].value()).strip()
    if not node_id:
        node_id = uuid.uuid4().hex
        node["kyven_uuid"].setValue(node_id)
    try:
        node_name = str(node.fullName())
        candidates = _nuke().allNodes("Group", recurseGroups=True)
    except (AttributeError, RuntimeError):
        return False
    duplicate = any(
        str(candidate.fullName()) != node_name
        and "kyven_uuid" in candidate.knobs()
        and str(candidate["kyven_uuid"].value()).strip() == node_id
        for candidate in candidates
    )
    if not duplicate:
        return False

    node["kyven_uuid"].setValue(uuid.uuid4().hex)
    # A pasted Group can also contain Reads pointing at the original node cache.
    # Disconnect them immediately so it cannot display the other node's result.
    _disconnect_cached_matte(node)
    if "cache_location" in node.knobs():
        node["cache_location"].setValue(str(_cache_root_path(node)))
    elif "cache_folder" in node.knobs():
        node["cache_folder"].setValue(str(_cache_root_path(node)))
    if "kyven_status" in node.knobs():
        node["kyven_status"].setValue("Copied node isolated: a new cache was assigned.")
    return True


def create_read_from_current_matte() -> None:
    """Create a regular Read node for the matte currently connected inside the Group."""
    nuke = _nuke()
    node = nuke.thisNode()
    node.begin()
    try:
        matte = nuke.toNode("KyvenMatteRead")
        switch = nuke.toNode("KyvenMatteSwitch")
        if matte is None or switch is None or int(switch["which"].value()) != 1:
            nuke.message("Process a frame or range first; this node has no current matte yet.")
            return
        file_path = str(matte["file"].value())
        frame_values = {
            name: int(matte[name].value())
            for name in ("first", "last", "origfirst", "origlast")
            if name in matte.knobs()
        }
    finally:
        node.end()

    read = nuke.nodes.Read(file=file_path)
    set_data_io(read)
    for name, value in frame_values.items():
        if name in read.knobs():
            read[name].setValue(value)
    read["label"].setValue("Kyven cached matte")
    read.setXpos(node.xpos() + 140)
    read.setYpos(node.ypos() + 120)
    node["kyven_status"].setValue(f"Created Matte Read: {read.name()}")


def delete_node_cache() -> None:
    """Delete only the cache owned by the current Kyven node."""
    nuke = _nuke()
    node = nuke.thisNode()
    target = _cache_root_path(node)
    if not target.exists():
        node["kyven_status"].setValue("This node cache is already empty.")
        return
    if not nuke.ask(f"Delete this Kyven node cache?\n\n{target}"):
        return

    _disconnect_cached_matte(node)
    try:
        shutil.rmtree(target)
    except OSError as exc:
        node["kyven_status"].setValue(f"Cache deletion failed: {exc}")
        nuke.message(f"Could not delete this node cache:\n{exc}")
        return
    node["kyven_status"].setValue("This node cache was deleted.")


def _disconnect_cached_matte(node: Any) -> None:
    """Disconnect every cached Read type used by any Kyven Group."""
    nuke = _nuke()
    node.begin()
    try:
        for switch_name in (
            "KyvenMatteSwitch",
            "KyvenTrimapSwitch",
            "KyvenResultSwitch",
            "KyvenProcessedMaskSwitch",
        ):
            switch = nuke.toNode(switch_name)
            if switch is not None and "which" in switch.knobs():
                switch["which"].setValue(0)
        for read_name in (
            "KyvenMatteRead",
            "KyvenTrimapRead",
            "KyvenResultRead",
            "KyvenProcessedMaskRead",
        ):
            read = nuke.toNode(read_name)
            if read is not None:
                nuke.delete(read)
    finally:
        node.end()


def delete_all_cache() -> None:
    """Delete the complete Nuke cache while preserving models and server files."""
    nuke = _nuke()
    cache_root = config.cache_dir().resolve()
    expected = (config.runtime_dir().resolve() / "nuke_cache").resolve()
    if cache_root != expected or cache_root.name != "nuke_cache":
        nuke.message(f"Refusing to delete an unexpected cache path:\n{cache_root}")
        return
    if not cache_root.exists():
        nuke.message("The Kyven cache is already empty.")
        return
    if not nuke.ask(
        "Delete ALL Kyven Nuke cache?\n\n"
        f"{cache_root}\n\n"
        "All generated source frames and mattes will be removed. Models are kept."
    ):
        return

    affected = 0
    for candidate in nuke.allNodes("Group", recurseGroups=True):
        if "kyven_uuid" not in candidate.knobs():
            continue
        _disconnect_cached_matte(candidate)
        if "kyven_status" in candidate.knobs():
            candidate["kyven_status"].setValue("All Kyven cache was deleted.")
        affected += 1
    try:
        shutil.rmtree(cache_root)
    except OSError as exc:
        nuke.message(f"Could not delete all Kyven cache:\n{exc}")
        return
    nuke.message(f"Kyven cache deleted. Updated {affected} Kyven node(s).")


def _cache_paths(node: Any, frame: int) -> tuple[Path, Path, Path]:
    root = _cache_root(node)
    return (
        root / f"source.{frame:04d}.png",
        root / f"raw_matte.{frame:04d}.png",
        root / f"matte.{frame:04d}.png",
    )


def _cache_patterns(node: Any) -> tuple[Path, Path, Path]:
    root = _cache_root(node)
    return (
        root / "source.%04d.png",
        root / "raw_matte.%04d.png",
        root / "matte.%04d.png",
    )


def _video_cache_paths(
    node: Any,
    first: int,
    last: int,
) -> tuple[Path, Path, Path, Path]:
    root = _cache_root(node)
    frames_dir = root / f"sam2_video_{first}_{last}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("*.jpg"):
        stale.unlink(missing_ok=True)
    source_pattern = frames_dir / "%05d.jpg"
    raw_output_pattern = root / "raw_tracked_matte.%04d.png"
    output_pattern = root / "tracked_matte.%04d.png"
    return frames_dir, source_pattern, raw_output_pattern, output_pattern


def _path_for_frame(pattern: Path, frame: int) -> Path:
    return Path(str(pattern) % frame)


def _has_prompts(node: Any) -> bool:
    return bool(node["positive_enabled"].value() or node["negative_enabled"].value())


def _payload_for_paths(
    node: Any,
    source: Any,
    source_path: Path,
    raw_matte_path: Path,
    matte_path: Path,
) -> dict[str, Any]:
    return segment_payload(
        source=str(source_path.resolve()),
        output=str(matte_path.resolve()),
        raw_output=str(raw_matte_path.resolve()),
        model_index=int(node["model"].getValue()),
        profile=str(node["profile"].value()),
        image_width=int(source.width()),
        image_height=int(source.height()),
        positive_points=_collect_points(node, "positive"),
        negative_points=_collect_points(node, "negative"),
        box_enabled=bool(node["box_enabled"].value()),
        box=tuple(node["prompt_box"].value()),
        fill_holes=bool(node["fill_holes"].value()) if "fill_holes" in node.knobs() else True,
        max_hole_area=(
            int(node["max_hole_area"].value()) if "max_hole_area" in node.knobs() else 2_048
        ),
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


def reset_roi_to_input() -> None:
    """Reset only Processing ROI, preserving every point and its animation."""

    nuke = _nuke()
    node = nuke.thisNode()
    source = node.input(0)
    if source is None:
        nuke.message("Connect the Kyven Segment Source input first.")
        return
    width = float(source.width())
    height = float(source.height())
    node["prompt_box"].setValue([0.0, 0.0, width, height])
    node["kyven_status"].setValue(f"Processing ROI reset to {int(width)}x{int(height)}.")


def process_current_frame(node: Any | None = None, live: bool = False) -> None:
    nuke = _nuke()
    node = node or nuke.thisNode()
    source = node.input(0)
    if source is None:
        if not live:
            nuke.message("Kyven Segment requires a Source input.")
        return
    if not _has_prompts(node):
        if not live:
            nuke.message("Enable at least one positive or negative point.")
        return

    if "kyven_busy" in node.knobs() and bool(node["kyven_busy"].value()):
        return

    frame = int(nuke.frame())
    source_path, raw_matte_path, matte_path = _cache_paths(node, frame)
    payload = _payload_for_paths(node, source, source_path, raw_matte_path, matte_path)

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
    if "kyven_busy" in node.knobs():
        node["kyven_busy"].setValue(True)
    if "kyven_live_frame" in node.knobs():
        node["kyven_live_frame"].setValue(frame)
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
        nuke.message("Enable at least one positive or negative point.")
        return
    if "kyven_busy" in node.knobs() and bool(node["kyven_busy"].value()):
        return
    first = int(node["range_first"].value())
    last = int(node["range_last"].value())
    if last < first:
        nuke.message("Kyven frame range requires Last to be greater than or equal to First.")
        return

    source_pattern, raw_matte_pattern, matte_pattern = _cache_patterns(node)
    writer = _inside(node, "KyvenSourceWrite")
    writer["file"].setValue(_nuke_file_path(source_pattern))
    total = last - first + 1
    node_name = node.fullName()
    _start_progress(node_name, "Kyven Segment Frame Range", "Preparing source export")
    if "kyven_busy" in node.knobs():
        node["kyven_busy"].setValue(True)
    node["kyven_status"].setValue(f"Exporting {total} source frames...")
    try:
        for position, frame in enumerate(range(first, last + 1), start=1):
            if _progress_cancelled(node_name):
                node["kyven_status"].setValue(f"Range cancelled before frame {frame}.")
                _finish_progress(node_name)
                node["kyven_busy"].setValue(False)
                return
            _update_progress(
                node_name,
                round(20 * (position - 1) / total),
                f"Exporting frame {frame} ({position}/{total})",
            )
            nuke.execute(writer, frame, frame)
    except Exception as exc:  # noqa: BLE001 - host boundary must report useful context
        _finish_progress(node_name)
        node["kyven_busy"].setValue(False)
        node["kyven_status"].setValue(f"Range export failed: {exc}")
        nuke.message(f"Kyven range export failed:\n{exc}")
        return

    payloads = []
    for frame in range(first, last + 1):
        source_path = _path_for_frame(source_pattern, frame)
        if not source_path.is_file():
            _finish_progress(node_name)
            node["kyven_busy"].setValue(False)
            node["kyven_status"].setValue(f"Range export failed: frame {frame} was not created.")
            return
        matte_path = _path_for_frame(matte_pattern, frame)
        raw_matte_path = _path_for_frame(raw_matte_pattern, frame)
        payloads.append(
            (
                frame,
                _payload_for_paths(node, source, source_path, raw_matte_path, matte_path),
            )
        )

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
    old_frame = (
        int(node["key_frame_previous"].value())
        if "key_frame_previous" in node.knobs()
        else int(node["key_frame"].value())
    )
    _snapshot_correction_prompts(node, frame)
    if old_frame != frame:
        _remove_point_keyframes_at(node, old_frame)
    node_name = str(node.fullName())
    _correction_sync_nodes.add(node_name)
    try:
        node["key_frame"].setValue(frame)
        if "key_frame_previous" in node.knobs():
            node["key_frame_previous"].setValue(frame)
    finally:
        _correction_sync_nodes.discard(node_name)
    frames = _correction_frames(node)
    frames.discard(old_frame)
    frames.discard(frame)
    _set_correction_frames(node, frames)
    node["kyven_status"].setValue(f"Primary key set to {frame}.")


def _correction_frames(node: Any) -> set[int]:
    if "correction_frame_count" in node.knobs():
        count = int(node["correction_frame_count"].value())
        return {
            int(node[f"correction_frame_{index}"].value())
            for index in range(1, count + 1)
        }
    if "correction_frames" not in node.knobs():
        return set()
    value = str(node["correction_frames"].value()).strip()
    try:
        return {int(item) for item in value.split(",") if item.strip()}
    except ValueError:
        return set()


def _stored_correction_frames(node: Any) -> list[int]:
    value = str(node["correction_frames"].value()).strip()
    try:
        return sorted(int(item) for item in value.split(",") if item.strip())
    except ValueError:
        return []


def _point_keyframe_knobs(node: Any) -> list[Any]:
    return [
        node[name]
        for kind in ("positive", "negative")
        for name in _point_knob_names(kind, MAX_PROMPT_POINTS)
        if name in node.knobs()
    ]


def _remove_point_keyframes_at(node: Any, frame: int) -> None:
    for knob in _point_keyframe_knobs(node):
        for dimension in range(2):
            try:
                knob.removeKeyAt(frame, dimension)
            except (AttributeError, RuntimeError, TypeError):
                try:
                    knob.animation(dimension).removeKey(frame)
                except (AttributeError, RuntimeError, TypeError):
                    continue


def _move_point_keyframes(node: Any, old_frame: int, new_frame: int) -> None:
    if old_frame == new_frame:
        return
    for knob in _point_keyframe_knobs(node):
        for dimension in range(2):
            try:
                value = float(knob.valueAt(old_frame, dimension))
                animated = bool(knob.isAnimated(dimension))
            except (AttributeError, RuntimeError, TypeError):
                continue
            if not animated:
                continue
            try:
                knob.removeKeyAt(old_frame, dimension)
            except (AttributeError, RuntimeError, TypeError):
                try:
                    knob.animation(dimension).removeKey(old_frame)
                except (AttributeError, RuntimeError, TypeError):
                    pass
            knob.setValueAt(value, new_frame, dimension)


def _move_correction_frame(node: Any, index: int, new_frame: int) -> None:
    frames = _stored_correction_frames(node)
    if not 1 <= index <= len(frames):
        return
    old_frame = frames[index - 1]
    if old_frame == new_frame:
        return
    _move_point_keyframes(node, old_frame, new_frame)
    frames[index - 1] = new_frame
    _set_correction_frames(node, set(frames))
    node["kyven_status"].setValue(
        f"Correction and point keyframes moved {old_frame} → {new_frame}."
    )


def _move_primary_key_frame(node: Any, new_frame: int) -> None:
    old_frame = (
        int(node["key_frame_previous"].value())
        if "key_frame_previous" in node.knobs()
        else new_frame
    )
    if old_frame == new_frame:
        return
    _move_point_keyframes(node, old_frame, new_frame)
    frames = _correction_frames(node)
    frames.discard(old_frame)
    frames.discard(new_frame)
    if "key_frame_previous" in node.knobs():
        node["key_frame_previous"].setValue(new_frame)
    _set_correction_frames(node, frames)
    node["kyven_status"].setValue(
        f"Primary key and point keyframes moved {old_frame} → {new_frame}."
    )


def _set_correction_frames(node: Any, frames: set[int]) -> None:
    ordered = sorted(frames)
    node["correction_frames"].setValue(",".join(str(frame) for frame in ordered))
    if "correction_frame_count" in node.knobs():
        count = min(len(ordered), MAX_CORRECTION_FRAMES)
        node["correction_frame_count"].setValue(count)
        node_name = str(node.fullName())
        _correction_sync_nodes.add(node_name)
        try:
            _sync_correction_rows(node, ordered[:count])
        finally:
            _correction_sync_nodes.discard(node_name)
    if "correction_frames_display" in node.knobs():
        label = ", ".join(str(frame) for frame in ordered) if ordered else "None"
        node["correction_frames_display"].setValue(f"Correction frames: <b>{label}</b>")


def _sync_correction_rows(node: Any, frames: list[int]) -> None:
    """Update correction rows without removing knobs while the properties panel is open."""

    nuke = _nuke()
    anchor = "set_key_frame"
    for index, frame in enumerate(frames, start=1):
        row_name = f"correction_frame_{index}"
        remove_name = f"remove_correction_{index}"
        if row_name not in node.knobs():
            row = nuke.Int_Knob(row_name, f"Correction {index}")
            _add_knob(nuke, node, row)
        else:
            row = node[row_name]
        row.setValue(frame)
        row.setVisible(True)
        if remove_name in node.knobs() and node[remove_name].Class() != "Boolean_Knob":
            node.removeKnob(node[remove_name])
        if remove_name not in node.knobs():
            remove = _remove_button(
                nuke,
                remove_name,
                f"kyven_nuke.node.remove_correction({index})",
                "Remove this correction frame",
            )
            _add_knob(nuke, node, remove, start_line=False)
        else:
            remove = node[remove_name]
        remove.setLabel("")
        if not bool(remove.value()):
            remove.setValue(True)
        remove.setTooltip("Remove this correction frame")
        remove.setVisible(True)
        _place_knob_after(node, row_name, anchor)
        _place_knob_after(node, remove_name, row_name)
        anchor = remove_name
    _place_knob_after(node, "add_correction", anchor)
    index = len(frames) + 1
    while f"correction_frame_{index}" in node.knobs():
        node[f"correction_frame_{index}"].setVisible(False)
        remove_name = f"remove_correction_{index}"
        if remove_name in node.knobs():
            node[remove_name].setVisible(False)
        index += 1
    _arrange_tracking_controls(node)


def _arrange_tracking_controls(node: Any) -> None:
    if "add_correction" not in node.knobs():
        return
    anchor = "add_correction"
    for name in ("propagate_backward", "propagate_both", "propagate_forward"):
        if name in node.knobs():
            _place_knob_after(node, name, anchor)
            anchor = name


def _snapshot_correction_prompts(node: Any, frame: int) -> None:
    """Key the current prompt values so each correction retains an independent snapshot."""

    vector_names = [
        *_point_knob_names("positive", _point_count(node, "positive")),
        *_point_knob_names("negative", _point_count(node, "negative")),
    ]
    for name in vector_names:
        knob = node[name]
        values = tuple(knob.value())
        for index, value in enumerate(values):
            try:
                knob.setAnimated(index)
                knob.setValueAt(float(value), frame, index)
            except (AttributeError, TypeError):
                break


def add_current_correction() -> None:
    nuke = _nuke()
    node = nuke.thisNode()
    frame = int(nuke.frame())
    frames = _correction_frames(node)
    if frame == int(node["key_frame"].value()):
        _snapshot_correction_prompts(node, frame)
        node["kyven_status"].setValue(f"Primary key updated at frame {frame}.")
        return
    if frame not in frames and len(frames) >= MAX_CORRECTION_FRAMES:
        node["kyven_status"].setValue(
            f"Maximum {MAX_CORRECTION_FRAMES} correction frames reached."
        )
        return
    _snapshot_correction_prompts(node, frame)
    _set_correction_frames(node, frames | {frame})
    node["kyven_status"].setValue(f"Correction saved at frame {frame}.")


def remove_correction(index: int, node_name: str | None = None) -> None:
    nuke = _nuke()
    node = nuke.toNode(node_name) if node_name else nuke.thisNode()
    if node is None:
        return
    frames = sorted(_correction_frames(node))
    if not 1 <= index <= len(frames):
        return
    removed = frames.pop(index - 1)
    _remove_point_keyframes_at(node, removed)
    _set_correction_frames(node, set(frames))
    node["kyven_status"].setValue(
        f"Correction and point keyframes removed at frame {removed}."
    )


def remove_current_correction() -> None:
    nuke = _nuke()
    node = nuke.thisNode()
    frame = int(nuke.frame())
    frames = _correction_frames(node)
    if frame in frames:
        _remove_point_keyframes_at(node, frame)
        frames.discard(frame)
    _set_correction_frames(node, frames)
    node["kyven_status"].setValue(f"Correction removed at frame {frame}.")


def clear_corrections() -> None:
    node = _nuke().thisNode()
    for frame in _correction_frames(node):
        _remove_point_keyframes_at(node, frame)
    _set_correction_frames(node, set())
    node["kyven_status"].setValue("Correction frames cleared; primary key will still be used.")


def propagate_video(direction: str) -> None:
    nuke = _nuke()
    node = nuke.thisNode()
    source = node.input(0)
    if source is None:
        nuke.message("Kyven Segment requires a Source input.")
        return
    if "kyven_busy" in node.knobs() and bool(node["kyven_busy"].value()):
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

    correction_frames = _correction_frames(node) | {key_frame}
    corrections = [
        {
            "frame": frame,
            "positive_points": _collect_points(node, "positive", frame),
            "negative_points": _collect_points(node, "negative", frame),
        }
        for frame in sorted(correction_frames)
        if first <= frame <= last
    ]
    corrections = [
        correction
        for correction in corrections
        if correction["positive_points"] or correction["negative_points"]
    ]
    if not corrections:
        nuke.message("Enable at least one point on a key/correction frame.")
        return
    primary = next(
        (correction for correction in corrections if correction["frame"] == key_frame),
        corrections[0],
    )
    positive_points = primary["positive_points"]
    negative_points = primary["negative_points"]
    roi_enabled = bool(_knob_value_at(node["box_enabled"], key_frame))
    key_roi = _bbox_at(node, key_frame)
    animated_rois: list[tuple[int, tuple[float, float, float, float]]] = []
    if roi_enabled:
        roi_values = [(frame, _bbox_at(node, frame)) for frame in range(first, last + 1)]
        if any(frame_roi != key_roi for _frame, frame_roi in roi_values):
            animated_rois = roi_values

    frames_dir, source_pattern, raw_output_pattern, output_pattern = _video_cache_paths(
        node, first, last
    )
    writer = _inside(node, "KyvenVideoWrite")
    writer["file"].setValue(_nuke_file_path(source_pattern))
    total = last - first + 1
    node_name = node.fullName()
    _start_progress(node_name, "Kyven SAM 2 Video Tracking", "Preparing frame export")
    if "kyven_busy" in node.knobs():
        node["kyven_busy"].setValue(True)
    node["kyven_status"].setValue(f"Exporting {total} tracking frames...")
    try:
        for position, frame in enumerate(range(first, last + 1), start=1):
            if _progress_cancelled(node_name):
                node["kyven_status"].setValue(f"Tracking cancelled before frame {frame}.")
                _finish_progress(node_name)
                node["kyven_busy"].setValue(False)
                return
            _update_progress(
                node_name,
                round(25 * (position - 1) / total),
                f"Exporting frame {frame} ({position}/{total})",
            )
            nuke.execute(writer, frame, frame)
    except Exception as exc:  # noqa: BLE001 - host boundary must report useful context
        _finish_progress(node_name)
        node["kyven_busy"].setValue(False)
        node["kyven_status"].setValue(f"Tracking export failed: {exc}")
        nuke.message(f"Kyven tracking export failed:\n{exc}")
        return
    exported = list(frames_dir.glob("*.jpg"))
    if len(exported) != total:
        _finish_progress(node_name)
        node["kyven_busy"].setValue(False)
        node["kyven_status"].setValue(
            f"Tracking export failed: expected {total} JPEGs, found {len(exported)}."
        )
        return

    payload = segment_video_payload(
        frames_dir=str(frames_dir.resolve()),
        output_pattern=str(output_pattern.resolve()),
        raw_output_pattern=str(raw_output_pattern.resolve()),
        model_index=int(node["model"].getValue()),
        profile=str(node["profile"].value()),
        image_width=int(source.width()),
        image_height=int(source.height()),
        positive_points=positive_points,
        negative_points=negative_points,
        box_enabled=roi_enabled,
        box=key_roi,
        first_frame=first,
        last_frame=last,
        key_frame=key_frame,
        direction=direction,
        fill_holes=bool(node["fill_holes"].value()) if "fill_holes" in node.knobs() else True,
        max_hole_area=(
            int(node["max_hole_area"].value()) if "max_hole_area" in node.knobs() else 2_048
        ),
        animated_rois=animated_rois,
        corrections=corrections,
    )
    node["kyven_status"].setValue("Starting SAM 2 video predictor...")
    threading.Thread(
        target=_submit_video_and_wait,
        args=(node_name, payload),
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
            models = [model for model in ensure_server().models() if model.get("task") == "segment"]
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


def start_server_for_node() -> None:
    """Start or validate Kyven Server and report through the calling node Status."""

    nuke = _nuke()
    node_name = nuke.thisNode().fullName()
    _set_status(node_name, "Starting Kyven Server...")

    def start() -> None:
        try:
            ensure_server()
            nuke.executeInMainThread(_set_status, args=(node_name, "Kyven Server is ready."))
        except Exception as exc:  # noqa: BLE001
            nuke.executeInMainThread(_set_status, args=(node_name, f"Server start failed: {exc}"))

    threading.Thread(target=start, name="kyven-node-server-start", daemon=True).start()


def stop_server_for_node() -> None:
    """Stop the authenticated Kyven Server after explicit artist confirmation."""

    nuke = _nuke()
    node = nuke.thisNode()
    if not nuke.ask("Stop Kyven Server?\n\nActive processing may be interrupted."):
        return
    node_name = node.fullName()
    _set_status(node_name, "Stopping Kyven Server...")

    def stop() -> None:
        try:
            stopped = stop_server()
            message = "Kyven Server stopped." if stopped else "Kyven Server is already stopped."
            nuke.executeInMainThread(_set_status, args=(node_name, message))
        except Exception as exc:  # noqa: BLE001
            nuke.executeInMainThread(_set_status, args=(node_name, f"Server stop failed: {exc}"))

    threading.Thread(target=stop, name="kyven-node-server-stop", daemon=True).start()


def _add_knob(nuke: Any, node: Any, knob: Any, *, start_line: bool = True) -> Any:
    if start_line:
        knob.setFlag(nuke.STARTLINE)
    else:
        knob.clearFlag(nuke.STARTLINE)
    node.addKnob(knob)
    return knob


def _remove_button(nuke: Any, name: str, command: str, tooltip: str) -> Any:
    """Create a compact native square with Nuke's own checked x glyph."""

    del command  # Removal is dispatched centrally from knobChanged.
    knob = nuke.Boolean_Knob(name, "")
    knob.setValue(True)
    knob.setTooltip(tooltip)
    return knob


def _add_section(nuke: Any, node: Any, name: str, title: str) -> None:
    _add_knob(nuke, node, nuke.Text_Knob(name, "", _section_markup(title)))


def _ensure_double_slider(
    node: Any,
    name: str,
    label: str,
    minimum: float,
    maximum: float,
    default: float | None = None,
) -> Any:
    """Create or upgrade a numeric user knob to a full-width animated slider."""

    nuke = _nuke()
    existing = node[name] if name in node.knobs() else None
    if existing is not None and isinstance(existing, nuke.Double_Knob):
        existing.setLabel(label)
        existing.setRange(minimum, maximum)
        existing.setFlag(nuke.STARTLINE)
        return existing
    value_script = existing.toScript() if existing is not None else ""
    fallback = float(existing.value()) if existing is not None else default
    names = list(node.knobs())
    tail = []
    if existing is not None:
        tail = [node[knob_name] for knob_name in names[names.index(name) :]]
        for knob in tail:
            node.removeKnob(knob)
    slider = nuke.Double_Knob(name, label)
    slider.setRange(minimum, maximum)
    slider.setFlag(nuke.STARTLINE)
    if value_script:
        try:
            slider.fromScript(value_script)
        except Exception:  # noqa: BLE001 - fall back to the current scalar value
            if fallback is not None:
                slider.setValue(fallback)
    elif fallback is not None:
        slider.setValue(fallback)
    node.addKnob(slider)
    for knob in tail[1:]:
        node.addKnob(knob)
    return slider


def _place_knob_after(node: Any, knob_name: str, anchor_name: str) -> None:
    """Move an existing user knob directly after an anchor without recreating controls."""

    names = list(node.knobs())
    if knob_name not in names or anchor_name not in names:
        return
    anchor_index = names.index(anchor_name)
    knob_index = names.index(knob_name)
    if knob_index == anchor_index + 1 or knob_index <= anchor_index:
        return
    tail_names = names[anchor_index + 1 :]
    tail = [node[name] for name in tail_names]
    for knob in tail:
        node.removeKnob(knob)
    ordered = [node_knob for node_knob in tail if node_knob.name() == knob_name]
    ordered.extend(node_knob for node_knob in tail if node_knob.name() != knob_name)
    for knob in ordered:
        node.addKnob(knob)


def _section_markup(title: str) -> str:
    return f'<br><font color="#9fc7e8"><b>{title}</b></font>'


def _restyle_node_ui(node: Any) -> None:
    """Apply current labels and section styling to new or upgraded nodes."""
    nuke = _nuke()
    add_node_branding(node, nuke)
    _place_knob_after(node, "open_model_manager", "refresh_models")
    sections = {
        "model_section": "MODEL AND PERFORMANCE",
        "positive_section": "POSITIVE POINTS / OBJECT TO KEEP",
        "negative_section": "NEGATIVE POINTS / AREAS TO REMOVE",
        "box_section": "PROCESSING ROI / MODEL CROP",
        "postprocess_section": "MASK POST-PROCESS",
        "processing_section": "INDEPENDENT FRAME PROCESSING",
        "tracking_section": "SAM 2 VIDEO TRACKING",
        "output_section": "OUTPUT",
        "cache_section": "CACHE",
        "status_section": "STATUS",
    }
    for name, title in sections.items():
        if name in node.knobs():
            node[name].setValue(_section_markup(title))
    labels = {
        "refresh_models": "Refresh Models",
        "open_model_manager": "Model Manager...",
        "add_positive_point": "+ Positive Point",
        "remove_positive_point": "- Last Positive",
        "add_negative_point": "+ Negative Point",
        "remove_negative_point": "- Last Negative",
        "reset_prompts": "Reset ROI to Input",
        "process_range": "Process Frame Range",
        "cancel": "Cancel",
        "set_key_frame": "Set Current as Key",
        "propagate_forward": "Forward →",
        "propagate_backward": "← Backward",
        "propagate_both": "↔ Both Directions",
        "create_matte_read": "Create Matte Read",
        "delete_node_cache": "Delete Node Cache",
        "delete_all_cache": "Delete All Kyven Cache",
    }
    for name, label in labels.items():
        if name in node.knobs():
            node[name].setLabel(label)
    same_line = {
        "open_model_manager",
        "remove_positive_point",
        "remove_negative_point",
        "range_last",
        "cancel",
        "set_key_frame",
        "propagate_both",
        "propagate_forward",
        "delete_node_cache",
    }
    for name in same_line:
        if name in node.knobs():
            node[name].clearFlag(nuke.STARTLINE)
    if "refresh_models" in node.knobs():
        node["refresh_models"].setFlag(nuke.STARTLINE)
    if "delete_all_cache" in node.knobs():
        node["delete_all_cache"].setFlag(nuke.STARTLINE)
    if "propagate_backward" in node.knobs():
        node["propagate_backward"].setFlag(nuke.STARTLINE)
    _ensure_processing_layout(node)
    _arrange_tracking_controls(node)
    if "kyven_title" in node.knobs():
        node["kyven_title"].setValue(
            '<font size="5" color="#dce9f2"><b>KYVEN / SEGMENT</b></font><br>'
            '<font color="#91a3b0">SAM 2 | Local inference | API 27</font>'
        )
    if "output_help" in node.knobs():
        node["output_help"].setValue(
            "<b>Matte</b>: mask in RGB + alpha &nbsp; | &nbsp; "
            "<b>Source + Alpha</b>: original RGB, mask in alpha<br>"
            "<b>Cutout</b>: premultiplied foreground &nbsp; | &nbsp; "
            "<b>Source</b>: bypass"
        )
    if "roi_help" in node.knobs():
        node["roi_help"].setValue(
            "Crops before SAM; points are translated and the matte returns at full size.<br>"
            "The ROI can be animated without changing the output format."
        )
    if "postprocess_help" in node.knobs():
        node["postprocess_help"].setValue(
            "CPU-only preview from the cached raw SAM mask; controls never rerun SAM.<br>"
            "Fills enclosed holes only; <b>0</b> means all hole sizes."
        )
    if "live_help" in node.knobs():
        node["live_help"].setValue(
            "Live follows timeline and prompt / ROI edits asynchronously.<br>"
            "Disable Live before range or tracking renders."
        )


def _ensure_processing_layout(node: Any) -> None:
    """Group independent current-frame and range actions in reading order."""

    nuke = _nuke()
    if "current_processing_label" not in node.knobs():
        _add_knob(
            nuke,
            node,
            nuke.Text_Knob("current_processing_label", "", "<b>CURRENT FRAME</b>"),
        )
    if "range_processing_label" not in node.knobs():
        _add_knob(
            nuke,
            node,
            nuke.Text_Knob("range_processing_label", "", "<b>FRAME RANGE</b>"),
        )
    anchor = "live_help" if "live_help" in node.knobs() else "live_mode"
    for name in (
        "current_processing_label",
        "process_frame",
        "range_processing_label",
        "range_first",
        "range_last",
        "process_range",
        "cancel",
    ):
        if name in node.knobs() and anchor in node.knobs():
            _place_knob_after(node, name, anchor)
            anchor = name
    if "process_range" in node.knobs():
        node["process_range"].setFlag(nuke.STARTLINE)
    if "cancel" in node.knobs():
        node["cancel"].clearFlag(nuke.STARTLINE)


def _ensure_output_controls(node: Any) -> None:
    """Add the output selector and native Nuke compositing branch to a Segment Group."""
    nuke = _nuke()
    if "output_mode" not in node.knobs():
        _add_section(nuke, node, "output_section", "OUTPUT")
        _add_knob(
            nuke,
            node,
            nuke.Enumeration_Knob("output_mode", "Output", list(OUTPUT_MODES)),
        )
        node["output_mode"].setValue(1)
        help_text = nuke.Text_Knob(
            "output_help",
            "",
            "<b>Matte</b>: mask in RGB + alpha &nbsp; | &nbsp; "
            "<b>Source + Alpha</b>: original RGB, mask in alpha<br>"
            "<b>Cutout</b>: premultiplied foreground &nbsp; | &nbsp; "
            "<b>Source</b>: bypass",
        )
        _add_knob(nuke, node, help_text)

    node.begin()
    try:
        source = nuke.toNode("Source")
        matte = nuke.toNode("KyvenMatteSwitch")
        output = nuke.toNode("Output")
        if source is None or matte is None or output is None:
            raise RuntimeError("Selected node is not a compatible Kyven Segment Group.")

        matte_rgba = nuke.toNode("KyvenMatteRGBA")
        if matte_rgba is None:
            matte_rgba = nuke.nodes.Copy(name="KyvenMatteRGBA")
        matte_rgba.setInput(0, matte)
        matte_rgba.setInput(1, matte)
        matte_rgba["from0"].setValue("rgba.red")
        matte_rgba["to0"].setValue("rgba.alpha")

        source_alpha = nuke.toNode("KyvenSourceAlpha")
        if source_alpha is None:
            source_alpha = nuke.nodes.Copy(name="KyvenSourceAlpha")
        source_alpha.setInput(0, source)
        source_alpha.setInput(1, matte)
        source_alpha["from0"].setValue("rgba.red")
        source_alpha["to0"].setValue("rgba.alpha")

        cutout = nuke.toNode("KyvenCutout")
        if cutout is None:
            cutout = nuke.nodes.Premult(name="KyvenCutout")
        cutout.setInput(0, source_alpha)

        output_switch = nuke.toNode("KyvenOutputSwitch")
        if output_switch is None:
            output_switch = nuke.nodes.Switch(name="KyvenOutputSwitch")
        output_switch.setInput(0, matte_rgba)
        output_switch.setInput(1, source_alpha)
        output_switch.setInput(2, cutout)
        output_switch.setInput(3, source)
        output_switch["which"].setExpression("parent.output_mode")
        output.setInput(0, output_switch)
    finally:
        node.end()


def _ensure_cache_controls(node: Any) -> None:
    """Add cache inspection and maintenance controls to a Segment node."""
    nuke = _nuke()
    if "create_matte_read" in node.knobs():
        return
    _add_section(nuke, node, "cache_section", "CACHE")
    cache_location = nuke.String_Knob("cache_location", "Cache Folder")
    cache_location.setValue(_nuke_file_path(_cache_root_path(node)))
    cache_location.setFlag(nuke.READ_ONLY)
    _add_knob(nuke, node, cache_location)
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "create_matte_read",
            "Create Matte Read",
            "kyven_nuke.node.create_read_from_current_matte()",
        ),
    )
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "delete_node_cache",
            "Delete Node Cache",
            "kyven_nuke.node.delete_node_cache()",
        ),
        start_line=False,
    )
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "delete_all_cache",
            "Delete All Kyven Cache",
            "kyven_nuke.node.delete_all_cache()",
        ),
        start_line=True,
    )


def _ensure_postprocess_controls(node: Any) -> None:
    """Add safe mask cleanup controls to a Segment node."""
    nuke = _nuke()
    if "fill_holes" not in node.knobs():
        _add_section(nuke, node, "postprocess_section", "MASK POST-PROCESS")
        fill_holes = nuke.Boolean_Knob("fill_holes", "Fill Enclosed Holes")
        fill_holes.setValue(True)
        _add_knob(nuke, node, fill_holes)
        _ensure_double_slider(
            node,
            "max_hole_area",
            "Max Hole Area (px)",
            0,
            100_000,
            2_048,
        )
        _add_knob(
            nuke,
            node,
            nuke.Text_Knob(
                "postprocess_help",
                "",
                "Fills only enclosed black islands; the outer silhouette is unchanged. "
                "Set 0 to fill all enclosed holes.",
            ),
        )
    _ensure_double_slider(node, "max_hole_area", "Max Hole Area (px)", 0, 100_000, 2_048)
    if "postprocess_help" in node.knobs():
        node["postprocess_help"].setValue(
            "CPU-only preview from the cached raw SAM mask; controls never rerun SAM.<br>"
            "Fills enclosed holes only; <b>0</b> means all hole sizes."
        )


def _ensure_live_controls(node: Any, kind: str = "segment") -> None:
    """Add Live state while keeping old nodes upgradeable."""
    nuke = _nuke()
    if "live_mode" not in node.knobs():
        live = nuke.Boolean_Knob("live_mode", "Live Current Frame")
        live.setValue(False)
        _add_knob(nuke, node, live)
    for name, value in (
        ("kyven_kind", kind),
        ("kyven_live_frame", "-2147483647"),
        ("kyven_busy", "0"),
    ):
        if name in node.knobs():
            continue
        if name == "kyven_busy":
            knob = nuke.Boolean_Knob(name, name)
            knob.setValue(False)
        elif name == "kyven_live_frame":
            knob = nuke.Int_Knob(name, name)
            knob.setValue(int(value))
        else:
            knob = nuke.String_Knob(name, name)
            knob.setValue(value)
        knob.setVisible(False)
        knob.clearFlag(nuke.STARTLINE)
        node.addKnob(knob)


def _ensure_model_manager_control(node: Any) -> None:
    nuke = _nuke()
    if "open_model_manager" not in node.knobs():
        _add_knob(
            nuke,
            node,
            nuke.PyScript_Knob(
                "open_model_manager",
                "Model Manager...",
                "kyven_nuke.model_manager.show_model_manager()",
            ),
            start_line=False,
        )


def _ensure_server_controls(node: Any) -> None:
    """Add and position shared server lifecycle controls beside Status."""

    nuke = _nuke()
    if "start_server_control" not in node.knobs():
        _add_knob(
            nuke,
            node,
            nuke.PyScript_Knob(
                "start_server_control",
                "Start Server",
                "kyven_nuke.node.start_server_for_node()",
            ),
        )
    if "stop_server_control" not in node.knobs():
        _add_knob(
            nuke,
            node,
            nuke.PyScript_Knob(
                "stop_server_control",
                "Stop Server",
                "kyven_nuke.node.stop_server_for_node()",
            ),
            start_line=False,
        )
    _place_knob_after(node, "start_server_control", "kyven_status")
    _place_knob_after(node, "stop_server_control", "start_server_control")


def _ensure_correction_controls(node: Any) -> None:
    """Add persistent multi-frame correction controls to new or legacy Segment nodes."""

    nuke = _nuke()
    saved_frames = _correction_frames(node)
    saved_frames.discard(int(node["key_frame"].value()))
    if "key_frame_previous" not in node.knobs():
        previous = nuke.Int_Knob("key_frame_previous", "Previous Key Frame")
        previous.setValue(int(node["key_frame"].value()))
        previous.setVisible(False)
        node.addKnob(previous)
    if "correction_frames" not in node.knobs():
        stored = nuke.String_Knob("correction_frames", "Correction Frames")
        stored.setVisible(False)
        node.addKnob(stored)
    if "correction_frame_count" not in node.knobs():
        count = nuke.Int_Knob("correction_frame_count", "Correction Count")
        count.setVisible(False)
        node.addKnob(count)
    if "add_correction" not in node.knobs():
        _add_knob(
            nuke,
            node,
            nuke.PyScript_Knob(
                "add_correction",
                "+ Correction at Current Frame",
                "kyven_nuke.node.add_current_correction()",
            ),
        )
    else:
        node["add_correction"].setLabel("+ Correction at Current Frame")
    for legacy_name in ("correction_frames_display", "remove_correction", "clear_corrections"):
        if legacy_name in node.knobs():
            node[legacy_name].setVisible(False)
    if "set_key_frame" in node.knobs():
        _place_knob_after(node, "add_correction", "set_key_frame")
    _set_correction_frames(node, saved_frames)


def _ensure_point_remove_controls(node: Any) -> None:
    """Add a per-row remove button without recreating animated point knobs."""

    nuke = _nuke()
    for kind in ("positive", "negative"):
        for index, point_name in enumerate(
            _point_knob_names(kind, MAX_PROMPT_POINTS), start=1
        ):
            remove_name = f"remove_{kind}_point_{index}"
            if remove_name in node.knobs() and node[remove_name].Class() != "Boolean_Knob":
                node.removeKnob(node[remove_name])
            if remove_name not in node.knobs():
                _add_knob(
                    nuke,
                    node,
                    _remove_button(
                        nuke,
                        remove_name,
                        f"kyven_nuke.node.remove_point('{kind}', {index})",
                        f"Remove {kind} point {index}",
                    ),
                    start_line=False,
                )
            node[remove_name].setLabel("")
            if not bool(node[remove_name].value()):
                node[remove_name].setValue(True)
            node[remove_name].setTooltip(f"Remove {kind} point {index}")
            _place_knob_after(node, remove_name, point_name)
        legacy = f"remove_{kind}_point"
        if legacy in node.knobs():
            node[legacy].setVisible(False)


def _upgrade_roi_controls(node: Any) -> None:
    """Update legacy prompt-box labels to the Processing ROI terminology."""
    if "box_section" in node.knobs():
        node["box_section"].setValue(_section_markup("PROCESSING ROI / MODEL CROP"))
    if "box_enabled" in node.knobs():
        node["box_enabled"].setLabel("Enable Processing ROI")
    if "prompt_box" in node.knobs():
        node["prompt_box"].setLabel("Processing ROI")
    if "reset_prompts" in node.knobs():
        node["reset_prompts"].setLabel("Reset ROI to Input")
        node["reset_prompts"].setCommand("kyven_nuke.node.reset_roi_to_input()")


def upgrade_selected_segment_node() -> None:
    """Add current UI and output features to an already-created Kyven Segment node."""
    nuke = _nuke()
    try:
        node = nuke.selectedNode()
    except Exception:  # noqa: BLE001 - Nuke raises when nothing is selected
        nuke.message("Select an existing Kyven Segment node first.")
        return
    try:
        _ensure_output_controls(node)
        _ensure_cache_controls(node)
        _ensure_postprocess_controls(node)
        _ensure_live_controls(node)
        _ensure_model_manager_control(node)
        _ensure_server_controls(node)
        _ensure_correction_controls(node)
        _ensure_point_remove_controls(node)
        _upgrade_roi_controls(node)
        _restyle_node_ui(node)
        sync_prompt_visibility(node)
        cached_matte = _inside(node, "KyvenMatteRead")
        if cached_matte is not None:
            set_data_io(cached_matte)
    except Exception as exc:  # noqa: BLE001 - host boundary must report useful context
        nuke.message(f"Could not upgrade the selected node:\n{exc}")
        return
    if "kyven_status" in node.knobs():
        node["kyven_status"].setValue(
            "Kyven node upgraded. Process once to create a raw matte for instant post-process."
        )


def create_segment_node() -> Any:
    nuke = _nuke()
    selected = nuke.selectedNode() if nuke.selectedNodes() else None
    node = nuke.nodes.Group(name="KyvenSegment")
    node_uuid = uuid.uuid4().hex
    node.setInput(0, selected)
    node["label"].setValue("[value kyven_status]")
    node.addKnob(nuke.Tab_Knob("kyven", "Kyven Segment"))
    add_node_branding(node, nuke)
    _add_knob(
        nuke,
        node,
        nuke.Text_Knob(
            "kyven_title",
            "",
            '<font size="5" color="#dce9f2"><b>KYVEN / SEGMENT</b></font><br>'
            '<font color="#91a3b0">SAM 2 | Local inference | API 27</font>',
        ),
    )

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
            "Refresh Models",
            "kyven_nuke.node.refresh_models()",
        ),
        start_line=True,
    )
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "open_model_manager",
            "Model Manager...",
            "kyven_nuke.model_manager.show_model_manager()",
        ),
        start_line=False,
    )

    _add_section(nuke, node, "positive_section", "POSITIVE POINTS (object to keep)")
    _add_knob(nuke, node, nuke.Boolean_Knob("positive_enabled", "Enable Positive Points"))
    node["positive_enabled"].setValue(True)
    positive_count = nuke.Int_Knob("positive_point_count", "Positive Point Count")
    positive_count.setValue(1)
    positive_count.setVisible(False)
    _add_knob(nuke, node, positive_count)
    positive_count.clearFlag(nuke.STARTLINE)
    for index, name in enumerate(_point_knob_names("positive", MAX_PROMPT_POINTS), start=1):
        _add_knob(nuke, node, nuke.XY_Knob(name, f"Positive Point {index}"))
        _add_knob(
            nuke,
            node,
            _remove_button(
                nuke,
                f"remove_positive_point_{index}",
                f"kyven_nuke.node.remove_point('positive', {index})",
                f"Remove positive point {index}",
            ),
            start_line=False,
        )
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "add_positive_point",
            "+ Positive Point",
            "kyven_nuke.node.add_point('positive')",
        ),
    )
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "remove_positive_point",
            "- Last Positive",
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
    negative_count.clearFlag(nuke.STARTLINE)
    for index, name in enumerate(_point_knob_names("negative", MAX_PROMPT_POINTS), start=1):
        _add_knob(nuke, node, nuke.XY_Knob(name, f"Negative Point {index}"))
        _add_knob(
            nuke,
            node,
            _remove_button(
                nuke,
                f"remove_negative_point_{index}",
                f"kyven_nuke.node.remove_point('negative', {index})",
                f"Remove negative point {index}",
            ),
            start_line=False,
        )
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "add_negative_point",
            "+ Negative Point",
            "kyven_nuke.node.add_point('negative')",
        ),
    )
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "remove_negative_point",
            "- Last Negative",
            "kyven_nuke.node.remove_point('negative')",
        ),
        start_line=False,
    )

    _add_section(nuke, node, "box_section", "PROCESSING ROI / MODEL CROP")
    _add_knob(nuke, node, nuke.Boolean_Knob("box_enabled", "Enable Processing ROI"))
    _add_knob(nuke, node, nuke.BBox_Knob("prompt_box", "Processing ROI"))
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "reset_prompts",
            "Reset ROI to Input",
            "kyven_nuke.node.reset_roi_to_input()",
        ),
    )
    _add_knob(
        nuke,
        node,
        nuke.Text_Knob(
            "roi_help",
            "",
            "Crops frames before SAM; points are translated and the matte returns at full size.",
        ),
    )
    _ensure_postprocess_controls(node)

    _add_section(nuke, node, "processing_section", "INDEPENDENT FRAME PROCESSING")
    _ensure_live_controls(node)
    _add_knob(
        nuke,
        node,
        nuke.Text_Knob(
            "live_help",
            "",
            "When enabled, changing the timeline processes the current frame asynchronously.",
        ),
    )
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "process_frame",
            "Process Current Frame",
            "kyven_nuke.node.process_current_frame()",
        ),
    )
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob("cancel", "Cancel", "kyven_nuke.node.cancel_current_job()"),
        start_line=False,
    )
    first = nuke.Int_Knob("range_first", "Range First")
    first.setValue(int(nuke.root().firstFrame()))
    _add_knob(nuke, node, first)
    last = nuke.Int_Knob("range_last", "Range Last")
    last.setValue(int(nuke.root().lastFrame()))
    _add_knob(nuke, node, last, start_line=False)
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "process_range",
            "Process Frame Range",
            "kyven_nuke.node.process_frame_range()",
        ),
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
            "Set Current as Key",
            "kyven_nuke.node.set_key_frame_to_current()",
        ),
        start_line=False,
    )
    _ensure_correction_controls(node)
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "propagate_forward",
            "Forward",
            "kyven_nuke.node.propagate_video('forward')",
        ),
    )
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "propagate_backward",
            "Backward",
            "kyven_nuke.node.propagate_video('backward')",
        ),
        start_line=False,
    )
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "propagate_both",
            "Both Directions",
            "kyven_nuke.node.propagate_video('both')",
        ),
        start_line=False,
    )
    _add_section(nuke, node, "output_section", "OUTPUT")
    _add_knob(
        nuke,
        node,
        nuke.Enumeration_Knob("output_mode", "Output", list(OUTPUT_MODES)),
    )
    node["output_mode"].setValue(1)
    _add_knob(
        nuke,
        node,
        nuke.Text_Knob(
            "output_help",
            "",
            "<b>Matte</b>: mask in RGB + alpha &nbsp; | &nbsp; "
            "<b>Source + Alpha</b>: original RGB, mask in alpha<br>"
            "<b>Cutout</b>: premultiplied foreground &nbsp; | &nbsp; "
            "<b>Source</b>: bypass",
        ),
    )
    internal_id = nuke.String_Knob("kyven_uuid", "UUID")
    internal_id.setValue(node_uuid)
    internal_id.setVisible(False)
    internal_id.clearFlag(nuke.STARTLINE)
    node.addKnob(internal_id)
    _ensure_cache_controls(node)
    _add_section(nuke, node, "status_section", "STATUS")
    status = nuke.String_Knob("kyven_status", "Status")
    status.setFlag(nuke.READ_ONLY)
    status.setValue("Ready")
    _add_knob(nuke, node, status)
    _ensure_server_controls(node)
    job_id = nuke.String_Knob("kyven_job_id", "Job ID")
    job_id.setVisible(False)
    job_id.clearFlag(nuke.STARTLINE)
    node.addKnob(job_id)
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
        nuke.nodes.Output(name="Output")
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
    _ensure_output_controls(node)
    _restyle_node_ui(node)
    _reset_prompts(node)
    sync_prompt_visibility(node)
    return node
