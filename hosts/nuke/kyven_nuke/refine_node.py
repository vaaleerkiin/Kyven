"""Kyven Refine Group node and asynchronous ViTMatte orchestration."""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any

from kyven_nuke.branding import add_node_branding
from kyven_nuke.color_management import set_data_io
from kyven_nuke.input_cache import current_input, prepare_range, use_input
from kyven_nuke.node import (
    _add_knob,
    _add_section,
    _artist_cache_name,
    _begin_node_operation,
    _cache_root,
    _cache_root_path,
    _cache_work_root,
    _clamp_bbox_knob,
    _create_group_at_cursor,
    _defer_ui_action,
    _disable_live_for_manual_process,
    _disconnect_cached_matte,
    _ensure_cache_controls,
    _ensure_double_slider,
    _ensure_live_controls,
    _ensure_server_controls,
    _finish_progress,
    _format_eta,
    _inside,
    _job_error_text,
    _job_timing_text,
    _knob_value_at,
    _missing_sequence_frames,
    _nuke,
    _nuke_file_path,
    _operation_node,
    _path_for_frame,
    _place_knob_after,
    _progress_cancelled,
    _refresh_node_output,
    _section_markup,
    _sequence_frame_range,
    _server_progress,
    _set_busy,
    _set_matte_read,
    _set_operation_controls,
    _set_status,
    _start_progress,
    _update_progress,
    _uses_artist_cache,
)
from kyven_nuke.payload import REFINE_MODEL_IDS, REFINE_MODEL_LABELS, refine_payload, roi_box
from kyven_nuke.preferences import default_model_index
from kyven_nuke.runtime import ensure_server

_range_cancellations: set[str] = set()
_range_cancel_lock = threading.RLock()
_trimap_preview_lock = threading.RLock()
_trimap_preview_revisions: dict[str, int] = {}
_trimap_preview_running: set[str] = set()
_suspended_reads: dict[str, dict[str, str]] = {}
MASK_INPUT = 0
SOURCE_INPUT = 1
REFINE_OUTPUT_MODES = (
    "Refined Matte",
    "Source + Refined Alpha",
    "Refined Cutout",
    "Trimap",
    "Source + Trimap Alpha",
    "Trimap Cutout",
    "Source (Bypass)",
)


def _suspend_result_reads(node: Any) -> None:
    """Release Windows file handles before atomically replacing cached Refine outputs."""

    node_name = str(node.fullName())
    saved: dict[str, str] = {}
    node.begin()
    try:
        nuke = _nuke()
        for name in ("KyvenMatteRead", "KyvenTrimapRead"):
            read = nuke.toNode(name)
            if read is None or "file" not in read.knobs():
                continue
            value = str(read["file"].value())
            if value:
                saved[name] = value
                read["file"].setValue("")
    finally:
        node.end()
    if saved:
        _suspended_reads[node_name] = saved


def _restore_result_reads(node_name: str) -> None:
    saved = _suspended_reads.pop(node_name, None)
    node = _operation_node(node_name)
    if not saved or node is None:
        return
    node.begin()
    try:
        nuke = _nuke()
        for name, value in saved.items():
            read = nuke.toNode(name)
            if read is not None and "file" in read.knobs():
                read["file"].setValue(value)
                if "reload" in read.knobs():
                    read["reload"].execute()
    finally:
        node.end()


def _forget_suspended_reads(node_name: str) -> None:
    _suspended_reads.pop(node_name, None)
REFINE_OUTPUT_HELP = (
    "<b>Refined</b>: Matte / Source + Alpha / Cutout<br>"
    "<b>Trimap</b>: Matte / Source + Alpha / Cutout<br>"
    "<b>Source</b>: bypass &nbsp; | &nbsp; Default: Source + Refined Alpha"
)
REFINE_TRIMAP_HELP = (
    "CPU-only preview; these controls never run ViTMatte.<br>"
    "<b>On</b>: build from mask &nbsp; | &nbsp; <b>Off</b>: use artist trimap"
)
REFINE_LIVE_HELP = (
    "Live follows timeline and model / ROI changes asynchronously.<br>"
    "Trimap controls remain CPU-only."
)
REFINE_ROI_HELP = (
    "<b>Auto</b>: trimap bounds plus context &nbsp; | &nbsp; "
    "<b>Manual</b>: animated ROI &nbsp; | &nbsp; <b>Full</b>: entire frame"
)


def _ensure_input_order(node: Any) -> None:
    """Keep Source on the left connector and Mask/Trimap on the right."""

    old_source = node.input(0)
    old_mask = node.input(1)
    needs_migration = False
    node.begin()
    try:
        source = _nuke().toNode("Source")
        mask = _nuke().toNode("MaskOrTrimap")
        if source is not None and mask is not None:
            needs_migration = int(source["number"].value()) == 0
            source["number"].setValue(SOURCE_INPUT)
            mask["number"].setValue(MASK_INPUT)
    finally:
        node.end()
    if needs_migration:
        node.setInput(MASK_INPUT, old_mask)
        node.setInput(SOURCE_INPUT, old_source)


def _cache_paths(node: Any, frame: int) -> tuple[Path, Path, Path, Path]:
    root = _cache_work_root(node) if _uses_artist_cache(node) else _cache_root(node)
    source_root = root / "source" if _uses_artist_cache(node) else root
    source_root.mkdir(parents=True, exist_ok=True)
    output = (
        _path_for_frame(_refine_public_pattern(node), frame)
        if _uses_artist_cache(node)
        else root / f"refined_matte.{frame:04d}.png"
    )
    return (
        source_root / f"refine_source.{frame:04d}.tif",
        root / f"refine_mask.{frame:04d}.tif",
        output,
        root / f"trimap.{frame:04d}.png",
    )


def _cache_patterns(node: Any) -> tuple[Path, Path, Path, Path]:
    root = _cache_work_root(node) if _uses_artist_cache(node) else _cache_root(node)
    source_root = root / "source" if _uses_artist_cache(node) else root
    source_root.mkdir(parents=True, exist_ok=True)
    return (
        source_root / "refine_source.%04d.tif",
        root / "refine_mask.%04d.tif",
        (
            _refine_public_pattern(node)
            if _uses_artist_cache(node)
            else root / "refined_matte.%04d.png"
        ),
        root / "trimap.%04d.png",
    )


def _refine_public_pattern(node: Any) -> Path:
    root = _cache_root_path(node)
    name = _artist_cache_name(node)
    version = str(node["cache_version"].value()).strip()
    pattern = root / f"{name}_{version}.%04d.png"
    if root.is_dir():
        for legacy in root.glob("refined_matte.*.png"):
            token = legacy.name.removeprefix("refined_matte.")
            try:
                legacy.replace(root / f"{name}_{version}.{token}")
            except OSError:
                pass
    return pattern


def _load_refine_cache(node: Any, *, report_status: bool = True) -> bool:
    try:
        pattern = _refine_public_pattern(node)
    except RuntimeError as exc:
        if report_status:
            node["kyven_status"].setValue(str(exc))
        return False
    frame_range = _sequence_frame_range(pattern)
    if frame_range is None:
        if report_status:
            node["kyven_status"].setValue(
                f"No cached sequence at {_nuke_file_path(pattern)}"
            )
        return False
    first, last = frame_range
    _set_matte_read(node, _nuke_file_path(pattern), first, last)
    trimap_pattern = _cache_work_root(node) / "trimap.%04d.png"
    if _path_for_frame(trimap_pattern, first).is_file():
        _set_trimap_read(node, _nuke_file_path(trimap_pattern), first, last)
    if report_status:
        node["kyven_status"].setValue(
            f"Loaded {pattern.name}: frames {first}-{last}"
        )
        _refresh_node_output(node)
    return True


def _apply_refine_cache_fields(node_name: str) -> None:
    node = _nuke().toNode(node_name)
    if node is None:
        return
    if "kyven_busy" in node.knobs() and bool(node["kyven_busy"].value()):
        node["kyven_status"].setValue("Cancel the active operation before changing cache.")
        return
    if "previous_cache_location" in node.knobs():
        node.removeKnob(node["previous_cache_location"])
    _disconnect_cached_matte(node)
    if "kyven_job_id" in node.knobs():
        node["kyven_job_id"].setValue("")
    if "kyven_live_frame" in node.knobs():
        node["kyven_live_frame"].setValue(-2147483647)
    _load_refine_cache(node)


def _trimap_preview_paths(root: Path, frame: int, revision: int) -> tuple[Path, Path]:
    return (
        root / f"trimap_preview_input.{frame:04d}.{revision}.tif",
        root / f"trimap_preview_r{revision}.{frame:04d}.png",
    )


def _tile_size(node: Any) -> int:
    profile = str(node["profile"].value())
    automatic = {"low_memory": 512, "balanced": 1024, "quality": 0}[profile]
    custom = int(node["tile_size"].value())
    return custom if custom else automatic


def _configure_export_writers(source_writer: Any | None, mask_writer: Any) -> None:
    """Use fast lossless formats for temporary inputs consumed only by Kyven."""

    if source_writer is not None:
        source_writer["file_type"].setValue("tiff")
        if "compression" in source_writer.knobs():
            source_writer["compression"].setValue(0)
    mask_writer["file_type"].setValue("tiff")
    set_data_io(mask_writer)
    if "compression" in mask_writer.knobs():
        mask_writer["compression"].setValue(0)


def _payload(
    node: Any,
    source: Any,
    source_path: Path,
    mask_path: Path,
    output_path: Path,
    trimap_output_path: Path,
    frame: int,
) -> dict[str, Any]:
    knob_names = node.knobs() if hasattr(node, "knobs") else node.values
    legacy_roi_enabled = bool(_knob_value_at(node["roi_enabled"], frame))
    crop_mode = (
        str(node["crop_mode"].value())
        if "crop_mode" in knob_names
        else ("manual" if legacy_roi_enabled else "full")
    )
    return refine_payload(
        source=str(source_path.resolve()),
        mask=str(mask_path.resolve()),
        output=str(output_path.resolve()),
        trimap_output=str(trimap_output_path.resolve()),
        model_index=int(node["model"].getValue()),
        profile=str(node["profile"].value()),
        image_width=int(source.width()),
        image_height=int(source.height()),
        roi_enabled=crop_mode == "manual",
        roi=tuple(_knob_value_at(node["processing_roi"], frame, dimensions=4)),
        crop_mode=crop_mode,
        context_padding=(
            int(_knob_value_at(node["context_padding"], frame))
            if "context_padding" in knob_names
            else 128
        ),
        generate_trimap=bool(node["generate_trimap"].value()),
        foreground_radius=int(node["foreground_radius"].value()),
        background_radius=int(node["background_radius"].value()),
        tile_size=_tile_size(node),
        tile_overlap=int(node["tile_overlap"].value()),
    )


def _apply_result(node_name: str, job: dict[str, Any]) -> None:
    node = _operation_node(node_name)
    if node is None:
        return
    if job["status"] == "cancelled":
        _restore_result_reads(node_name)
        node["kyven_status"].setValue("Refinement cancelled.")
        return
    if job["status"] != "succeeded":
        _restore_result_reads(node_name)
        node["kyven_status"].setValue(f"Refine failed: {_job_error_text(job)}")
        return
    result = job["result"]
    if not (_uses_artist_cache(node) and _load_refine_cache(node, report_status=False)):
        _set_matte_read(node, _nuke_file_path(Path(result["output"])))
    trimap_output = result.get("trimap_output")
    if trimap_output:
        _set_trimap_read(node, _nuke_file_path(Path(trimap_output)))
    _forget_suspended_reads(node_name)
    metadata = result.get("metadata") or {}
    trimap = metadata.get("trimap") or {}
    mode = "auto trimap" if trimap.get("generated") else "input trimap"
    roi = metadata.get("processing_roi")
    roi_text = f" | ROI {roi['width']}x{roi['height']}" if roi else ""
    cache_text = " | cache hit" if metadata.get("cache_hit") else ""
    timing_text = _job_timing_text(job)
    node["kyven_status"].setValue(
        f"Refined - {mode} | {int(metadata.get('tiles', 1))} tile(s)"
        f"{roi_text}{cache_text}{timing_text}"
    )
    _refresh_node_output(node)


def _set_trimap_read(
    node: Any,
    output: str,
    first: int | None = None,
    last: int | None = None,
) -> None:
    nuke = _nuke()
    node.begin()
    try:
        trimap = nuke.toNode("KyvenTrimapRead")
        if trimap is None:
            trimap = nuke.nodes.Read(name="KyvenTrimapRead", file=output)
            nuke.toNode("KyvenTrimapSwitch").setInput(1, trimap)
        else:
            trimap["file"].setValue(output)
        set_data_io(trimap)
        if first is not None and last is not None:
            for knob_name, value in (
                ("first", first),
                ("last", last),
                ("origfirst", first),
                ("origlast", last),
            ):
                if knob_name in trimap.knobs():
                    trimap[knob_name].setValue(value)
        if "reload" in trimap.knobs():
            trimap["reload"].execute()
        nuke.toNode("KyvenTrimapSwitch")["which"].setValue(1)
    finally:
        node.end()


def request_trimap_preview(node: Any, delay_seconds: float = 0.18) -> None:
    """Show the native Nuke trimap graph immediately without server work."""

    del delay_seconds
    _activate_native_trimap_preview(node)


def _activate_native_trimap_preview(node: Any) -> None:
    """Switch Refine output to its dependency-free 0/0.5/1 trimap branch."""

    trimap_switch = _inside(node, "KyvenTrimapSwitch")
    if trimap_switch is None:
        _ensure_refine_output_controls(node)
        trimap_switch = _inside(node, "KyvenTrimapSwitch")
    if trimap_switch is not None:
        trimap_switch["which"].setValue(0)
    if "kyven_trimap_preview_frame" in node.knobs():
        node["kyven_trimap_preview_frame"].setValue(int(_nuke().frame()))
    _refresh_node_output(node)
    if "kyven_status" in node.knobs():
        node["kyven_status"].setValue("Trimap preview ready - native Nuke graph.")


def _request_server_trimap_preview(node: Any, delay_seconds: float = 0.18) -> None:
    """Legacy server preview retained for compatibility and diagnostics."""

    node_name = str(node.fullName())
    with _trimap_preview_lock:
        revision = _trimap_preview_revisions.get(node_name, 0) + 1
        _trimap_preview_revisions[node_name] = revision
    timer = threading.Timer(
        delay_seconds,
        _dispatch_trimap_preview,
        args=(node_name, revision),
    )
    timer.daemon = True
    timer.start()


def _dispatch_trimap_preview(node_name: str, revision: int) -> None:
    _nuke().executeInMainThread(_start_trimap_preview, args=(node_name, revision))


def _start_trimap_preview(node_name: str, revision: int) -> None:
    nuke = _nuke()
    with _trimap_preview_lock:
        if _trimap_preview_revisions.get(node_name) != revision:
            return
        if node_name in _trimap_preview_running:
            return
    node = nuke.toNode(node_name)
    if node is not None:
        _ensure_input_order(node)
    if node is None or node.input(MASK_INPUT) is None:
        return
    if nuke.executing() or bool(node["kyven_busy"].value()):
        timer = threading.Timer(0.2, _dispatch_trimap_preview, args=(node_name, revision))
        timer.daemon = True
        timer.start()
        return
    frame = int(nuke.frame())
    root = _cache_work_root(node) if _uses_artist_cache(node) else _cache_root(node)
    mask_path, output_path = _trimap_preview_paths(root, frame, revision)
    writer = _inside(node, "KyvenRefineMaskWrite")
    _configure_export_writers(None, writer)
    writer["file"].setValue(_nuke_file_path(mask_path))
    try:
        nuke.execute(writer, frame, frame)
    except Exception as exc:  # noqa: BLE001
        node["kyven_status"].setValue(f"Trimap preview export failed: {exc}")
        return
    if not mask_path.is_file():
        node["kyven_status"].setValue("Trimap preview export did not create a PNG.")
        return
    image_height = int(node.input(SOURCE_INPUT).height()) if node.input(SOURCE_INPUT) is not None else int(node.input(MASK_INPUT).height())
    payload = {
        "mask": str(mask_path.resolve()),
        "output": str(output_path.resolve()),
        "generate_trimap": bool(node["generate_trimap"].value()),
        "foreground_radius": int(node["foreground_radius"].value()),
        "background_radius": int(node["background_radius"].value()),
        "roi": (
            roi_box(
                tuple(node["processing_roi"].value()),
                image_height,
                (
                    int(node.input(SOURCE_INPUT).width())
                    if node.input(SOURCE_INPUT) is not None
                    else int(node.input(MASK_INPUT).width())
                ),
            )
            if bool(node["roi_enabled"].value())
            else None
        ),
    }
    node["kyven_status"].setValue("Updating trimap preview (CPU only)...")
    with _trimap_preview_lock:
        _trimap_preview_running.add(node_name)
    threading.Thread(
        target=_trimap_preview_worker,
        args=(node_name, revision, frame, mask_path, output_path, payload),
        name="kyven-trimap-preview",
        daemon=True,
    ).start()


def _trimap_preview_worker(
    node_name: str,
    revision: int,
    frame: int,
    mask_path: Path,
    output_path: Path,
    payload: dict[str, Any],
) -> None:
    error = ""
    try:
        ensure_server().preview_trimap(payload)
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    _nuke().executeInMainThread(
        _finish_trimap_preview,
        args=(node_name, revision, frame, mask_path, output_path, error),
    )


def _finish_trimap_preview(
    node_name: str,
    revision: int,
    frame: int,
    mask_path: Path,
    output_path: Path,
    error: str,
) -> None:
    mask_path.unlink(missing_ok=True)
    with _trimap_preview_lock:
        _trimap_preview_running.discard(node_name)
        current = _trimap_preview_revisions.get(node_name)
    if current != revision:
        output_path.unlink(missing_ok=True)
        if current is not None:
            _start_trimap_preview(node_name, current)
        return
    node = _operation_node(node_name)
    if node is None:
        output_path.unlink(missing_ok=True)
        return
    if error:
        output_path.unlink(missing_ok=True)
        node["kyven_status"].setValue(f"Trimap preview failed: {error}")
        return
    if not output_path.is_file():
        node["kyven_status"].setValue("Trimap preview failed: server did not create the PNG.")
        return
    _set_trimap_read(node, _nuke_file_path(output_path), frame, frame)
    if "kyven_trimap_preview_frame" in node.knobs():
        node["kyven_trimap_preview_frame"].setValue(frame)
    _refresh_node_output(node)
    node["kyven_status"].setValue("Trimap preview ready - ViTMatte was not run.")


def _submit_and_wait(
    node_name: str,
    payload: dict[str, Any],
    show_progress: bool,
) -> None:
    nuke = _nuke()
    started = time.monotonic()
    cancellation_sent = False
    try:
        client = ensure_server(progress_callback=_server_progress(node_name))
        job_id = client.submit_refine(payload)
        nuke.executeInMainThread(_set_job_id, args=(node_name, job_id))
        while True:
            job = client.job(job_id)
            progress = float(job.get("progress", 0.0))
            message = str(job.get("progress_message") or "ViTMatte refining")
            if show_progress:
                eta = _format_eta(time.monotonic() - started, progress)
                nuke.executeInMainThread(
                    _update_progress,
                    args=(node_name, 15 + round(progress * 84), f"{message} | ETA {eta}"),
                )
                cancelled = nuke.executeInMainThreadWithResult(
                    _progress_cancelled,
                    args=(node_name,),
                )
                if cancelled and not cancellation_sent:
                    client.cancel(job_id)
                    cancellation_sent = True
            if job["status"] in ("succeeded", "failed", "cancelled"):
                break
            if time.monotonic() - started > 1800.0:
                raise RuntimeError(f"Timed out waiting for Kyven job {job_id}.")
            time.sleep(0.2)
        nuke.executeInMainThread(_apply_result, args=(node_name, job))
    except Exception as exc:  # noqa: BLE001
        nuke.executeInMainThread(_restore_result_reads, args=(node_name,))
        nuke.executeInMainThread(_set_status, args=(node_name, f"Refine failed: {exc}"))
    finally:
        if show_progress:
            nuke.executeInMainThread(_finish_progress, args=(node_name,))
        nuke.executeInMainThread(_set_busy, args=(node_name, False))


def _set_job_id(node_name: str, job_id: str) -> None:
    node = _operation_node(node_name)
    if node is not None:
        node["kyven_job_id"].setValue(job_id)
        node["kyven_status"].setValue("ViTMatte refining...")


def _export_cancelled(node_name: str) -> bool:
    with _range_cancel_lock:
        requested = node_name in _range_cancellations
    return requested or _progress_cancelled(node_name)


def _export(node: Any, frame: int, source_path: Path, mask_path: Path) -> bool:
    nuke = _nuke()
    node_name = str(node.fullName())
    source_writer = _inside(node, "KyvenRefineSourceWrite")
    mask_writer = _inside(node, "KyvenRefineMaskWrite")
    _configure_export_writers(source_writer, mask_writer)
    source_writer["file"].setValue(_nuke_file_path(source_path))
    mask_writer["file"].setValue(_nuke_file_path(mask_path))
    try:
        current_input(nuke, node, source_writer, frame)
        if _export_cancelled(node_name):
            node["kyven_status"].setValue("Refine export cancelled after Source.")
            return False
        current_input(nuke, node, mask_writer, frame)
        if _export_cancelled(node_name):
            node["kyven_status"].setValue("Refine export cancelled after Mask.")
            return False
    except Exception as exc:  # noqa: BLE001
        node["kyven_status"].setValue(f"Refine input export failed: {exc}")
        return False
    return source_path.is_file() and mask_path.is_file()


def process_current_frame(node: Any | None = None, live: bool = False) -> None:
    nuke = _nuke()
    node = node or nuke.thisNode()
    _ensure_input_order(node)
    source = node.input(SOURCE_INPUT)
    mask = node.input(MASK_INPUT)
    if source is None or mask is None:
        if not live:
            nuke.message("Kyven Refine requires Source and Mask/Trimap inputs.")
        return
    frame = int(nuke.frame())
    node_name = node.fullName()
    show_progress = not live
    if not _begin_node_operation(node):
        return
    if not live:
        _disable_live_for_manual_process(node)
    with _range_cancel_lock:
        _range_cancellations.discard(node_name)
    if show_progress:
        _start_progress(node_name, "Kyven ViTMatte Refine", f"Exporting frame {frame}")
        _update_progress(node_name, 2, "Exporting Source and mask")
    source_path, mask_path, output_path, trimap_output_path = _cache_paths(node, frame)
    node["kyven_status"].setValue("Exporting source and mask...")
    if not _export(node, frame, source_path, mask_path):
        if show_progress:
            _finish_progress(node_name)
        _set_busy(node_name, False)
        return
    if show_progress:
        _update_progress(node_name, 15, "Starting ViTMatte")
    payload = _payload(
        node,
        source,
        source_path,
        mask_path,
        output_path,
        trimap_output_path,
        frame,
    )
    node["kyven_live_frame"].setValue(frame)
    threading.Thread(
        target=_submit_and_wait,
        args=(node_name, payload, show_progress),
        name="kyven-refine-submit",
        daemon=True,
    ).start()


def _apply_range(
    node_name: str,
    output: str,
    trimap_output: str,
    first: int,
    last: int,
) -> None:
    node = _operation_node(node_name)
    if node is not None:
        _set_matte_read(node, output, first, last)
        _set_trimap_read(node, trimap_output, first, last)
        _forget_suspended_reads(node_name)
        node["kyven_status"].setValue(f"Refined range ready: {first}-{last}")
        _refresh_node_output(node)


def _range_worker(
    node_name: str,
    payloads: list[tuple[int, dict[str, Any]]],
    output_pattern: str,
    trimap_output_pattern: str,
    first: int,
    last: int,
) -> None:
    nuke = _nuke()
    started = time.monotonic()
    completed = False
    try:
        client = ensure_server(progress_callback=_server_progress(node_name))
        total = len(payloads)
        for index, (frame, payload) in enumerate(payloads, start=1):
            with _range_cancel_lock:
                if node_name in _range_cancellations:
                    nuke.executeInMainThread(
                        _set_status,
                        args=(node_name, f"Refine range cancelled before frame {frame}."),
                    )
                    return
            job_id = client.submit_refine(payload)
            nuke.executeInMainThread(_set_job_id, args=(node_name, job_id))
            nuke.executeInMainThread(
                _set_status,
                args=(node_name, f"Refining frame {frame} ({index}/{len(payloads)})..."),
            )
            cancellation_sent = False
            job_started = time.monotonic()
            while True:
                job = client.job(job_id)
                job_progress = float(job.get("progress", 0.0))
                overall = (index - 1 + job_progress) / total
                message = str(job.get("progress_message") or f"Refining frame {frame}")
                eta = _format_eta(time.monotonic() - started, overall)
                nuke.executeInMainThread(
                    _update_progress,
                    args=(
                        node_name,
                        20 + round(overall * 79),
                        f"Frame {frame} ({index}/{total}) | {message} | ETA {eta}",
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
                if time.monotonic() - job_started > 1800.0:
                    raise RuntimeError(f"Timed out waiting to refine frame {frame}.")
                time.sleep(0.2)
            if job["status"] == "cancelled":
                nuke.executeInMainThread(
                    _set_status,
                    args=(node_name, f"Refine range cancelled at frame {frame}."),
                )
                return
            if job["status"] != "succeeded":
                error = job.get("error") or {}
                raise RuntimeError(error.get("message", job["status"]))
        nuke.executeInMainThread(
            _apply_range,
            args=(node_name, output_pattern, trimap_output_pattern, first, last),
        )
        completed = True
    except Exception as exc:  # noqa: BLE001
        nuke.executeInMainThread(_set_status, args=(node_name, f"Refine range failed: {exc}"))
    finally:
        if not completed:
            nuke.executeInMainThread(_restore_result_reads, args=(node_name,))
        with _range_cancel_lock:
            _range_cancellations.discard(node_name)
        nuke.executeInMainThread(_finish_progress, args=(node_name,))
        nuke.executeInMainThread(_set_busy, args=(node_name, False))


def process_frame_range() -> None:
    nuke = _nuke()
    node = nuke.thisNode()
    _ensure_input_order(node)
    source = node.input(SOURCE_INPUT)
    if source is None or node.input(MASK_INPUT) is None:
        nuke.message("Kyven Refine requires Source and Mask/Trimap inputs.")
        return
    first = int(node["range_first"].value())
    last = int(node["range_last"].value())
    if last < first:
        nuke.message("Range Last must be greater than or equal to Range First.")
        return
    source_pattern, mask_pattern, output_pattern, trimap_output_pattern = _cache_patterns(node)
    source_writer = _inside(node, "KyvenRefineSourceWrite")
    mask_writer = _inside(node, "KyvenRefineMaskWrite")
    _configure_export_writers(source_writer, mask_writer)
    source_writer["file"].setValue(_nuke_file_path(source_pattern))
    mask_writer["file"].setValue(_nuke_file_path(mask_pattern))
    if not _begin_node_operation(node):
        return
    node_name = node.fullName()
    _disable_live_for_manual_process(node)
    with _range_cancel_lock:
        _range_cancellations.discard(node_name)
    _start_progress(node_name, "Kyven ViTMatte Frame Range", "Preparing input export")
    node["kyven_status"].setValue(f"Exporting refine inputs {first}-{last}...")
    try:
        if not prepare_range(nuke, node, first, last, cancelled=lambda: _export_cancelled(node_name)):
            _finish_progress(node_name)
            _set_busy(node_name, False)
            return


        total = last - first + 1
        for index, frame in enumerate(range(first, last + 1), start=1):
            if _export_cancelled(node_name):
                node["kyven_status"].setValue(f"Refine range cancelled before frame {frame}.")
                _finish_progress(node_name)
                _set_busy(node_name, False)
                return
            percent = 2 + round((index - 1) / total * 18)
            _update_progress(node_name, percent, f"Exporting Source {frame} ({index}/{total})")
            use_input(nuke, node, source_writer, frame)
            if _export_cancelled(node_name):
                node["kyven_status"].setValue(f"Refine range cancelled after Source {frame}.")
                _finish_progress(node_name)
                _set_busy(node_name, False)
                return
            _update_progress(node_name, percent + 1, f"Exporting Mask {frame} ({index}/{total})")
            use_input(nuke, node, mask_writer, frame)
        _update_progress(node_name, 20, "Refine inputs exported")
    except Exception as exc:  # noqa: BLE001
        _finish_progress(node_name)
        _set_busy(node_name, False)
        node["kyven_status"].setValue(f"Refine range export failed: {exc}")
        return
    missing_source = _missing_sequence_frames(source_pattern, first, last)
    missing_mask = _missing_sequence_frames(mask_pattern, first, last)
    if missing_source or missing_mask:
        _finish_progress(node_name)
        _set_busy(node_name, False)
        kind, missing = ("Source", missing_source) if missing_source else ("Mask", missing_mask)
        node["kyven_status"].setValue(f"Missing exported refine {kind} at frame {missing[0]}.")
        return

    payloads = []
    for frame in range(first, last + 1):
        source_path = _path_for_frame(source_pattern, frame)
        mask_path = _path_for_frame(mask_pattern, frame)
        output_path = _path_for_frame(output_pattern, frame)
        trimap_output_path = _path_for_frame(trimap_output_pattern, frame)
        payloads.append(
            (
                frame,
                _payload(
                    node,
                    source,
                    source_path,
                    mask_path,
                    output_path,
                    trimap_output_path,
                    frame,
                ),
            )
        )
    threading.Thread(
        target=_range_worker,
        args=(
            node_name,
            payloads,
            _nuke_file_path(output_pattern),
            _nuke_file_path(trimap_output_pattern),
            first,
            last,
        ),
        name="kyven-refine-range",
        daemon=True,
    ).start()


def cancel_current_job() -> None:
    nuke = _nuke()
    node = nuke.thisNode()
    node_name = node.fullName()
    with _range_cancel_lock:
        _range_cancellations.add(node_name)
    try:
        nuke.cancel()
    except Exception:  # noqa: BLE001, S110 - there may be no active Nuke render
        pass
    job_id = str(node["kyven_job_id"].value())
    if not job_id:
        node["kyven_status"].setValue("Cancellation requested.")
        return

    def cancel() -> None:
        try:
            ensure_server().cancel(job_id)
            nuke.executeInMainThread(_set_status, args=(node_name, "Cancellation requested."))
        except Exception as exc:  # noqa: BLE001
            nuke.executeInMainThread(_set_status, args=(node_name, f"Cancel failed: {exc}"))

    threading.Thread(target=cancel, name="kyven-refine-cancel", daemon=True).start()


def reset_roi_to_input() -> None:
    nuke = _nuke()
    node = nuke.thisNode()
    _ensure_input_order(node)
    source = node.input(SOURCE_INPUT)
    if source is None:
        nuke.message("Connect a Source first.")
        return
    node["processing_roi"].setValue([0.0, 0.0, float(source.width()), float(source.height())])
    node["kyven_status"].setValue("Processing ROI reset to Source size.")


def knob_changed() -> None:
    nuke = _nuke()
    node = nuke.thisNode()
    name = nuke.thisKnob().name()
    if name in {"cache_root", "refine_name", "cache_version"}:
        _defer_ui_action(_apply_refine_cache_fields, str(node.fullName()))
        return
    if name in {"inputChange", "processing_roi", "crop_mode", "roi_enabled"}:
        _clamp_bbox_knob(node, "processing_roi", SOURCE_INPUT)
    generated = bool(node["generate_trimap"].value())
    node["foreground_radius"].setVisible(generated)
    node["background_radius"].setVisible(generated)
    crop_mode = (
        str(node["crop_mode"].value())
        if "crop_mode" in node.knobs()
        else ("manual" if bool(node["roi_enabled"].value()) else "full")
    )
    node["processing_roi"].setVisible(crop_mode == "manual")
    if "reset_roi" in node.knobs():
        node["reset_roi"].setVisible(crop_mode == "manual")
    if "context_padding" in node.knobs():
        node["context_padding"].setVisible(crop_mode == "auto")
    if "roi_enabled" in node.knobs():
        node["roi_enabled"].setVisible(False)
    if name in {
        "inputChange",
        "mask_channel",
        "generate_trimap",
        "foreground_radius",
        "background_radius",
        "crop_mode",
        "context_padding",
        "roi_enabled",
        "processing_roi",
    }:
        if "kyven_trimap_preview_frame" in node.knobs():
            node["kyven_trimap_preview_frame"].setValue(-2147483647)
        request_trimap_preview(node)
    from kyven_nuke.live import affects_live_result, request_live_update

    if affects_live_result(name, "refine"):
        node["kyven_live_frame"].setValue(-2147483647)
        request_live_update(node)


def refresh_models() -> None:
    nuke = _nuke()
    node = nuke.thisNode()
    node_name = node.fullName()

    def refresh() -> None:
        try:
            models = [model for model in ensure_server().models() if model.get("task") == "refine"]
            labels = []
            for model in models:
                suffix = "installed" if model["installed"] else "not installed"
                if model["compatible"] is False:
                    suffix += ", VRAM warning"
                labels.append(f"{model['display_name']} [{suffix}]")
            nuke.executeInMainThread(_apply_model_labels, args=(node_name, labels))
        except Exception as exc:  # noqa: BLE001
            nuke.executeInMainThread(_set_status, args=(node_name, f"Model refresh failed: {exc}"))

    threading.Thread(target=refresh, name="kyven-refine-models", daemon=True).start()


def _apply_model_labels(node_name: str, labels: list[str]) -> None:
    node = _nuke().toNode(node_name)
    if node is not None and labels:
        previous = int(node["model"].getValue())
        node["model"].setValues(labels)
        node["model"].setValue(min(previous, len(labels) - 1))
        node["kyven_status"].setValue("Refinement model list refreshed.")


def _ensure_refine_crop_controls(node: Any) -> None:
    """Add crop modes to old nodes while preserving their previous ROI behavior."""

    nuke = _nuke()
    knobs = node.knobs()
    if "roi_section" not in knobs:
        _add_section(nuke, node, "roi_section", "PROCESSING ROI / MODEL CROP")
    if "crop_mode" not in node.knobs():
        legacy_manual = (
            bool(node["roi_enabled"].value()) if "roi_enabled" in node.knobs() else False
        )
        crop_mode = nuke.Enumeration_Knob(
            "crop_mode", "Crop Mode", ["auto", "manual", "full"]
        )
        crop_mode.setValue("manual" if legacy_manual else "full")
        _add_knob(nuke, node, crop_mode)
        _place_knob_after(node, "crop_mode", "roi_section")
    if "context_padding" not in node.knobs():
        padding = nuke.Int_Knob("context_padding", "Context Padding (px)")
        padding.setRange(0, 1024)
        padding.setValue(128)
        _add_knob(nuke, node, padding)
        _place_knob_after(node, "context_padding", "crop_mode")
    if "processing_roi" not in node.knobs():
        _add_knob(nuke, node, nuke.BBox_Knob("processing_roi", "Manual ROI"))
    if "reset_roi" not in node.knobs():
        _add_knob(
            nuke,
            node,
            nuke.PyScript_Knob(
                "reset_roi",
                "Reset ROI to Source",
                "kyven_nuke.refine_node.reset_roi_to_input()",
            ),
        )
    if "roi_help" not in node.knobs():
        _add_knob(nuke, node, nuke.Text_Knob("roi_help", "", REFINE_ROI_HELP))
    if "roi_enabled" in node.knobs():
        node["roi_enabled"].setVisible(False)


def _ensure_refine_output_controls(node: Any) -> None:
    """Build Refine and exact-trimap output branches for new or existing Groups."""

    nuke = _nuke()
    _ensure_input_order(node)
    _ensure_refine_crop_controls(node)
    if "kyven_title" in node.knobs():
        node["kyven_title"].setValue(
            '<font size="5" color="#dce9f2"><b>KYVEN / REFINE</b></font><br>'
            '<font color="#91a3b0">ViTMatte + MEMatte | Source + Mask | API 32</font>'
        )
    created_selector = "output_mode" not in node.knobs()
    previous_label = str(node["output_mode"].value()) if not created_selector else None
    if created_selector:
        _add_section(nuke, node, "output_section", "OUTPUT")
        _add_knob(
            nuke,
            node,
            nuke.Enumeration_Knob("output_mode", "Output", list(REFINE_OUTPUT_MODES)),
        )
    else:
        node["output_mode"].setValues(list(REFINE_OUTPUT_MODES))
    if created_selector:
        node["output_mode"].setValue(1)
    elif previous_label in REFINE_OUTPUT_MODES:
        node["output_mode"].setValue(REFINE_OUTPUT_MODES.index(previous_label))
    elif previous_label == "Matte":
        node["output_mode"].setValue(0)
    elif previous_label == "Source + Alpha":
        node["output_mode"].setValue(1)
    elif previous_label == "Cutout":
        node["output_mode"].setValue(2)
    elif previous_label == "Source (Bypass)":
        node["output_mode"].setValue(6)

    if "output_help" not in node.knobs():
        _add_knob(nuke, node, nuke.Text_Knob("output_help", "", REFINE_OUTPUT_HELP))
    else:
        node["output_help"].setValue(REFINE_OUTPUT_HELP)
    if "kyven_trimap_preview_frame" not in node.knobs():
        preview_frame = nuke.Int_Knob("kyven_trimap_preview_frame", "Trimap Preview Frame")
        preview_frame.setValue(-2147483647)
        preview_frame.setVisible(False)
        preview_frame.clearFlag(nuke.STARTLINE)
        node.addKnob(preview_frame)
    _ensure_double_slider(node, "foreground_radius", "Foreground Erosion (px)", 0, 100, 10)
    _ensure_double_slider(node, "background_radius", "Background Dilation (px)", 0, 100, 15)
    if "trimap_help" in node.knobs():
        node["trimap_help"].setValue(REFINE_TRIMAP_HELP)

    node.begin()
    try:
        source = nuke.toNode("Source")
        mask_input = nuke.toNode("MaskOrTrimap")
        refined = nuke.toNode("KyvenMatteSwitch")
        mask_channel = nuke.toNode("KyvenMaskChannelSwitch")
        output = nuke.toNode("Output")
        if source is None or refined is None or mask_channel is None or output is None:
            raise RuntimeError("Selected node is not a compatible Kyven Refine Group.")
        source.setXpos(-120)
        if mask_input is not None:
            mask_input.setXpos(120)

        foreground = nuke.toNode("KyvenTrimapForeground")
        if foreground is None:
            foreground = nuke.nodes.Dilate(name="KyvenTrimapForeground")
        foreground.setInput(0, mask_channel)
        foreground["size"].setExpression("-parent.foreground_radius")

        possible = nuke.toNode("KyvenTrimapPossibleForeground")
        if possible is None:
            possible = nuke.nodes.Dilate(name="KyvenTrimapPossibleForeground")
        possible.setInput(0, mask_channel)
        possible["size"].setExpression("parent.background_radius")

        foreground_half = nuke.toNode("KyvenTrimapForegroundHalf")
        if foreground_half is None:
            foreground_half = nuke.nodes.Multiply(
                name="KyvenTrimapForegroundHalf", value=0.5
            )
        foreground_half.setInput(0, foreground)
        foreground_half["value"].setValue(0.5)

        possible_half = nuke.toNode("KyvenTrimapPossibleHalf")
        if possible_half is None:
            possible_half = nuke.nodes.Multiply(
                name="KyvenTrimapPossibleHalf", value=0.5
            )
        possible_half.setInput(0, possible)
        possible_half["value"].setValue(0.5)

        generated_trimap = nuke.toNode("KyvenGeneratedTrimap")
        if generated_trimap is None:
            generated_trimap = nuke.nodes.Merge2(
                name="KyvenGeneratedTrimap", operation="plus"
            )
        generated_trimap.setInput(0, possible_half)
        generated_trimap.setInput(1, foreground_half)
        generated_trimap["operation"].setValue("plus")

        generated_selector = nuke.toNode("KyvenGeneratedTrimapSwitch")
        if generated_selector is None:
            generated_selector = nuke.nodes.Switch(name="KyvenGeneratedTrimapSwitch")
        generated_selector.setInput(0, mask_channel)
        generated_selector.setInput(1, generated_trimap)
        generated_selector["which"].setExpression("parent.generate_trimap")

        trimap = nuke.toNode("KyvenTrimapSwitch")
        if trimap is None:
            trimap = nuke.nodes.Switch(name="KyvenTrimapSwitch")
            trimap["which"].setValue(0)
        trimap.setInput(0, generated_selector)

        refined_rgba = nuke.toNode("KyvenMatteRGBA")
        if refined_rgba is None:
            refined_rgba = nuke.nodes.Copy(name="KyvenMatteRGBA")
        refined_rgba.setInput(0, refined)
        refined_rgba.setInput(1, refined)
        refined_rgba["from0"].setValue("rgba.red")
        refined_rgba["to0"].setValue("rgba.alpha")

        source_refined = nuke.toNode("KyvenSourceAlpha")
        if source_refined is None:
            source_refined = nuke.nodes.Copy(name="KyvenSourceAlpha")
        source_refined.setInput(0, source)
        source_refined.setInput(1, refined)
        source_refined["from0"].setValue("rgba.red")
        source_refined["to0"].setValue("rgba.alpha")

        refined_cutout = nuke.toNode("KyvenCutout")
        if refined_cutout is None:
            refined_cutout = nuke.nodes.Premult(name="KyvenCutout")
        refined_cutout.setInput(0, source_refined)

        trimap_rgba = nuke.toNode("KyvenTrimapRGBA")
        if trimap_rgba is None:
            trimap_rgba = nuke.nodes.Copy(name="KyvenTrimapRGBA")
        trimap_rgba.setInput(0, trimap)
        trimap_rgba.setInput(1, trimap)
        trimap_rgba["from0"].setValue("rgba.red")
        trimap_rgba["to0"].setValue("rgba.alpha")

        source_trimap = nuke.toNode("KyvenSourceTrimapAlpha")
        if source_trimap is None:
            source_trimap = nuke.nodes.Copy(name="KyvenSourceTrimapAlpha")
        source_trimap.setInput(0, source)
        source_trimap.setInput(1, trimap)
        source_trimap["from0"].setValue("rgba.red")
        source_trimap["to0"].setValue("rgba.alpha")

        trimap_cutout = nuke.toNode("KyvenTrimapCutout")
        if trimap_cutout is None:
            trimap_cutout = nuke.nodes.Premult(name="KyvenTrimapCutout")
        trimap_cutout.setInput(0, source_trimap)

        output_switch = nuke.toNode("KyvenOutputSwitch")
        if output_switch is None:
            output_switch = nuke.nodes.Switch(name="KyvenOutputSwitch")
        for index, branch in enumerate(
            (
                refined_rgba,
                source_refined,
                refined_cutout,
                trimap_rgba,
                source_trimap,
                trimap_cutout,
                source,
            )
        ):
            output_switch.setInput(index, branch)
        output_switch["which"].setExpression("parent.output_mode")
        output.setInput(0, output_switch)
    finally:
        node.end()


def _restyle_refine_ui(node: Any) -> None:
    """Apply the shared compact Kyven layout to new and upgraded Refine nodes."""

    from kyven_nuke.input_cache import ensure_controls
    ensure_controls(node)
    nuke = _nuke()
    add_node_branding(node, nuke)
    _place_knob_after(node, "open_model_manager", "refresh_models")
    sections = {
        "model_section": "MODEL AND PERFORMANCE",
        "trimap_section": "TRIMAP",
        "roi_section": "PROCESSING ROI / MODEL CROP",
        "processing_section": "INDEPENDENT FRAME PROCESSING",
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
        "tile_size": "Tile Size (0 = Auto)",
        "tile_overlap": "Tile Overlap",
        "mask_channel": "Mask Input Channel",
        "generate_trimap": "Generate Trimap from Mask",
        "foreground_radius": "Foreground Erosion (px)",
        "background_radius": "Background Dilation (px)",
        "crop_mode": "Crop Mode",
        "context_padding": "Context Padding (px)",
        "processing_roi": "Manual ROI",
        "reset_roi": "Reset ROI to Source",
        "live_mode": "Live Current Frame",
        "process_frame": "Process Current Frame",
        "cancel": "Cancel",
        "range_first": "Range First",
        "range_last": "Range Last",
        "process_range": "Process Frame Range",
        "output_mode": "Output",
        "cache_location": "Cache Folder",
        "create_matte_read": "Create Matte Read",
        "delete_node_cache": "Delete Node Cache",
        "delete_all_cache": "Delete All Kyven Cache",
        "kyven_status": "Status",
    }
    for name, label in labels.items():
        if name in node.knobs():
            node[name].setLabel(label)

    same_line = {
        "open_model_manager",
        "tile_overlap",
        "cancel",
        "range_last",
        "delete_node_cache",
    }
    for name in same_line:
        if name in node.knobs():
            node[name].clearFlag(nuke.STARTLINE)
    if "refresh_models" in node.knobs():
        node["refresh_models"].setFlag(nuke.STARTLINE)
    if "delete_all_cache" in node.knobs():
        node["delete_all_cache"].setFlag(nuke.STARTLINE)
    for name in (
        "tile_size",
        "foreground_radius",
        "background_radius",
        "create_matte_read",
    ):
        if name in node.knobs():
            node[name].setFlag(nuke.STARTLINE)

    if "kyven_title" in node.knobs():
        node["kyven_title"].setValue(
            '<font size="5" color="#dce9f2"><b>KYVEN / REFINE</b></font><br>'
            '<font color="#91a3b0">ViTMatte + MEMatte | Source + Mask | API 32</font>'
        )
    if "trimap_help" in node.knobs():
        node["trimap_help"].setValue(REFINE_TRIMAP_HELP)
    if "live_help" in node.knobs():
        node["live_help"].setValue(REFINE_LIVE_HELP)
    if "output_help" in node.knobs():
        node["output_help"].setValue(REFINE_OUTPUT_HELP)
    if "roi_help" in node.knobs():
        node["roi_help"].setValue(REFINE_ROI_HELP)
    from kyven_nuke.tooltips import apply_tooltips
    apply_tooltips(node, "refine")


def upgrade_selected_refine_node() -> None:
    """Add trimap outputs to an existing Kyven Refine node without changing its matte."""

    nuke = _nuke()
    try:
        node = nuke.selectedNode()
    except Exception:  # noqa: BLE001
        nuke.message("Select an existing Kyven Refine node first.")
        return
    try:
        _ensure_refine_output_controls(node)
        if "open_model_manager" not in node.knobs():
            _add_knob(
                nuke,
                node,
                nuke.PyScript_Knob(
                    "open_model_manager",
                    "Model Manager...",
                    "kyven_nuke.model_manager.show_model_manager()",
                ),
            )
        _ensure_server_controls(node)
        _restyle_refine_ui(node)
        for internal_name in ("KyvenMatteRead", "KyvenTrimapRead"):
            cached_data = _inside(node, internal_name)
            if cached_data is not None:
                set_data_io(cached_data)
    except Exception as exc:  # noqa: BLE001
        nuke.message(f"Could not upgrade the selected Refine node:\n{exc}")
        return
    request_trimap_preview(node)
    node["kyven_status"].setValue("Refine upgraded. Trimap preview uses CPU only.")


def create_refine_node() -> Any:
    nuke = _nuke()
    selected = nuke.selectedNode() if nuke.selectedNodes() else None
    node = _create_group_at_cursor(nuke, "KyvenRefine")
    node.setInput(SOURCE_INPUT, selected)
    node["label"].setValue("[value kyven_status]")
    node.addKnob(nuke.Tab_Knob("kyven", "Kyven Refine"))
    add_node_branding(node, nuke)
    _add_knob(
        nuke,
        node,
        nuke.Text_Knob(
            "kyven_title",
            "",
            '<font size="5" color="#dce9f2"><b>KYVEN / REFINE</b></font><br>'
            '<font color="#91a3b0">ViTMatte + MEMatte | Source + Mask | API 32</font>',
        ),
    )
    _add_section(nuke, node, "model_section", "MODEL AND PERFORMANCE")
    _add_knob(nuke, node, nuke.Enumeration_Knob("model", "Model", list(REFINE_MODEL_LABELS)))
    node["model"].setValue(default_model_index("refine", REFINE_MODEL_IDS))
    _add_knob(
        nuke,
        node,
        nuke.Enumeration_Knob("profile", "Memory Profile", ["low_memory", "balanced", "quality"]),
    )
    node["profile"].setValue(1)
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "refresh_models", "Refresh Models", "kyven_nuke.refine_node.refresh_models()"
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
    tile_size = nuke.Int_Knob("tile_size", "Tile Size (0 = Auto)")
    tile_size.setRange(0, 8192)
    tile_size.setValue(0)
    _add_knob(nuke, node, tile_size)
    tile_overlap = nuke.Int_Knob("tile_overlap", "Tile Overlap")
    tile_overlap.setRange(0, 512)
    tile_overlap.setValue(64)
    _add_knob(nuke, node, tile_overlap, start_line=False)

    _add_section(nuke, node, "trimap_section", "TRIMAP")
    _add_knob(
        nuke,
        node,
        nuke.Enumeration_Knob("mask_channel", "Mask Input Channel", ["Alpha", "Red"]),
    )
    generate = nuke.Boolean_Knob("generate_trimap", "Generate Trimap from Mask")
    generate.setValue(True)
    _add_knob(nuke, node, generate)
    _ensure_double_slider(node, "foreground_radius", "Foreground Erosion (px)", 0, 100, 10)
    _ensure_double_slider(node, "background_radius", "Background Dilation (px)", 0, 100, 15)
    _add_knob(
        nuke,
        node,
        nuke.Text_Knob(
            "trimap_help",
            "",
            REFINE_TRIMAP_HELP,
        ),
    )

    _add_section(nuke, node, "roi_section", "PROCESSING ROI / MODEL CROP")
    crop_mode = nuke.Enumeration_Knob(
        "crop_mode", "Crop Mode", ["auto", "manual", "full"]
    )
    crop_mode.setValue("auto")
    _add_knob(nuke, node, crop_mode)
    padding = nuke.Int_Knob("context_padding", "Context Padding (px)")
    padding.setRange(0, 1024)
    padding.setValue(128)
    _add_knob(nuke, node, padding)
    legacy_roi = nuke.Boolean_Knob("roi_enabled", "Legacy Processing ROI")
    legacy_roi.setVisible(False)
    _add_knob(nuke, node, legacy_roi)
    _add_knob(nuke, node, nuke.BBox_Knob("processing_roi", "Manual ROI"))
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "reset_roi", "Reset ROI to Source", "kyven_nuke.refine_node.reset_roi_to_input()"
        ),
    )
    _add_knob(nuke, node, nuke.Text_Knob("roi_help", "", REFINE_ROI_HELP))

    _add_section(nuke, node, "processing_section", "INDEPENDENT FRAME PROCESSING")
    _ensure_live_controls(node, "refine")
    _add_knob(
        nuke,
        node,
        nuke.Text_Knob(
            "live_help",
            "",
            REFINE_LIVE_HELP,
        ),
    )
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "process_frame", "Process Current Frame", "kyven_nuke.refine_node.process_current_frame()"
        ),
    )
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob("cancel", "Cancel", "kyven_nuke.refine_node.cancel_current_job()"),
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
            "kyven_nuke.refine_node.process_frame_range()",
        ),
    )

    _add_section(nuke, node, "output_section", "OUTPUT")
    _add_knob(
        nuke,
        node,
        nuke.Enumeration_Knob("output_mode", "Output", list(REFINE_OUTPUT_MODES)),
    )
    node["output_mode"].setValue(1)
    _add_knob(nuke, node, nuke.Text_Knob("output_help", "", REFINE_OUTPUT_HELP))

    internal_id = nuke.String_Knob("kyven_uuid", "UUID")
    internal_id.setValue(uuid.uuid4().hex)
    internal_id.setVisible(False)
    node.addKnob(internal_id)
    _ensure_cache_controls(
        node,
        name_field="refine_name",
        name_label="Refine Name",
        name_default="refine",
    )
    node["create_matte_read"].setCommand("kyven_nuke.node.create_read_from_current_matte()")
    _add_section(nuke, node, "status_section", "STATUS")
    status = nuke.String_Knob("kyven_status", "Status")
    status.setFlag(nuke.READ_ONLY)
    status.setValue("Ready")
    _add_knob(nuke, node, status)
    _ensure_server_controls(node)
    job_id = nuke.String_Knob("kyven_job_id", "Job ID")
    job_id.setVisible(False)
    node.addKnob(job_id)
    node["knobChanged"].setValue("kyven_nuke.refine_node.knob_changed()")

    node.begin()
    try:
        source = nuke.nodes.Input(name="Source")
        source["number"].setValue(SOURCE_INPUT)
        mask = nuke.nodes.Input(name="MaskOrTrimap")
        mask["number"].setValue(MASK_INPUT)
        alpha_extract = nuke.nodes.Shuffle(name="KyvenMaskExtractAlpha")
        alpha_extract.setInput(0, mask)
        for channel in ("red", "green", "blue", "alpha"):
            if channel in alpha_extract.knobs():
                alpha_extract[channel].setValue("alpha")
        red_extract = nuke.nodes.Shuffle(name="KyvenMaskExtractRed")
        red_extract.setInput(0, mask)
        for channel in ("red", "green", "blue", "alpha"):
            if channel in red_extract.knobs():
                red_extract[channel].setValue("red")
        mask_channel = nuke.nodes.Switch(name="KyvenMaskChannelSwitch")
        mask_channel.setInput(0, alpha_extract)
        mask_channel.setInput(1, red_extract)
        mask_channel["which"].setExpression("parent.mask_channel")
        empty = nuke.nodes.Multiply(name="KyvenEmptyMatte", value=0.0)
        empty.setInput(0, source)
        matte_switch = nuke.nodes.Switch(name="KyvenMatteSwitch")
        matte_switch.setInput(0, empty)
        matte_switch["which"].setValue(0)
        nuke.nodes.Output(name="Output")
        source_writer = nuke.nodes.Write(name="KyvenRefineSourceWrite")
        source_writer.setInput(0, source)
        source_writer["file_type"].setValue("tiff")
        source_writer["channels"].setValue("rgb")
        mask_writer = nuke.nodes.Write(name="KyvenRefineMaskWrite")
        mask_writer.setInput(0, mask_channel)
        mask_writer["file_type"].setValue("tiff")
        mask_writer["channels"].setValue("rgb")
    finally:
        node.end()
    _ensure_refine_output_controls(node)
    _restyle_refine_ui(node)
    reset_roi_to_input_for_node(node)
    knob_changed_for_node(node)
    _set_operation_controls(node, False)
    from kyven_nuke.tooltips import apply_tooltips
    apply_tooltips(node, "refine")
    return node


def reset_roi_to_input_for_node(node: Any) -> None:
    source = node.input(SOURCE_INPUT)
    if source is not None:
        node["processing_roi"].setValue(
            [0.0, 0.0, float(source.width()), float(source.height())]
        )


def knob_changed_for_node(node: Any) -> None:
    _clamp_bbox_knob(node, "processing_roi", SOURCE_INPUT)
    generated = bool(node["generate_trimap"].value())
    node["foreground_radius"].setVisible(generated)
    node["background_radius"].setVisible(generated)
    crop_mode = str(node["crop_mode"].value())
    node["processing_roi"].setVisible(crop_mode == "manual")
    node["reset_roi"].setVisible(crop_mode == "manual")
    node["context_padding"].setVisible(crop_mode == "auto")
    if "roi_enabled" in node.knobs():
        node["roi_enabled"].setVisible(False)
    if node.input(MASK_INPUT) is not None:
        request_trimap_preview(node)
