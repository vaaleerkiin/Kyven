"""Kyven Inpaint Nuke Group and LaMa orchestration."""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any

from kyven_nuke.branding import add_node_branding
from kyven_nuke.node import (
    _add_knob,
    _add_section,
    _cache_root,
    _ensure_double_slider,
    _ensure_live_controls,
    _ensure_server_controls,
    _finish_progress,
    _inside,
    _job_error_text,
    _nuke,
    _nuke_file_path,
    _place_knob_after,
    _progress_cancelled,
    _set_busy,
    _set_status,
    _start_progress,
    _update_progress,
)
from kyven_nuke.payload import INPAINT_MODEL_LABELS, inpaint_payload
from kyven_nuke.runtime import ensure_server

INPAINT_OUTPUT_MODES = (
    "Result",
    "Result + Source Alpha",
    "Result Premult",
    "Generated Patch",
    "Model Mask Preview",
    "Difference",
    "Source",
)
_mask_preview_lock = threading.RLock()
_mask_preview_worker_lock = threading.Lock()
_mask_preview_revisions: dict[str, int] = {}
_mask_preview_running: set[str] = set()


def _paths(node: Any, frame: int) -> tuple[Path, Path, Path, Path, Path]:
    root = _cache_root(node)
    return (
        root / f"inpaint_source.{frame:04d}.tif",
        root / f"inpaint_mask.{frame:04d}.png",
        root / f"inpaint_result.{frame:04d}.png",
        root / f"inpaint_processed_mask.{frame:04d}.png",
        root / f"inpaint_patch.{frame:04d}.png",
    )


def _mask_preview_paths(node: Any, frame: int, revision: int) -> tuple[Path, Path]:
    root = _cache_root(node)
    return (
        root / f"inpaint_mask_preview_input.{frame:04d}.{revision}.png",
        root / f"inpaint_model_mask_preview_r{revision}.{frame:04d}.png",
    )


def _payload(node: Any, source: Any, source_path: Path, mask_path: Path, output_path: Path, processed_mask_path: Path, patch_path: Path, mask_channel: str) -> dict[str, Any]:
    return inpaint_payload(
        source=str(source_path.resolve()), mask=str(mask_path.resolve()), output=str(output_path.resolve()), mask_output=str(processed_mask_path.resolve()), patch_output=str(patch_path.resolve()),
        model_index=int(node["model"].getValue()), profile=str(node["profile"].value()),
        image_width=int(source.width()), image_height=int(source.height()),
        crop_mode=str(node["crop_mode"].value()), roi=tuple(node["processing_roi"].value()),
        context_padding=int(node["context_padding"].value()), mask_grow=int(node["mask_grow"].value()),
        edge_color_match=(
            float(node["edge_color_match"].value())
            if "edge_color_match" in node.knobs()
            else 1.0
        ),
        mask_threshold=float(node["mask_threshold"].value()),
        invert_mask=bool(node["invert_mask"].value()), mask_channel=mask_channel, processing_size=0,
        preprocess_mask=bool(node["preprocess_mask"].value()),
    )


def _set_result(node: Any, path: Path, first: int | None = None, last: int | None = None) -> None:
    nuke = _nuke()
    node.begin()
    try:
        read = nuke.toNode("KyvenResultRead")
        if read is None:
            read = nuke.nodes.Read(name="KyvenResultRead", file=_nuke_file_path(path))
            nuke.toNode("KyvenResultSwitch").setInput(1, read)
        else:
            read["file"].setValue(_nuke_file_path(path))
        if first is not None and last is not None:
            for name, value in (("first", first), ("last", last), ("origfirst", first), ("origlast", last)):
                if name in read.knobs(): read[name].setValue(value)
        if "reload" in read.knobs(): read["reload"].execute()
        nuke.toNode("KyvenResultSwitch")["which"].setValue(1)
    finally:
        node.end()


def _set_processed_mask(node: Any, path: Path, first: int | None = None, last: int | None = None) -> None:
    nuke = _nuke(); node.begin()
    try:
        read = nuke.toNode("KyvenProcessedMaskRead")
        if read is None:
            read = nuke.nodes.Read(name="KyvenProcessedMaskRead", file=_nuke_file_path(path))
            nuke.toNode("KyvenProcessedMaskSwitch").setInput(1, read)
        else: read["file"].setValue(_nuke_file_path(path))
        if first is not None and last is not None:
            for name, value in (("first", first), ("last", last), ("origfirst", first), ("origlast", last)):
                if name in read.knobs(): read[name].setValue(value)
        if "reload" in read.knobs(): read["reload"].execute()
        nuke.toNode("KyvenProcessedMaskSwitch")["which"].setValue(1)
    finally: node.end()


def _set_patch(node: Any, path: Path, first: int | None = None, last: int | None = None) -> None:
    nuke = _nuke()
    node.begin()
    try:
        read = nuke.toNode("KyvenPatchRead")
        patch_switch = nuke.toNode("KyvenPatchSwitch")
        if patch_switch is None:
            return
        if read is None:
            read = nuke.nodes.Read(name="KyvenPatchRead", file=_nuke_file_path(path))
            patch_switch.setInput(1, read)
        else:
            read["file"].setValue(_nuke_file_path(path))
        if first is not None and last is not None:
            for name, value in (("first", first), ("last", last), ("origfirst", first), ("origlast", last)):
                if name in read.knobs():
                    read[name].setValue(value)
        if "reload" in read.knobs():
            read["reload"].execute()
        patch_switch["which"].setValue(1)
    finally:
        node.end()


def _set_model_mask_preview(node: Any, path: Path, frame: int) -> None:
    nuke = _nuke()
    node.begin()
    try:
        read = nuke.toNode("KyvenModelMaskPreviewRead")
        model_switch = nuke.toNode("KyvenModelMaskSwitch")
        if model_switch is None:
            return
        if read is None:
            read = nuke.nodes.Read(name="KyvenModelMaskPreviewRead", file=_nuke_file_path(path))
            model_switch.setInput(1, read)
        else:
            read["file"].setValue(_nuke_file_path(path))
        for name in ("first", "last", "origfirst", "origlast"):
            if name in read.knobs():
                read[name].setValue(frame)
        if "reload" in read.knobs():
            read["reload"].execute()
        model_switch["which"].setValue(1)
    finally:
        node.end()


def _ensure_inpaint_preview_graph(node: Any) -> None:
    """Add the model-mask preview branch and current six-way output switch wiring."""

    nuke = _nuke()
    if "output_mode" in node.knobs():
        previous = str(node["output_mode"].value())
        node["output_mode"].setValues(list(INPAINT_OUTPUT_MODES))
        previous = {
            "Result": "Result + Source Alpha",
            "Patch": "Generated Patch",
            "Processed Mask": "Result + Source Alpha",
            "Blend Mask": "Result + Source Alpha",
        }.get(previous, previous)
        if previous in INPAINT_OUTPUT_MODES:
            node["output_mode"].setValue(INPAINT_OUTPUT_MODES.index(previous))
    node.begin()
    try:
        channel_switch = nuke.toNode("KyvenInpaintMaskChannel")
        output_switch = nuke.toNode("KyvenOutputSwitch")
        difference = nuke.toNode("KyvenDifference")
        source = nuke.toNode("Source")
        result_switch = nuke.toNode("KyvenResultSwitch")
        result_opaque = nuke.toNode("KyvenResultOpaque")
        if result_opaque is None:
            result_opaque = nuke.nodes.Shuffle(name="KyvenResultOpaque")
        result_opaque.setInput(0, result_switch)
        if "alpha" in result_opaque.knobs():
            result_opaque["alpha"].setValue("white")
        result_source_alpha = nuke.toNode("KyvenResultSourceAlpha")
        if result_source_alpha is None:
            result_source_alpha = nuke.nodes.Copy(name="KyvenResultSourceAlpha")
        result_source_alpha.setInput(0, result_opaque)
        result_source_alpha.setInput(1, source)
        result_source_alpha["from0"].setValue("rgba.alpha")
        result_source_alpha["to0"].setValue("rgba.alpha")
        result_premult = nuke.toNode("KyvenResultPremult")
        if result_premult is None:
            result_premult = nuke.nodes.Premult(name="KyvenResultPremult")
        result_premult.setInput(0, result_source_alpha)
        patch_switch = nuke.toNode("KyvenPatchSwitch")
        if patch_switch is None:
            patch_switch = nuke.nodes.Switch(name="KyvenPatchSwitch")
            patch_switch.setInput(0, source)
        model_switch = nuke.toNode("KyvenModelMaskSwitch")
        if model_switch is None:
            model_switch = nuke.nodes.Switch(name="KyvenModelMaskSwitch")
            model_switch.setInput(0, channel_switch)
        writer = nuke.toNode("KyvenInpaintMaskPreviewWrite")
        if writer is None:
            writer = nuke.nodes.Write(name="KyvenInpaintMaskPreviewWrite")
            writer.setInput(0, channel_switch)
            writer["file_type"].setValue("png")
            writer["channels"].setValue("rgb")
        if output_switch is not None:
            output_switch.setInput(0, result_opaque)
            output_switch.setInput(1, result_source_alpha)
            output_switch.setInput(2, result_premult)
            output_switch.setInput(3, patch_switch)
            output_switch.setInput(4, model_switch)
            output_switch.setInput(5, difference)
            output_switch.setInput(6, source)
            if "preview_model_mask" in node.knobs():
                selector = "parent.preview_model_mask ? 4 : parent.output_mode"
            else:
                selector = "parent.output_mode"
            output_switch["which"].setExpression(selector)
    finally:
        node.end()


def request_mask_preview(node: Any, delay_seconds: float = 0.18) -> None:
    """Debounce mask controls into a CPU-only preview without running LaMa."""

    node_name = str(node.fullName())
    with _mask_preview_lock:
        revision = _mask_preview_revisions.get(node_name, 0) + 1
        _mask_preview_revisions[node_name] = revision
    if not _model_preview_visible(node):
        return
    timer = threading.Timer(delay_seconds, _dispatch_mask_preview, args=(node_name, revision))
    timer.daemon = True
    timer.start()


def _dispatch_mask_preview(node_name: str, revision: int) -> None:
    _nuke().executeInMainThread(_start_mask_preview, args=(node_name, revision))


def _model_preview_visible(node: Any) -> bool:
    if "preview_model_mask" in node.knobs() and bool(node["preview_model_mask"].value()):
        return True
    return (
        "output_mode" in node.knobs()
        and str(node["output_mode"].value()) == "Model Mask Preview"
    )


def _start_mask_preview(node_name: str, revision: int) -> None:
    nuke = _nuke()
    with _mask_preview_lock:
        if _mask_preview_revisions.get(node_name) != revision:
            return
        if node_name in _mask_preview_running:
            timer = threading.Timer(0.2, _dispatch_mask_preview, args=(node_name, revision))
            timer.daemon = True
            timer.start()
            return
    node = nuke.toNode(node_name)
    if node is None or not _model_preview_visible(node):
        return
    if nuke.executing() or bool(node["kyven_busy"].value()):
        return
    if node.input(1) is None:
        return
    frame = int(nuke.frame())
    input_path, model_output_path = _mask_preview_paths(
        node,
        frame,
        revision,
    )
    input_path.parent.mkdir(parents=True, exist_ok=True)
    writer = _inside(node, "KyvenInpaintMaskPreviewWrite")
    if writer is None:
        _ensure_inpaint_preview_graph(node)
        writer = _inside(node, "KyvenInpaintMaskPreviewWrite")
    writer["file"].setValue(_nuke_file_path(input_path))
    try:
        nuke.execute(writer, frame, frame)
    except Exception as exc:  # noqa: BLE001
        _set_status(node_name, f"Mask preview export failed: {exc}")
        return
    payload = {
        "mask": str(input_path.resolve()),
        "output": str(model_output_path.resolve()),
        "mask_channel": "luminance",
        "preprocess_mask": bool(node["preprocess_mask"].value()),
        "invert_mask": bool(node["invert_mask"].value()),
        "mask_threshold": float(node["mask_threshold"].value()),
        "mask_grow": int(node["mask_grow"].value()),
    }
    with _mask_preview_lock:
        _mask_preview_running.add(node_name)
    _set_status(node_name, "Updating model mask preview (CPU only)...")
    threading.Thread(
        target=_mask_preview_worker,
        args=(
            node_name,
            revision,
            frame,
            input_path,
            model_output_path,
            payload,
        ),
        name="kyven-inpaint-mask-preview",
        daemon=True,
    ).start()


def _mask_preview_worker(
    node_name: str,
    revision: int,
    frame: int,
    input_path: Path,
    model_output_path: Path,
    payload: dict[str, Any],
) -> None:
    error = ""
    result: dict[str, Any] = {}
    try:
        with _mask_preview_worker_lock:
            with _mask_preview_lock:
                current = _mask_preview_revisions.get(node_name)
            if current == revision:
                result = ensure_server().preview_inpaint_mask(payload)
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    _nuke().executeInMainThread(
        _finish_mask_preview,
        args=(
            node_name,
            revision,
            frame,
            input_path,
            model_output_path,
            result,
            error,
        ),
    )


def _finish_mask_preview(
    node_name: str,
    revision: int,
    frame: int,
    input_path: Path,
    model_output_path: Path,
    result: dict[str, Any],
    error: str,
) -> None:
    input_path.unlink(missing_ok=True)
    with _mask_preview_lock:
        _mask_preview_running.discard(node_name)
        current = _mask_preview_revisions.get(node_name)
    if current != revision:
        model_output_path.unlink(missing_ok=True)
        if current is not None and node_name not in _mask_preview_running:
            _start_mask_preview(node_name, current)
        return
    node = _nuke().toNode(node_name)
    if node is None:
        model_output_path.unlink(missing_ok=True)
        return
    if error or not model_output_path.is_file():
        _set_status(node_name, f"Mask preview failed: {error or 'output was not created'}")
        return
    _set_model_mask_preview(node, model_output_path, frame)
    mode = "preprocessed" if result.get("preprocess_mask") else "clean input"
    _set_status(node_name, f"Model mask preview ready ({mode}).")


def _apply(node_name: str, job: dict[str, Any], output: Path, processed_mask: Path, patch: Path) -> None:
    node = _nuke().toNode(node_name)
    if node is None: return
    _set_busy(node_name, False)
    if job["status"] != "succeeded":
        _set_status(node_name, f"Inpaint failed: {_job_error_text(job)}")
        if _model_preview_visible(node):
            request_mask_preview(node)
        return
    _set_result(node, output)
    _set_processed_mask(node, processed_mask)
    _set_patch(node, patch)
    roi = (job.get("result") or {}).get("metadata", {}).get("processing_roi")
    suffix = f" | crop {roi['width']}x{roi['height']}" if roi else ""
    _set_status(node_name, f"Inpaint complete{suffix}")
    if _model_preview_visible(node):
        request_mask_preview(node)


def process_current_frame_for_node(node: Any, live: bool = False) -> None:
    nuke = _nuke()
    node_name = str(node.fullName())
    if bool(node["kyven_busy"].value()):
        if not live:
            _set_status(node_name, "Inpaint is already processing this node.")
        return
    source, mask = node.input(0), node.input(1)
    if source is None or mask is None:
        if not live:
            nuke.message("Kyven Inpaint needs Source on input 0 and Mask on input 1.")
        return
    _set_busy(node_name, True)
    with _mask_preview_lock:
        _mask_preview_revisions[node_name] = _mask_preview_revisions.get(node_name, 0) + 1
    frame = int(nuke.frame())
    source_path, mask_path, output_path, processed_mask_path, patch_path = _paths(node, frame)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_writer = _inside(node, "KyvenInpaintSourceWrite")
    combined = _inside(node, "KyvenInpaintCombined") is not None
    mask_writer = None if combined else _inside(node, "KyvenInpaintMaskWrite")
    source_writer["file"].setValue(_nuke_file_path(source_path))
    if combined:
        mask_path = source_path
    else:
        mask_writer["file"].setValue(_nuke_file_path(mask_path))
    _set_status(node.fullName(), "Exporting inpaint inputs...")
    show_progress = not live
    if show_progress:
        _start_progress(node_name, "Kyven Inpaint", f"Exporting frame {frame}")
    try:
        nuke.execute(source_writer, frame, frame)
        if mask_writer is not None: nuke.execute(mask_writer, frame, frame)
    except Exception as exc:  # noqa: BLE001
        if show_progress:
            _finish_progress(node_name)
            nuke.message(f"Inpaint export failed: {exc}")
        else:
            _set_status(node_name, f"Live inpaint export failed: {exc}")
        _set_busy(node_name, False)
        return
    payload = _payload(node, source, source_path, mask_path, output_path, processed_mask_path, patch_path, "alpha" if combined else "luminance")
    def work() -> None:
        try:
            client = ensure_server(); job_id = client.submit_inpaint(payload)
            _nuke().executeInMainThread(lambda: node["kyven_job_id"].setValue(job_id))
            while True:
                job = client.job(job_id)
                if job["status"] in ("succeeded", "failed", "cancelled"): break
                progress = float(job.get("progress") or 0.0)
                message = str(job.get("progress_message") or "Inpainting")
                if show_progress:
                    _nuke().executeInMainThread(_update_progress, args=(node_name, int(progress * 100), message))
                    cancelled = _nuke().executeInMainThreadWithResult(
                        _progress_cancelled,
                        args=(node_name,),
                    )
                    if cancelled:
                        client.cancel(job_id)
                time.sleep(0.15)
            _nuke().executeInMainThread(_apply, args=(node_name, job, output_path, processed_mask_path, patch_path))
        except Exception as exc:  # noqa: BLE001
            _nuke().executeInMainThread(_set_busy, args=(node_name, False))
            _nuke().executeInMainThread(_set_status, args=(node_name, f"Inpaint failed: {exc}"))
        finally:
            if show_progress:
                _nuke().executeInMainThread(_finish_progress, args=(node_name,))
    threading.Thread(target=work, name="kyven-inpaint", daemon=True).start()


def process_current_frame(node: Any | None = None, live: bool = False) -> None:
    process_current_frame_for_node(node or _nuke().thisNode(), live=live)


def process_frame_range() -> None:
    nuke = _nuke(); node = nuke.thisNode(); source, mask = node.input(0), node.input(1)
    node_name = str(node.fullName())
    if bool(node["kyven_busy"].value()):
        _set_status(node_name, "Inpaint is already processing this node.")
        return
    first = int(node["range_first"].value()); last = int(node["range_last"].value())
    if source is None or mask is None: nuke.message("Connect Source and Mask first."); return
    if last < first: nuke.message("Range Last must be equal to or greater than Range First."); return
    _set_busy(node_name, True)
    with _mask_preview_lock:
        _mask_preview_revisions[node_name] = _mask_preview_revisions.get(node_name, 0) + 1
    root = _cache_root(node); source_pattern = root / "inpaint_source.%04d.tif"; mask_pattern = root / "inpaint_mask.%04d.png"; output_pattern = root / "inpaint_result.%04d.png"; processed_mask_pattern = root / "inpaint_processed_mask.%04d.png"; patch_pattern = root / "inpaint_patch.%04d.png"
    source_writer = _inside(node, "KyvenInpaintSourceWrite")
    combined = _inside(node, "KyvenInpaintCombined") is not None
    mask_writer = None if combined else _inside(node, "KyvenInpaintMaskWrite")
    source_writer["file"].setValue(_nuke_file_path(source_pattern))
    if combined:
        mask_pattern = source_pattern
    else:
        mask_writer["file"].setValue(_nuke_file_path(mask_pattern))
    _start_progress(node_name, "Kyven Inpaint", f"Exporting frames {first}-{last}")
    try:
        nuke.execute(source_writer, first, last)
        if mask_writer is not None: nuke.execute(mask_writer, first, last)
    except Exception as exc:  # noqa: BLE001
        _finish_progress(node_name); _set_busy(node_name, False); nuke.message(f"Inpaint export failed: {exc}"); return
    payloads = [
        _payload(node, source, Path(str(source_pattern) % frame), Path(str(mask_pattern) % frame), Path(str(output_pattern) % frame), Path(str(processed_mask_pattern) % frame), Path(str(patch_pattern) % frame), "alpha" if combined else "luminance")
        for frame in range(first, last + 1)
    ]
    def work() -> None:
        try:
            client = ensure_server(); total = len(payloads)
            for index, payload in enumerate(payloads, start=1):
                cancelled = _nuke().executeInMainThreadWithResult(
                    _progress_cancelled,
                    args=(node_name,),
                )
                if cancelled:
                    break
                job_id = client.submit_inpaint(payload)
                _nuke().executeInMainThread(lambda value=job_id: node["kyven_job_id"].setValue(value))
                job = client.wait(job_id)
                if job["status"] != "succeeded":
                    error = job.get("error") or {}; raise RuntimeError(error.get("message", job["status"]))
                _nuke().executeInMainThread(_update_progress, args=(node_name, int(index * 100 / total), f"Inpaint frame {first + index - 1} ({index}/{total})"))
            cancelled = _nuke().executeInMainThreadWithResult(
                _progress_cancelled,
                args=(node_name,),
            )
            if not cancelled:
                _nuke().executeInMainThread(_set_result, args=(node, output_pattern, first, last))
                _nuke().executeInMainThread(_set_processed_mask, args=(node, processed_mask_pattern, first, last))
                _nuke().executeInMainThread(_set_patch, args=(node, patch_pattern, first, last))
                _nuke().executeInMainThread(_set_status, args=(node_name, f"Inpaint range ready: {first}-{last}"))
        except Exception as exc:  # noqa: BLE001
            _nuke().executeInMainThread(_set_status, args=(node_name, f"Inpaint range failed: {exc}"))
        finally:
            _nuke().executeInMainThread(_finish_progress, args=(node_name,))
            _nuke().executeInMainThread(_set_busy, args=(node_name, False))
    threading.Thread(target=work, name="kyven-inpaint-range", daemon=True).start()


def cancel_current_job() -> None:
    node = _nuke().thisNode(); job_id = str(node["kyven_job_id"].value())
    if job_id:
        threading.Thread(target=lambda: ensure_server().cancel(job_id), daemon=True).start()


def create_read_from_current_result() -> None:
    """Create a root-level Read matching the cached result frame or sequence."""

    nuke = _nuke()
    node = nuke.thisNode()
    node.begin()
    try:
        cached = nuke.toNode("KyvenResultRead")
        switch = nuke.toNode("KyvenResultSwitch")
        if cached is None or switch is None or int(switch["which"].value()) != 1:
            nuke.message("Process a frame or frame range first; this node has no cached result yet.")
            return
        file_path = str(cached["file"].value())
        frame_values = {
            name: int(cached[name].value())
            for name in ("first", "last", "origfirst", "origlast")
            if name in cached.knobs()
        }
    finally:
        node.end()

    read = nuke.nodes.Read(file=file_path)
    for name, value in frame_values.items():
        if name in read.knobs():
            read[name].setValue(value)
    read["label"].setValue("Kyven cached result")
    read.setXpos(node.xpos() + 140)
    read.setYpos(node.ypos() + 120)
    read.setSelected(True)
    _set_status(node.fullName(), f"Created Read: {read.name()}")


def delete_this_node_cache() -> None:
    from kyven_nuke.node import delete_node_cache

    delete_node_cache()


def reset_roi_to_input() -> None:
    node = _nuke().thisNode(); source = node.input(0)
    if source is not None: node["processing_roi"].setValue([0, 0, float(source.width()), float(source.height())])


def knob_changed() -> None:
    nuke = _nuke(); node = nuke.thisNode(); knob_name = nuke.thisKnob().name()
    node["processing_roi"].setVisible(str(node["crop_mode"].value()) == "manual")
    node["context_padding"].setVisible(str(node["crop_mode"].value()) == "auto")
    if knob_name == "model":
        if int(node["model"].getValue()) == 0:
            node["model_help"].setValue("Fast: fixed 512 x 512 model input, CPU-friendly and best for Live. Use a tight Auto ROI for more detail.")
        else:
            node["model_help"].setValue("Quality: native ROI resolution (padded only to a multiple of 8). Better detail, but slower and uses more memory.")
    preprocess = bool(node["preprocess_mask"].value()) if "preprocess_mask" in node.knobs() else True
    for name in ("invert_mask", "mask_threshold", "mask_grow"):
        if name in node.knobs():
            node[name].setVisible(preprocess)
    if knob_name in {
        "inputChange",
        "output_mode",
        "mask_channel",
        "preprocess_mask",
        "preview_model_mask",
        "invert_mask",
        "mask_threshold",
        "mask_grow",
    }:
        request_mask_preview(node)
    from kyven_nuke.live import affects_live_result, request_live_update
    if affects_live_result(knob_name, "inpaint"):
        node["kyven_live_frame"].setValue(-2147483647)
        request_live_update(node)


def refresh_models() -> None:
    nuke = _nuke()
    node = nuke.thisNode()
    node_name = node.fullName()

    def refresh() -> None:
        try:
            models = [model for model in ensure_server().models() if model.get("task") == "inpaint"]
            labels = []
            for model in models:
                suffix = "installed" if model["installed"] else "not installed"
                if model["compatible"] is False:
                    suffix += ", VRAM warning"
                labels.append(f"{model['display_name']} [{suffix}]")
            nuke.executeInMainThread(_apply_model_labels, args=(node_name, labels))
        except Exception as exc:  # noqa: BLE001
            nuke.executeInMainThread(_set_status, args=(node_name, f"Model refresh failed: {exc}"))

    threading.Thread(target=refresh, name="kyven-inpaint-models", daemon=True).start()


def _apply_model_labels(node_name: str, labels: list[str]) -> None:
    node = _nuke().toNode(node_name)
    if node is not None and labels:
        previous = int(node["model"].getValue())
        node["model"].setValues(labels)
        node["model"].setValue(min(previous, len(labels) - 1))
        node["kyven_status"].setValue("Inpaint model list refreshed.")


def _ensure_inpaint_mask_sliders(node: Any) -> None:
    """Use full-width sliders for all continuous input-mask controls."""

    nuke = _nuke()
    if "preprocess_mask" not in node.knobs():
        enabled = nuke.Boolean_Knob("preprocess_mask", "Preprocess Input Mask")
        enabled.setValue(True)
        _add_knob(nuke, node, enabled)
    if "preview_model_mask" not in node.knobs():
        preview_model = nuke.Boolean_Knob("preview_model_mask", "Preview Model Mask")
        preview_model.setValue(False)
        _add_knob(nuke, node, preview_model, start_line=False)
    for obsolete in ("preview_blend_mask", "preview_processed_mask", "blend_grow", "mask_feather"):
        if obsolete in node.knobs():
            node[obsolete].setVisible(False)
            if obsolete.startswith("preview_"):
                node[obsolete].setValue(False)
    _place_knob_after(node, "preprocess_mask", "mask_channel")
    _place_knob_after(node, "preview_model_mask", "preprocess_mask")
    node["preview_model_mask"].clearFlag(nuke.STARTLINE)
    _ensure_double_slider(node, "mask_threshold", "Threshold", 0, 1, 0.5)
    _ensure_double_slider(node, "mask_grow", "Model Mask Grow (px)", -128, 128, 12)
    _ensure_double_slider(node, "edge_color_match", "Edge Color Match", 0, 1, 1)
    preprocess = bool(node["preprocess_mask"].value())
    for name in ("invert_mask", "mask_threshold", "mask_grow"):
        if name in node.knobs():
            node[name].setVisible(preprocess)


def _restyle_inpaint_cache(node: Any) -> None:
    """Apply the shared Cache labels and compact three-row layout."""

    nuke = _nuke()
    labels = {
        "cache_folder": "Cache Folder",
        "cache_location": "Cache Folder",
        "create_result_read": "Create Result Read",
        "delete_node_cache": "Delete Node Cache",
        "delete_all_cache": "Delete All Kyven Cache",
    }
    for name, label in labels.items():
        if name in node.knobs():
            node[name].setLabel(label)
    if "create_result_read" in node.knobs():
        node["create_result_read"].setFlag(nuke.STARTLINE)
        node["create_result_read"].setCommand(
            "kyven_nuke.inpaint_node.create_read_from_current_result()"
        )
    if "delete_node_cache" in node.knobs():
        node["delete_node_cache"].clearFlag(nuke.STARTLINE)
        node["delete_node_cache"].setCommand(
            "kyven_nuke.inpaint_node.delete_this_node_cache()"
        )
    if "delete_all_cache" in node.knobs():
        node["delete_all_cache"].setFlag(nuke.STARTLINE)
        node["delete_all_cache"].setCommand("kyven_nuke.node.delete_all_cache()")


def create_inpaint_node() -> Any:
    nuke = _nuke(); selected = nuke.selectedNodes(); source = selected[0] if selected else None; mask = selected[1] if len(selected) > 1 else None
    node = nuke.nodes.Group(name="KyvenInpaint"); node.setInput(0, source); node.setInput(1, mask)
    node["label"].setValue("[value kyven_status]"); node.addKnob(nuke.Tab_Knob("kyven", "Kyven Inpaint")); add_node_branding(node, nuke)
    _add_knob(nuke, node, nuke.Text_Knob("kyven_title", "", '<font size="5" color="#dce9f2"><b>KYVEN / INPAINT</b></font><br><font color="#91a3b0">LaMa | Source + Mask | API 18</font>'))
    _add_section(nuke, node, "model_section", "MODEL AND PERFORMANCE")
    _add_knob(nuke, node, nuke.Enumeration_Knob("model", "Model", list(INPAINT_MODEL_LABELS)))
    _add_knob(nuke, node, nuke.Enumeration_Knob("profile", "Memory Profile", ["low_memory", "balanced", "quality"])); node["profile"].setValue(1)
    _add_knob(nuke, node, nuke.PyScript_Knob("refresh_models", "Refresh Models", "kyven_nuke.inpaint_node.refresh_models()"))
    _add_knob(nuke, node, nuke.PyScript_Knob("open_model_manager", "Model Manager...", "kyven_nuke.model_manager.show_model_manager()"), start_line=False)
    size = nuke.Int_Knob("processing_size", "Processing Size"); size.setValue(512); size.setVisible(False); node.addKnob(size)
    _add_knob(nuke, node, nuke.Text_Knob("model_help", "", "Fast: fixed 512 x 512 model input, CPU-friendly and best for Live. Use a tight Auto ROI for more detail."))
    _add_section(nuke, node, "mask_section", "MASK")
    _add_knob(nuke, node, nuke.Enumeration_Knob("mask_channel", "Input 1 Channel", ["Alpha", "Red"]))
    preprocess = nuke.Boolean_Knob("preprocess_mask", "Preprocess Input Mask"); preprocess.setValue(True); _add_knob(nuke, node, preprocess)
    invert = nuke.Boolean_Knob("invert_mask", "Invert Input Mask"); _add_knob(nuke, node, invert)
    _ensure_inpaint_mask_sliders(node)
    _add_knob(nuke, node, nuke.Text_Knob("mask_help", "", "Preview Model Mask shows exactly what LaMa receives. Threshold converts soft gray pixels to black or white; Model Grow expands or erodes that binary area. The clean input mask is used for the final composite."))
    _add_section(nuke, node, "roi_section", "PROCESSING ROI / MODEL CROP")
    _add_knob(nuke, node, nuke.Enumeration_Knob("crop_mode", "Crop Mode", ["auto", "manual", "full"]))
    padding = nuke.Int_Knob("context_padding", "Context Padding (px)"); padding.setRange(0, 1024); padding.setValue(128); _add_knob(nuke, node, padding)
    _add_knob(nuke, node, nuke.BBox_Knob("processing_roi", "Manual ROI"))
    _add_knob(nuke, node, nuke.PyScript_Knob("reset_roi", "Reset ROI to Source", "kyven_nuke.inpaint_node.reset_roi_to_input()"))
    _add_knob(nuke, node, nuke.Text_Knob("roi_help", "", "Auto crops to model-mask bounds plus context. The generated result is composited through the clean input mask."))
    _add_section(nuke, node, "processing_section", "INDEPENDENT FRAME PROCESSING")
    _ensure_live_controls(node, "inpaint")
    _add_knob(nuke, node, nuke.PyScript_Knob("process_frame", "Process Current Frame", "kyven_nuke.inpaint_node.process_current_frame()"))
    _add_knob(nuke, node, nuke.PyScript_Knob("cancel", "Cancel", "kyven_nuke.inpaint_node.cancel_current_job()"), start_line=False)
    first = nuke.Int_Knob("range_first", "Range First"); first.setValue(int(nuke.root().firstFrame())); _add_knob(nuke, node, first)
    last = nuke.Int_Knob("range_last", "Range Last"); last.setValue(int(nuke.root().lastFrame())); _add_knob(nuke, node, last, start_line=False)
    _add_knob(nuke, node, nuke.PyScript_Knob("process_range", "Process Frame Range", "kyven_nuke.inpaint_node.process_frame_range()"))
    _add_section(nuke, node, "output_section", "OUTPUT")
    _add_knob(nuke, node, nuke.Enumeration_Knob("output_mode", "Output", list(INPAINT_OUTPUT_MODES)))
    node["output_mode"].setValue(1)
    _add_knob(nuke, node, nuke.Text_Knob("output_help", "", "Result is opaque RGB. Result + Source Alpha is the default. Result Premult applies that alpha. Generated Patch is the uncomposited model RGB for an external Merge."))
    uid = nuke.String_Knob("kyven_uuid", "UUID"); uid.setValue(uuid.uuid4().hex); uid.setVisible(False); node.addKnob(uid)
    _add_section(nuke, node, "cache_section", "CACHE")
    folder = nuke.String_Knob("cache_location", "Cache Folder"); folder.setFlag(nuke.READ_ONLY); folder.setValue(str(_cache_root(node))); _add_knob(nuke, node, folder)
    _add_knob(nuke, node, nuke.PyScript_Knob("create_result_read", "Create Result Read", "kyven_nuke.inpaint_node.create_read_from_current_result()"))
    _add_knob(nuke, node, nuke.PyScript_Knob("delete_node_cache", "Delete Node Cache", "kyven_nuke.inpaint_node.delete_this_node_cache()"), start_line=False)
    _add_knob(nuke, node, nuke.PyScript_Knob("delete_all_cache", "Delete All Kyven Cache", "kyven_nuke.node.delete_all_cache()"))
    job = nuke.String_Knob("kyven_job_id", "Job ID"); job.setVisible(False); node.addKnob(job)
    _add_section(nuke, node, "status_section", "STATUS"); status = nuke.String_Knob("kyven_status", "Status"); status.setFlag(nuke.READ_ONLY); status.setValue("Ready"); _add_knob(nuke, node, status); _ensure_server_controls(node)
    node["knobChanged"].setValue("kyven_nuke.inpaint_node.knob_changed()")
    node.begin()
    try:
        src = nuke.nodes.Input(name="Source"); src["number"].setValue(0)
        msk = nuke.nodes.Input(name="Mask"); msk["number"].setValue(1)
        result_switch = nuke.nodes.Switch(name="KyvenResultSwitch"); result_switch.setInput(0, src)
        processed_mask = nuke.nodes.Switch(name="KyvenProcessedMaskSwitch"); processed_mask.setInput(0, msk)
        result_opaque = nuke.nodes.Shuffle(name="KyvenResultOpaque"); result_opaque.setInput(0, result_switch); result_opaque["alpha"].setValue("white")
        result_alpha = nuke.nodes.Copy(name="KyvenResultSourceAlpha"); result_alpha.setInput(0, result_opaque); result_alpha.setInput(1, src); result_alpha["from0"].setValue("rgba.alpha"); result_alpha["to0"].setValue("rgba.alpha")
        result_premult = nuke.nodes.Premult(name="KyvenResultPremult"); result_premult.setInput(0, result_alpha)
        patch = nuke.nodes.Switch(name="KyvenPatchSwitch"); patch.setInput(0, src)
        difference = nuke.nodes.Merge2(name="KyvenDifference", operation="difference"); difference.setInput(0, result_switch); difference.setInput(1, src)
        output_switch = nuke.nodes.Switch(name="KyvenOutputSwitch"); output_switch.setInput(0, result_opaque); output_switch.setInput(1, result_alpha); output_switch.setInput(2, result_premult); output_switch.setInput(3, patch); output_switch.setInput(4, processed_mask); output_switch.setInput(5, difference); output_switch.setInput(6, src); output_switch["which"].setExpression("parent.output_mode")
        out = nuke.nodes.Output(name="Output"); out.setInput(0, output_switch)
        sw = nuke.nodes.Write(name="KyvenInpaintSourceWrite"); sw.setInput(0, src); sw["file_type"].setValue("tiff"); sw["channels"].setValue("rgb")
        alpha = nuke.nodes.Shuffle(name="KyvenInpaintMaskAlpha"); alpha.setInput(0, msk)
        red = nuke.nodes.Shuffle(name="KyvenInpaintMaskRed"); red.setInput(0, msk)
        for channel in ("red", "green", "blue", "alpha"):
            if channel in alpha.knobs(): alpha[channel].setValue("alpha")
            if channel in red.knobs(): red[channel].setValue("red")
        channel_switch = nuke.nodes.Switch(name="KyvenInpaintMaskChannel"); channel_switch.setInput(0, alpha); channel_switch.setInput(1, red); channel_switch["which"].setExpression("parent.mask_channel")
        combined = nuke.nodes.Copy(name="KyvenInpaintCombined"); combined.setInput(0, src); combined.setInput(1, channel_switch); combined["from0"].setValue("rgba.red"); combined["to0"].setValue("rgba.alpha")
        sw.setInput(0, combined); sw["channels"].setValue("rgba")
        if "compression" in sw.knobs(): sw["compression"].setValue(0)
        if "datatype" in sw.knobs(): sw["datatype"].setValue("8 bit")
    finally: node.end()
    _ensure_inpaint_preview_graph(node)
    reset = source
    if reset is not None: node["processing_roi"].setValue([0, 0, float(reset.width()), float(reset.height())])
    node["processing_roi"].setVisible(False)
    _restyle_inpaint_cache(node)
    add_node_branding(node, nuke)
    if mask is not None:
        request_mask_preview(node)
    return node


def upgrade_selected_inpaint_node() -> None:
    nuke = _nuke()
    selected = nuke.selectedNodes()
    if len(selected) != 1 or selected[0].Class() != "Group":
        nuke.message("Select one Kyven Inpaint Group node first.")
        return
    node = selected[0]
    if "create_result_read" not in node.knobs() or "kyven_uuid" not in node.knobs():
        nuke.message("The selected Group is not a Kyven Inpaint node.")
        return
    _ensure_inpaint_mask_sliders(node)
    _ensure_inpaint_preview_graph(node)
    if "refresh_models" not in node.knobs():
        _add_knob(
            nuke,
            node,
            nuke.PyScript_Knob(
                "refresh_models",
                "Refresh Models",
                "kyven_nuke.inpaint_node.refresh_models()",
            ),
        )
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
    _place_knob_after(node, "open_model_manager", "refresh_models")
    if "delete_all_cache" not in node.knobs():
        tail = [
            node[name]
            for name in ("kyven_job_id", "status_section", "kyven_status")
            if name in node.knobs()
        ]
        for knob in tail:
            node.removeKnob(knob)
        _add_knob(
            nuke,
            node,
            nuke.PyScript_Knob(
                "delete_all_cache",
                "Delete All Kyven Cache",
                "kyven_nuke.node.delete_all_cache()",
            ),
        )
        for knob in tail:
            node.addKnob(knob)
    _ensure_server_controls(node)
    _restyle_inpaint_cache(node)
    add_node_branding(node, nuke)
    if "kyven_title" in node.knobs():
        node["kyven_title"].setValue(
            '<font size="5" color="#dce9f2"><b>KYVEN / INPAINT</b></font><br>'
            '<font color="#91a3b0">LaMa | Source + Mask | API 18</font>'
        )
    if "mask_help" in node.knobs():
        node["mask_help"].setValue(
            "Preview Model Mask shows exactly what LaMa receives. Threshold converts soft gray "
            "pixels to black or white; Model Grow expands or erodes that binary area. The clean "
            "input mask is used for the final composite."
        )
    if "output_help" in node.knobs():
        node["output_help"].setValue(
            "Result is opaque RGB. Result + Source Alpha is the default. Result Premult applies "
            "that alpha. Generated Patch is the uncomposited model RGB for an external Merge."
        )
    if "kyven_status" in node.knobs():
        node["kyven_status"].setValue("Inpaint UI upgraded. Cached results were preserved.")
    if node.input(1) is not None:
        request_mask_preview(node)
