"""Kyven Inpaint Nuke Group and LaMa orchestration."""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any

from kyven_nuke.branding import add_node_branding
from kyven_nuke.client import NukeKyvenClientError
from kyven_nuke.color_management import match_color_io, set_data_io, set_interchange_color_io
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
    "Result + Mask Alpha",
    "Result Premult",
    "Generated Patch",
    "Difference",
    "Source",
)
MASK_INPUT = 0
SOURCE_INPUT = 1


def _ensure_input_order(node: Any) -> None:
    """Keep Source on the left connector and Mask on the right without swapping media."""

    old_source = node.input(0)
    old_mask = node.input(1)
    needs_migration = False
    node.begin()
    try:
        source = _nuke().toNode("Source")
        mask = _nuke().toNode("Mask")
        if source is not None and mask is not None:
            needs_migration = int(source["number"].value()) == 0
            source["number"].setValue(SOURCE_INPUT)
            mask["number"].setValue(MASK_INPUT)
    finally:
        node.end()
    if needs_migration:
        node.setInput(MASK_INPUT, old_mask)
        node.setInput(SOURCE_INPUT, old_source)


def _paths(node: Any, frame: int) -> tuple[Path, Path, Path, Path, Path, Path]:
    root = _cache_root(node)
    return (
        root / f"inpaint_source.{frame:04d}.tif",
        root / f"inpaint_mask.{frame:04d}.png",
        root / f"inpaint_model_mask.{frame:04d}.png",
        root / f"inpaint_result.{frame:04d}.png",
        root / f"inpaint_processed_mask.{frame:04d}.png",
        root / f"inpaint_patch.{frame:04d}.png",
    )


def _payload(node: Any, source: Any, source_path: Path, mask_path: Path, model_mask_path: Path, output_path: Path, processed_mask_path: Path, patch_path: Path, mask_channel: str) -> dict[str, Any]:
    return inpaint_payload(
        source=str(source_path.resolve()), mask=str(mask_path.resolve()), model_mask=str(model_mask_path.resolve()), output=str(output_path.resolve()), mask_output=str(processed_mask_path.resolve()), patch_output=str(patch_path.resolve()),
        model_index=int(node["model"].getValue()), profile=str(node["profile"].value()),
        image_width=int(source.width()), image_height=int(source.height()),
        crop_mode=str(node["crop_mode"].value()), roi=tuple(node["processing_roi"].value()),
        context_padding=int(node["context_padding"].value()), mask_grow=int(node["mask_grow"].value()),
        edge_color_match=(
            float(node["edge_color_match"].value())
            if "edge_color_match" in node.knobs()
            else 1.0
        ),
        edge_softness=(
            float(node["edge_softness"].value())
            if "edge_softness" in node.knobs()
            else 6.0
        ),
        mask_threshold=float(node["mask_threshold"].value()),
        invert_mask=bool(node["invert_mask"].value()), mask_channel=mask_channel, processing_size=0,
        preprocess_mask=bool(node["preprocess_mask"].value()),
        quality_mode=(
            "refined"
            if "quality_mode" in node.knobs() and int(node["quality_mode"].getValue()) == 1
            else "standard"
        ),
        refinement_steps=(
            int(node["refinement_steps"].value())
            if "refinement_steps" in node.knobs()
            else 15
        ),
        refinement_strength=(
            float(node["refinement_strength"].value())
            if "refinement_strength" in node.knobs()
            else 1.0
        ),
        refinement_scales=(
            int(node["refinement_scales"].value())
            if "refinement_scales" in node.knobs()
            else 3
        ),
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
        source_writer = nuke.toNode("KyvenInpaintSourceWrite")
        if source_writer is not None:
            match_color_io(source_writer, read)
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
        set_data_io(read)
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
        source_writer = nuke.toNode("KyvenInpaintSourceWrite")
        if source_writer is not None:
            match_color_io(source_writer, read)
        if first is not None and last is not None:
            for name, value in (("first", first), ("last", last), ("origfirst", first), ("origlast", last)):
                if name in read.knobs():
                    read[name].setValue(value)
        if "reload" in read.knobs():
            read["reload"].execute()
        patch_switch["which"].setValue(1)
    finally:
        node.end()


def _ensure_inpaint_preview_graph(node: Any) -> None:
    """Build the live Nuke model mask and wire the public output modes."""

    nuke = _nuke()
    _ensure_input_order(node)
    if "output_mode" in node.knobs():
        previous = str(node["output_mode"].value())
        node["output_mode"].setValues(list(INPAINT_OUTPUT_MODES))
        previous = {
            "Result": "Result + Mask Alpha",
            "Result + Source Alpha": "Result + Mask Alpha",
            "Patch": "Generated Patch",
            "Model Mask Preview": "Result + Mask Alpha",
            "Processed Mask": "Result + Mask Alpha",
            "Blend Mask": "Result + Mask Alpha",
        }.get(previous, previous)
        if previous in INPAINT_OUTPUT_MODES:
            node["output_mode"].setValue(INPAINT_OUTPUT_MODES.index(previous))
    node.begin()
    try:
        channel_switch = nuke.toNode("KyvenInpaintMaskChannel")
        output_switch = nuke.toNode("KyvenOutputSwitch")
        difference = nuke.toNode("KyvenDifference")
        source = nuke.toNode("Source")
        mask = nuke.toNode("Mask")
        if source is not None:
            source.setXpos(-120)
        if mask is not None:
            mask.setXpos(120)
        result_switch = nuke.toNode("KyvenResultSwitch")
        result_opaque = nuke.toNode("KyvenResultOpaque")
        if result_opaque is None:
            result_opaque = nuke.nodes.Shuffle(name="KyvenResultOpaque")
        result_opaque.setInput(0, result_switch)
        if "alpha" in result_opaque.knobs():
            result_opaque["alpha"].setValue("white")
        patch_switch = nuke.toNode("KyvenPatchSwitch")
        if patch_switch is None:
            patch_switch = nuke.nodes.Switch(name="KyvenPatchSwitch")
            patch_switch.setInput(0, source)
        threshold = nuke.toNode("KyvenModelMaskThreshold")
        if threshold is None:
            threshold = nuke.nodes.Expression(name="KyvenModelMaskThreshold")
        threshold.setInput(0, channel_switch)
        for index, channel in enumerate(("r", "g", "b", "a")):
            expression = (
                "parent.preprocess_mask ? "
                f"((parent.invert_mask ? 1-{channel} : {channel}) >= parent.mask_threshold) "
                f": ({channel} >= 0.5)"
            )
            knob_name = f"expr{index}"
            if knob_name in threshold.knobs():
                threshold[knob_name].setValue(expression)
        model_mask = nuke.toNode("KyvenModelMaskGrow")
        if model_mask is None:
            model_mask = nuke.nodes.FilterErode(name="KyvenModelMaskGrow")
        model_mask.setInput(0, threshold)
        if "channels" in model_mask.knobs():
            model_mask["channels"].setValue("rgba")
        model_mask["size"].setExpression(
            "parent.preprocess_mask ? -parent.mask_grow : 0"
        )
        effective_mask = nuke.toNode("KyvenEffectiveMask")
        if effective_mask is None:
            effective_mask = nuke.nodes.Switch(name="KyvenEffectiveMask")
        effective_mask.setInput(0, channel_switch)
        effective_mask.setInput(1, model_mask)
        effective_mask["which"].setExpression("parent.preprocess_mask")
        processed_mask = nuke.toNode("KyvenProcessedMaskSwitch")
        if processed_mask is None:
            processed_mask = nuke.nodes.Switch(name="KyvenProcessedMaskSwitch")
        processed_mask.setInput(0, effective_mask)
        patch_mask_alpha = nuke.toNode("KyvenPatchMaskAlpha")
        if patch_mask_alpha is None:
            patch_mask_alpha = nuke.nodes.Copy(name="KyvenPatchMaskAlpha")
        patch_mask_alpha.setInput(0, patch_switch)
        patch_mask_alpha.setInput(1, processed_mask)
        patch_mask_alpha["from0"].setValue("rgba.red")
        patch_mask_alpha["to0"].setValue("rgba.alpha")
        patch_premult = nuke.toNode("KyvenPatchPremult")
        if patch_premult is None:
            patch_premult = nuke.nodes.Premult(name="KyvenPatchPremult")
        patch_premult.setInput(0, patch_mask_alpha)
        result_composite = nuke.toNode("KyvenResultComposite")
        if result_composite is None:
            result_composite = nuke.nodes.Merge2(name="KyvenResultComposite", operation="over")
        result_composite.setInput(0, patch_premult)
        result_composite.setInput(1, source)
        result_opaque.setInput(0, result_composite)
        result_mask_alpha = nuke.toNode("KyvenResultSourceAlpha")
        if result_mask_alpha is None:
            result_mask_alpha = nuke.nodes.Copy(name="KyvenResultSourceAlpha")
        result_mask_alpha.setInput(0, result_opaque)
        # Public alpha comes directly from Nuke's data graph. It never travels
        # through an RGB file, so ACES cannot turn 1.0 into 0.61311.
        result_mask_alpha.setInput(1, effective_mask)
        result_mask_alpha["from0"].setValue("rgba.red")
        result_mask_alpha["to0"].setValue("rgba.alpha")
        result_premult = nuke.toNode("KyvenResultPremult")
        if result_premult is None:
            result_premult = nuke.nodes.Premult(name="KyvenResultPremult")
        result_premult.setInput(0, result_mask_alpha)
        writer = nuke.toNode("KyvenInpaintModelMaskWrite")
        if writer is None:
            writer = nuke.nodes.Write(name="KyvenInpaintModelMaskWrite")
            writer["file_type"].setValue("png")
            writer["channels"].setValue("rgb")
        set_data_io(writer)
        writer.setInput(0, model_mask)
        processed_mask_read = nuke.toNode("KyvenProcessedMaskRead")
        if processed_mask_read is not None:
            set_data_io(processed_mask_read)
        source_writer = nuke.toNode("KyvenInpaintSourceWrite")
        if source_writer is not None:
            set_interchange_color_io(source_writer)
            for read_name in ("KyvenResultRead", "KyvenPatchRead"):
                color_read = nuke.toNode(read_name)
                if color_read is not None:
                    match_color_io(source_writer, color_read)
        if difference is not None:
            difference.setInput(0, result_composite)
            difference.setInput(1, source)
        if output_switch is not None:
            output_switch.setInput(0, result_opaque)
            output_switch.setInput(1, result_mask_alpha)
            output_switch.setInput(2, result_premult)
            output_switch.setInput(3, patch_switch)
            output_switch.setInput(4, difference)
            output_switch.setInput(5, source)
            output_switch.setInput(6, model_mask)
            if "preview_model_mask" in node.knobs():
                selector = "parent.preview_model_mask ? 6 : parent.output_mode"
            else:
                selector = "parent.output_mode"
            output_switch["which"].setExpression(selector)
    finally:
        node.end()


def _apply(
    node_name: str,
    job: dict[str, Any],
    output: Path,
    processed_mask: Path,
    patch: Path,
    display_name: str = "Inpaint",
    expected_job_id: str | None = None,
) -> None:
    node = _nuke().toNode(node_name)
    if node is None: return
    if expected_job_id is not None and str(node["kyven_job_id"].value()) != expected_job_id:
        return
    _set_busy(node_name, False)
    if job["status"] != "succeeded":
        _set_status(node_name, f"{display_name} failed: {_job_error_text(job)}")
        return
    _set_result(node, output)
    _set_processed_mask(node, processed_mask)
    _set_patch(node, patch)
    roi = (job.get("result") or {}).get("metadata", {}).get("processing_roi")
    suffix = f" | crop {roi['width']}x{roi['height']}" if roi else ""
    _set_status(node_name, f"{display_name} complete{suffix}")


def process_current_frame_for_node(
    node: Any,
    live: bool = False,
    payload_builder: Any = _payload,
    submit_method: str = "submit_inpaint",
    display_name: str = "Inpaint",
) -> None:
    nuke = _nuke()
    node_name = str(node.fullName())
    if bool(node["kyven_busy"].value()):
        if not live:
            _set_status(node_name, f"{display_name} is already processing this node.")
        return
    _ensure_input_order(node)
    source, mask = node.input(SOURCE_INPUT), node.input(MASK_INPUT)
    if source is None or mask is None:
        if not live:
            nuke.message("Kyven Inpaint needs Source and Mask inputs.")
        return
    if _inside(node, "KyvenInpaintModelMaskWrite") is None:
        _ensure_inpaint_preview_graph(node)
    _set_busy(node_name, True)
    frame = int(nuke.frame())
    source_path, mask_path, model_mask_path, output_path, processed_mask_path, patch_path = _paths(node, frame)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_writer = _inside(node, "KyvenInpaintSourceWrite")
    model_mask_writer = _inside(node, "KyvenInpaintModelMaskWrite")
    combined = _inside(node, "KyvenInpaintCombined") is not None
    mask_writer = None if combined else _inside(node, "KyvenInpaintMaskWrite")
    source_writer["file"].setValue(_nuke_file_path(source_path))
    model_mask_writer["file"].setValue(_nuke_file_path(model_mask_path))
    if combined:
        mask_path = source_path
    else:
        mask_writer["file"].setValue(_nuke_file_path(mask_path))
    _set_status(node.fullName(), "Exporting inpaint inputs...")
    show_progress = not live
    if show_progress:
        _start_progress(node_name, f"Kyven {display_name}", f"Exporting frame {frame}")
    try:
        nuke.execute(source_writer, frame, frame)
        nuke.execute(model_mask_writer, frame, frame)
        if mask_writer is not None: nuke.execute(mask_writer, frame, frame)
    except Exception as exc:  # noqa: BLE001
        if show_progress:
            _finish_progress(node_name)
            nuke.message(f"Inpaint export failed: {exc}")
        else:
            _set_status(node_name, f"Live inpaint export failed: {exc}")
        _set_busy(node_name, False)
        return
    payload = payload_builder(node, source, source_path, mask_path, model_mask_path, output_path, processed_mask_path, patch_path, "alpha" if combined else "luminance")
    def work() -> None:
        try:
            client = ensure_server(); job_id = getattr(client, submit_method)(payload)
            _nuke().executeInMainThread(_set_node_job_id, args=(node_name, job_id))
            last_contact = time.monotonic()
            while True:
                try:
                    job = client.job(job_id)
                    last_contact = time.monotonic()
                except NukeKyvenClientError:
                    if time.monotonic() - last_contact > 120.0:
                        raise
                    if show_progress:
                        _nuke().executeInMainThread(
                            _update_progress,
                            args=(node_name, 0, "Server busy; reconnecting..."),
                        )
                    time.sleep(0.5)
                    continue
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
            _nuke().executeInMainThread(
                _apply,
                args=(node_name, job, output_path, processed_mask_path, patch_path, display_name, job_id),
            )
        except Exception as exc:  # noqa: BLE001
            _nuke().executeInMainThread(_set_busy, args=(node_name, False))
            _nuke().executeInMainThread(_set_status, args=(node_name, f"{display_name} failed: {exc}"))
        finally:
            if show_progress:
                _nuke().executeInMainThread(_finish_progress, args=(node_name,))
    threading.Thread(target=work, name=f"kyven-{display_name.lower().replace(' ', '-')}", daemon=True).start()


def process_current_frame(node: Any | None = None, live: bool = False) -> None:
    process_current_frame_for_node(node or _nuke().thisNode(), live=live)


def process_frame_range_for_node(
    node: Any,
    payload_builder: Any = _payload,
    submit_method: str = "submit_inpaint",
    display_name: str = "Inpaint",
) -> None:
    nuke = _nuke(); _ensure_input_order(node); source, mask = node.input(SOURCE_INPUT), node.input(MASK_INPUT)
    node_name = str(node.fullName())
    if bool(node["kyven_busy"].value()):
        _set_status(node_name, f"{display_name} is already processing this node.")
        return
    first = int(node["range_first"].value()); last = int(node["range_last"].value())
    if source is None or mask is None: nuke.message("Connect Source and Mask first."); return
    if last < first: nuke.message("Range Last must be equal to or greater than Range First."); return
    if _inside(node, "KyvenInpaintModelMaskWrite") is None:
        _ensure_inpaint_preview_graph(node)
    _set_busy(node_name, True)
    root = _cache_root(node); source_pattern = root / "inpaint_source.%04d.tif"; mask_pattern = root / "inpaint_mask.%04d.png"; model_mask_pattern = root / "inpaint_model_mask.%04d.png"; output_pattern = root / "inpaint_result.%04d.png"; processed_mask_pattern = root / "inpaint_processed_mask.%04d.png"; patch_pattern = root / "inpaint_patch.%04d.png"
    source_writer = _inside(node, "KyvenInpaintSourceWrite")
    model_mask_writer = _inside(node, "KyvenInpaintModelMaskWrite")
    combined = _inside(node, "KyvenInpaintCombined") is not None
    mask_writer = None if combined else _inside(node, "KyvenInpaintMaskWrite")
    source_writer["file"].setValue(_nuke_file_path(source_pattern))
    model_mask_writer["file"].setValue(_nuke_file_path(model_mask_pattern))
    if combined:
        mask_pattern = source_pattern
    else:
        mask_writer["file"].setValue(_nuke_file_path(mask_pattern))
    _start_progress(node_name, f"Kyven {display_name}", f"Exporting frames {first}-{last}")
    try:
        nuke.execute(source_writer, first, last)
        nuke.execute(model_mask_writer, first, last)
        if mask_writer is not None: nuke.execute(mask_writer, first, last)
    except Exception as exc:  # noqa: BLE001
        _finish_progress(node_name); _set_busy(node_name, False); nuke.message(f"Inpaint export failed: {exc}"); return
    payloads = [
        payload_builder(node, source, Path(str(source_pattern) % frame), Path(str(mask_pattern) % frame), Path(str(model_mask_pattern) % frame), Path(str(output_pattern) % frame), Path(str(processed_mask_pattern) % frame), Path(str(patch_pattern) % frame), "alpha" if combined else "luminance")
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
                job_id = getattr(client, submit_method)(payload)
                _nuke().executeInMainThread(_set_node_job_id, args=(node_name, job_id))
                job = client.wait(job_id)
                if job["status"] != "succeeded":
                    error = job.get("error") or {}; raise RuntimeError(error.get("message", job["status"]))
                _nuke().executeInMainThread(_update_progress, args=(node_name, int(index * 100 / total), f"{display_name} frame {first + index - 1} ({index}/{total})"))
            cancelled = _nuke().executeInMainThreadWithResult(
                _progress_cancelled,
                args=(node_name,),
            )
            if not cancelled:
                _nuke().executeInMainThread(
                    _apply_range,
                    args=(node_name, output_pattern, processed_mask_pattern, patch_pattern, first, last, job_id, display_name),
                )
        except Exception as exc:  # noqa: BLE001
            _nuke().executeInMainThread(_set_status, args=(node_name, f"{display_name} range failed: {exc}"))
        finally:
            _nuke().executeInMainThread(_finish_progress, args=(node_name,))
            _nuke().executeInMainThread(_set_busy, args=(node_name, False))
    threading.Thread(target=work, name=f"kyven-{display_name.lower().replace(' ', '-')}-range", daemon=True).start()


def _set_node_job_id(node_name: str, job_id: str) -> None:
    node = _nuke().toNode(node_name)
    if node is not None:
        node["kyven_job_id"].setValue(job_id)


def _apply_range(
    node_name: str,
    output: Path,
    processed_mask: Path,
    patch: Path,
    first: int,
    last: int,
    expected_job_id: str,
    display_name: str,
) -> None:
    node = _nuke().toNode(node_name)
    if node is None or str(node["kyven_job_id"].value()) != expected_job_id:
        return
    _set_result(node, output, first, last)
    _set_processed_mask(node, processed_mask, first, last)
    _set_patch(node, patch, first, last)
    _set_status(node_name, f"{display_name} range ready: {first}-{last}")


def process_frame_range() -> None:
    process_frame_range_for_node(_nuke().thisNode())


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
    node = _nuke().thisNode(); _ensure_input_order(node); source = node.input(SOURCE_INPUT)
    if source is not None: node["processing_roi"].setValue([0, 0, float(source.width()), float(source.height())])


def knob_changed() -> None:
    nuke = _nuke(); node = nuke.thisNode(); knob_name = nuke.thisKnob().name()
    if knob_name in {
        "output_mode",
        "mask_channel",
        "preprocess_mask",
        "preview_model_mask",
        "invert_mask",
        "mask_threshold",
        "mask_grow",
    } and _inside(node, "KyvenInpaintModelMaskWrite") is None:
        _ensure_inpaint_preview_graph(node)
    if knob_name in {
        "inputChange",
        "mask_channel",
        "preprocess_mask",
        "invert_mask",
        "mask_threshold",
        "mask_grow",
        "crop_mode",
        "processing_roi",
    }:
        processed_mask = _inside(node, "KyvenProcessedMaskSwitch")
        if processed_mask is not None:
            processed_mask["which"].setValue(0)
    node["processing_roi"].setVisible(str(node["crop_mode"].value()) == "manual")
    node["context_padding"].setVisible(str(node["crop_mode"].value()) == "auto")
    big_lama = int(node["model"].getValue()) == 1
    if knob_name == "model" and not big_lama and "quality_mode" in node.knobs():
        node["quality_mode"].setValue(0)
    refined = (
        big_lama
        and "quality_mode" in node.knobs()
        and int(node["quality_mode"].getValue()) == 1
    )
    if "quality_mode" in node.knobs():
        node["quality_mode"].setVisible(big_lama)
    for name in ("refinement_steps", "refinement_strength", "refinement_scales", "refinement_help"):
        if name in node.knobs():
            node[name].setVisible(refined)
    if knob_name == "model":
        if int(node["model"].getValue()) == 0:
            node["model_help"].setValue("Fast: fixed 512 x 512 model input, CPU-friendly and best for Live. Use a tight Auto ROI for more detail.")
        else:
            node["model_help"].setValue("Quality: native ROI resolution. Choose Refined for slow multi-scale feature optimization on the same Big-LaMa model.")
    preprocess = bool(node["preprocess_mask"].value()) if "preprocess_mask" in node.knobs() else True
    for name in ("invert_mask", "mask_threshold", "mask_grow"):
        if name in node.knobs():
            node[name].setVisible(preprocess)
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
    if "mask_channel" in node.knobs():
        node["mask_channel"].setLabel("Mask Input Channel")
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
    _ensure_double_slider(node, "edge_softness", "Result Edge Softness (px)", 0, 32, 6)
    _place_knob_after(node, "edge_softness", "edge_color_match")
    preprocess = bool(node["preprocess_mask"].value())
    for name in ("invert_mask", "mask_threshold", "mask_grow"):
        if name in node.knobs():
            node[name].setVisible(preprocess)


def _ensure_refinement_controls(node: Any) -> None:
    """Add the optional Big-LaMa feature-refinement controls to new and upgraded nodes."""

    nuke = _nuke()
    if "quality_mode" not in node.knobs():
        quality = nuke.Enumeration_Knob("quality_mode", "Quality Mode", ["Standard", "Refined"])
        _add_knob(nuke, node, quality)
    if "refinement_steps" not in node.knobs():
        steps = nuke.Int_Knob("refinement_steps", "Refinement Steps")
        steps.setRange(1, 30)
        steps.setValue(15)
        _add_knob(nuke, node, steps)
    if "refinement_strength" not in node.knobs():
        _ensure_double_slider(node, "refinement_strength", "Refinement Strength", 0.1, 2.0, 1.0)
    if "refinement_scales" not in node.knobs():
        scales = nuke.Int_Knob("refinement_scales", "Refinement Scales")
        scales.setRange(2, 4)
        scales.setValue(3)
        _add_knob(nuke, node, scales)
    if "refinement_help" not in node.knobs():
        _add_knob(
            nuke,
            node,
            nuke.Text_Knob(
                "refinement_help",
                "",
                "Refined reuses Big-LaMa and optimizes features from coarse to native ROI detail. It is much slower; use a tight Auto ROI. No extra model is downloaded.",
            ),
        )
    previous = "model_help"
    for name in (
        "quality_mode",
        "refinement_steps",
        "refinement_strength",
        "refinement_scales",
        "refinement_help",
    ):
        _place_knob_after(node, name, previous)
        previous = name
    big_lama = int(node["model"].getValue()) == 1
    refined = big_lama and int(node["quality_mode"].getValue()) == 1
    node["quality_mode"].setVisible(big_lama)
    for name in ("refinement_steps", "refinement_strength", "refinement_scales", "refinement_help"):
        node[name].setVisible(refined)


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
    node = nuke.nodes.Group(name="KyvenInpaint"); node.setInput(SOURCE_INPUT, source); node.setInput(MASK_INPUT, mask)
    node["label"].setValue("[value kyven_status]"); node.addKnob(nuke.Tab_Knob("kyven", "Kyven Inpaint")); add_node_branding(node, nuke)
    _add_knob(nuke, node, nuke.Text_Knob("kyven_title", "", '<font size="5" color="#dce9f2"><b>KYVEN / INPAINT</b></font><br><font color="#91a3b0">LaMa | Source + Mask | API 25</font>'))
    _add_section(nuke, node, "model_section", "MODEL AND PERFORMANCE")
    _add_knob(nuke, node, nuke.Enumeration_Knob("model", "Model", list(INPAINT_MODEL_LABELS)))
    _add_knob(nuke, node, nuke.Enumeration_Knob("profile", "Memory Profile", ["low_memory", "balanced", "quality"])); node["profile"].setValue(1)
    _add_knob(nuke, node, nuke.PyScript_Knob("refresh_models", "Refresh Models", "kyven_nuke.inpaint_node.refresh_models()"))
    _add_knob(nuke, node, nuke.PyScript_Knob("open_model_manager", "Model Manager...", "kyven_nuke.model_manager.show_model_manager()"), start_line=False)
    size = nuke.Int_Knob("processing_size", "Processing Size"); size.setValue(512); size.setVisible(False); node.addKnob(size)
    _add_knob(nuke, node, nuke.Text_Knob("model_help", "", "Fast: fixed 512 x 512 model input, CPU-friendly and best for Live. Use a tight Auto ROI for more detail."))
    _ensure_refinement_controls(node)
    _add_section(nuke, node, "mask_section", "MASK")
    _add_knob(nuke, node, nuke.Enumeration_Knob("mask_channel", "Mask Input Channel", ["Alpha", "Red"]))
    preprocess = nuke.Boolean_Knob("preprocess_mask", "Preprocess Input Mask"); preprocess.setValue(True); _add_knob(nuke, node, preprocess)
    invert = nuke.Boolean_Knob("invert_mask", "Invert Input Mask"); _add_knob(nuke, node, invert)
    _ensure_inpaint_mask_sliders(node)
    _add_knob(nuke, node, nuke.Text_Knob("mask_help", "", "When Preprocess is enabled, LaMa receives Invert + Threshold + Model Grow. Preview Model Mask shows that exact mask. Result Edge Softness removes hard RGB seams without changing the model mask or output alpha."))
    _add_section(nuke, node, "roi_section", "PROCESSING ROI / MODEL CROP")
    _add_knob(nuke, node, nuke.Enumeration_Knob("crop_mode", "Crop Mode", ["auto", "manual", "full"]))
    padding = nuke.Int_Knob("context_padding", "Context Padding (px)"); padding.setRange(0, 1024); padding.setValue(128); _add_knob(nuke, node, padding)
    _add_knob(nuke, node, nuke.BBox_Knob("processing_roi", "Manual ROI"))
    _add_knob(nuke, node, nuke.PyScript_Knob("reset_roi", "Reset ROI to Source", "kyven_nuke.inpaint_node.reset_roi_to_input()"))
    _add_knob(nuke, node, nuke.Text_Knob("roi_help", "", "Auto crops to model-mask bounds plus context. The generated result is composited through the effective Inpaint mask."))
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
    _add_knob(nuke, node, nuke.Text_Knob("output_help", "", "Result is opaque RGB with an inward-softened edge. Result + Mask Alpha and Result Premult use the exact mask sent to LaMa. Generated Patch is uncomposited model RGB."))
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
        src = nuke.nodes.Input(name="Source"); src["number"].setValue(SOURCE_INPUT)
        msk = nuke.nodes.Input(name="Mask"); msk["number"].setValue(MASK_INPUT)
        result_switch = nuke.nodes.Switch(name="KyvenResultSwitch"); result_switch.setInput(0, src)
        processed_mask = nuke.nodes.Switch(name="KyvenProcessedMaskSwitch"); processed_mask.setInput(0, msk)
        result_opaque = nuke.nodes.Shuffle(name="KyvenResultOpaque"); result_opaque.setInput(0, result_switch); result_opaque["alpha"].setValue("white")
        result_alpha = nuke.nodes.Copy(name="KyvenResultSourceAlpha"); result_alpha.setInput(0, result_opaque); result_alpha.setInput(1, src); result_alpha["from0"].setValue("rgba.alpha"); result_alpha["to0"].setValue("rgba.alpha")
        result_premult = nuke.nodes.Premult(name="KyvenResultPremult"); result_premult.setInput(0, result_alpha)
        patch = nuke.nodes.Switch(name="KyvenPatchSwitch"); patch.setInput(0, src)
        difference = nuke.nodes.Merge2(name="KyvenDifference", operation="difference"); difference.setInput(0, result_switch); difference.setInput(1, src)
        output_switch = nuke.nodes.Switch(name="KyvenOutputSwitch"); output_switch.setInput(0, result_opaque); output_switch.setInput(1, result_alpha); output_switch.setInput(2, result_premult); output_switch.setInput(3, patch); output_switch.setInput(4, difference); output_switch.setInput(5, src); output_switch["which"].setExpression("parent.output_mode")
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
    isolated_cache = _cache_root(node)
    if "cache_location" in node.knobs():
        node["cache_location"].setValue(str(isolated_cache))
    _ensure_inpaint_mask_sliders(node)
    _ensure_refinement_controls(node)
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
            '<font color="#91a3b0">LaMa | Source + Mask | API 25</font>'
        )
    if "mask_help" in node.knobs():
        node["mask_help"].setValue(
            "When Preprocess is enabled, LaMa receives Invert + Threshold + Model Grow. Preview "
            "Model Mask shows that exact mask. Result Edge Softness removes hard RGB seams "
            "without changing the model mask or output alpha."
        )
    if "output_help" in node.knobs():
        node["output_help"].setValue(
            "Result is opaque RGB with an inward-softened edge. Result + Mask Alpha and Result "
            "Premult use the exact mask sent to LaMa. Generated Patch is uncomposited model RGB."
        )
    if "kyven_status" in node.knobs():
        node["kyven_status"].setValue("Inpaint UI upgraded. Cached results were preserved.")
