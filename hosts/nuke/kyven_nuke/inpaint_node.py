"""Kyven Inpaint Nuke Group and LaMa orchestration."""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any

from kyven_nuke.node import (
    _add_knob,
    _add_section,
    _cache_root,
    _ensure_live_controls,
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

INPAINT_OUTPUT_MODES = ("Result", "Patch", "Processed Mask", "Difference", "Source")


def _paths(node: Any, frame: int) -> tuple[Path, Path, Path, Path]:
    root = _cache_root(node)
    return (
        root / f"inpaint_source.{frame:04d}.tif",
        root / f"inpaint_mask.{frame:04d}.png",
        root / f"inpaint_result.{frame:04d}.png",
        root / f"inpaint_processed_mask.{frame:04d}.png",
    )


def _payload(node: Any, source: Any, source_path: Path, mask_path: Path, output_path: Path, processed_mask_path: Path, mask_channel: str) -> dict[str, Any]:
    return inpaint_payload(
        source=str(source_path.resolve()), mask=str(mask_path.resolve()), output=str(output_path.resolve()), mask_output=str(processed_mask_path.resolve()),
        model_index=int(node["model"].getValue()), profile=str(node["profile"].value()),
        image_width=int(source.width()), image_height=int(source.height()),
        crop_mode=str(node["crop_mode"].value()), roi=tuple(node["processing_roi"].value()),
        context_padding=int(node["context_padding"].value()), mask_grow=int(node["mask_grow"].value()),
        blend_grow=int(node["blend_grow"].value()) if "blend_grow" in node.knobs() else 8,
        mask_feather=float(node["mask_feather"].value()),
        edge_color_match=(
            float(node["edge_color_match"].value())
            if "edge_color_match" in node.knobs()
            else 1.0
        ),
        mask_threshold=float(node["mask_threshold"].value()),
        invert_mask=bool(node["invert_mask"].value()), mask_channel=mask_channel, processing_size=0,
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


def _apply(node_name: str, job: dict[str, Any], output: Path, processed_mask: Path) -> None:
    node = _nuke().toNode(node_name)
    if node is None: return
    _set_busy(node_name, False)
    if job["status"] != "succeeded":
        _set_status(node_name, f"Inpaint failed: {_job_error_text(job)}")
        return
    _set_result(node, output)
    _set_processed_mask(node, processed_mask)
    roi = (job.get("result") or {}).get("metadata", {}).get("processing_roi")
    suffix = f" | crop {roi['width']}x{roi['height']}" if roi else ""
    _set_status(node_name, f"Inpaint complete{suffix}")


def process_current_frame_for_node(node: Any) -> None:
    nuke = _nuke()
    source, mask = node.input(0), node.input(1)
    if source is None or mask is None:
        nuke.message("Kyven Inpaint needs Source on input 0 and Mask on input 1.")
        return
    frame = int(nuke.frame())
    source_path, mask_path, output_path, processed_mask_path = _paths(node, frame)
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
    node_name = str(node.fullName())
    _start_progress(node_name, "Kyven Inpaint", f"Exporting frame {frame}")
    try:
        nuke.execute(source_writer, frame, frame)
        if mask_writer is not None: nuke.execute(mask_writer, frame, frame)
    except Exception as exc:  # noqa: BLE001
        _finish_progress(node_name); nuke.message(f"Inpaint export failed: {exc}"); return
    payload = _payload(node, source, source_path, mask_path, output_path, processed_mask_path, "alpha" if combined else "luminance")
    _set_busy(node_name, True)

    def work() -> None:
        try:
            client = ensure_server(); job_id = client.submit_inpaint(payload)
            _nuke().executeInMainThread(lambda: node["kyven_job_id"].setValue(job_id))
            while True:
                job = client.job(job_id)
                if job["status"] in ("succeeded", "failed", "cancelled"): break
                progress = float(job.get("progress") or 0.0)
                message = str(job.get("progress_message") or "Inpainting")
                _nuke().executeInMainThread(_update_progress, args=(node_name, int(progress * 100), message))
                if _progress_cancelled(node_name): client.cancel(job_id)
                time.sleep(0.15)
            _nuke().executeInMainThread(_apply, args=(node_name, job, output_path, processed_mask_path))
        except Exception as exc:  # noqa: BLE001
            _nuke().executeInMainThread(_set_busy, args=(node_name, False))
            _nuke().executeInMainThread(_set_status, args=(node_name, f"Inpaint failed: {exc}"))
        finally:
            _nuke().executeInMainThread(_finish_progress, args=(node_name,))
    threading.Thread(target=work, name="kyven-inpaint", daemon=True).start()


def process_current_frame(node: Any | None = None, live: bool = False) -> None:
    process_current_frame_for_node(node or _nuke().thisNode())


def process_frame_range() -> None:
    nuke = _nuke(); node = nuke.thisNode(); source, mask = node.input(0), node.input(1)
    first = int(node["range_first"].value()); last = int(node["range_last"].value())
    if source is None or mask is None: nuke.message("Connect Source and Mask first."); return
    if last < first: nuke.message("Range Last must be equal to or greater than Range First."); return
    root = _cache_root(node); source_pattern = root / "inpaint_source.%04d.tif"; mask_pattern = root / "inpaint_mask.%04d.png"; output_pattern = root / "inpaint_result.%04d.png"; processed_mask_pattern = root / "inpaint_processed_mask.%04d.png"
    source_writer = _inside(node, "KyvenInpaintSourceWrite")
    combined = _inside(node, "KyvenInpaintCombined") is not None
    mask_writer = None if combined else _inside(node, "KyvenInpaintMaskWrite")
    source_writer["file"].setValue(_nuke_file_path(source_pattern))
    if combined:
        mask_pattern = source_pattern
    else:
        mask_writer["file"].setValue(_nuke_file_path(mask_pattern))
    node_name = str(node.fullName()); _start_progress(node_name, "Kyven Inpaint", f"Exporting frames {first}-{last}")
    try:
        nuke.execute(source_writer, first, last)
        if mask_writer is not None: nuke.execute(mask_writer, first, last)
    except Exception as exc:  # noqa: BLE001
        _finish_progress(node_name); nuke.message(f"Inpaint export failed: {exc}"); return
    payloads = [
        _payload(node, source, Path(str(source_pattern) % frame), Path(str(mask_pattern) % frame), Path(str(output_pattern) % frame), Path(str(processed_mask_pattern) % frame), "alpha" if combined else "luminance")
        for frame in range(first, last + 1)
    ]
    _set_busy(node_name, True)

    def work() -> None:
        try:
            client = ensure_server(); total = len(payloads)
            for index, payload in enumerate(payloads, start=1):
                if _progress_cancelled(node_name): break
                job_id = client.submit_inpaint(payload)
                _nuke().executeInMainThread(lambda value=job_id: node["kyven_job_id"].setValue(value))
                job = client.wait(job_id)
                if job["status"] != "succeeded":
                    error = job.get("error") or {}; raise RuntimeError(error.get("message", job["status"]))
                _nuke().executeInMainThread(_update_progress, args=(node_name, int(index * 100 / total), f"Inpaint frame {first + index - 1} ({index}/{total})"))
            if not _progress_cancelled(node_name):
                _nuke().executeInMainThread(_set_result, args=(node, output_pattern, first, last))
                _nuke().executeInMainThread(_set_processed_mask, args=(node, processed_mask_pattern, first, last))
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
    node["label"].setValue("[value kyven_status]"); node.addKnob(nuke.Tab_Knob("kyven", "Kyven Inpaint"))
    _add_knob(nuke, node, nuke.Text_Knob("kyven_title", "", '<font size="5" color="#dce9f2"><b>KYVEN / INPAINT</b></font><br><font color="#91a3b0">LaMa | Source + Mask | API 15</font>'))
    _add_section(nuke, node, "model_section", "MODEL AND PERFORMANCE")
    _add_knob(nuke, node, nuke.Enumeration_Knob("model", "Model", list(INPAINT_MODEL_LABELS)))
    _add_knob(nuke, node, nuke.Enumeration_Knob("profile", "Memory Profile", ["low_memory", "balanced", "quality"])); node["profile"].setValue(1)
    _add_knob(nuke, node, nuke.PyScript_Knob("refresh_models", "Refresh Models", "kyven_nuke.inpaint_node.refresh_models()"))
    _add_knob(nuke, node, nuke.PyScript_Knob("open_model_manager", "Model Manager...", "kyven_nuke.model_manager.show_model_manager()"), start_line=False)
    size = nuke.Int_Knob("processing_size", "Processing Size"); size.setValue(512); size.setVisible(False); node.addKnob(size)
    _add_knob(nuke, node, nuke.Text_Knob("model_help", "", "Fast: fixed 512 x 512 model input, CPU-friendly and best for Live. Use a tight Auto ROI for more detail."))
    _add_section(nuke, node, "mask_section", "MASK")
    _add_knob(nuke, node, nuke.Enumeration_Knob("mask_channel", "Input 1 Channel", ["Alpha", "Red"]))
    invert = nuke.Boolean_Knob("invert_mask", "Invert Input Mask"); _add_knob(nuke, node, invert)
    threshold = nuke.Double_Knob("mask_threshold", "Threshold"); threshold.setRange(0, 1); threshold.setValue(0.5); _add_knob(nuke, node, threshold)
    grow = nuke.Int_Knob("mask_grow", "Model Mask Grow (px)"); grow.setRange(-128, 128); grow.setValue(12); _add_knob(nuke, node, grow)
    blend_grow = nuke.Int_Knob("blend_grow", "Blend Mask Grow (px)"); blend_grow.setRange(-128, 128); blend_grow.setValue(8); _add_knob(nuke, node, blend_grow)
    feather = nuke.Double_Knob("mask_feather", "Blend Feather (px)"); feather.setRange(0, 64); feather.setValue(4); _add_knob(nuke, node, feather)
    color_match = nuke.Double_Knob("edge_color_match", "Edge Color Match"); color_match.setRange(0, 1); color_match.setValue(1); _add_knob(nuke, node, color_match)
    _add_knob(nuke, node, nuke.Text_Knob("mask_help", "", "Model Grow removes the old antialiased edge. Blend Grow/Feather hide the seam; Edge Color Match aligns the patch to nearby clean pixels."))
    _add_section(nuke, node, "roi_section", "PROCESSING ROI / MODEL CROP")
    _add_knob(nuke, node, nuke.Enumeration_Knob("crop_mode", "Crop Mode", ["auto", "manual", "full"]))
    padding = nuke.Int_Knob("context_padding", "Context Padding (px)"); padding.setRange(0, 1024); padding.setValue(128); _add_knob(nuke, node, padding)
    _add_knob(nuke, node, nuke.BBox_Knob("processing_roi", "Manual ROI"))
    _add_knob(nuke, node, nuke.PyScript_Knob("reset_roi", "Reset ROI to Source", "kyven_nuke.inpaint_node.reset_roi_to_input()"))
    _add_knob(nuke, node, nuke.Text_Knob("roi_help", "", "Auto crops to mask bounds plus context. The result is pasted only through the grown / feathered mask."))
    _add_section(nuke, node, "processing_section", "INDEPENDENT FRAME PROCESSING")
    _ensure_live_controls(node, "inpaint")
    _add_knob(nuke, node, nuke.PyScript_Knob("process_frame", "Process Current Frame", "kyven_nuke.inpaint_node.process_current_frame()"))
    _add_knob(nuke, node, nuke.PyScript_Knob("cancel", "Cancel", "kyven_nuke.inpaint_node.cancel_current_job()"), start_line=False)
    first = nuke.Int_Knob("range_first", "Range First"); first.setValue(int(nuke.root().firstFrame())); _add_knob(nuke, node, first)
    last = nuke.Int_Knob("range_last", "Range Last"); last.setValue(int(nuke.root().lastFrame())); _add_knob(nuke, node, last, start_line=False)
    _add_knob(nuke, node, nuke.PyScript_Knob("process_range", "Process Frame Range", "kyven_nuke.inpaint_node.process_frame_range()"))
    _add_section(nuke, node, "output_section", "OUTPUT")
    _add_knob(nuke, node, nuke.Enumeration_Knob("output_mode", "Output", list(INPAINT_OUTPUT_MODES)))
    _add_knob(nuke, node, nuke.Text_Knob("output_help", "", "Result keeps Source alpha. Patch is premultiplied by the processed mask. Difference reveals changed pixels."))
    uid = nuke.String_Knob("kyven_uuid", "UUID"); uid.setValue(uuid.uuid4().hex); uid.setVisible(False); node.addKnob(uid)
    _add_section(nuke, node, "cache_section", "CACHE")
    folder = nuke.String_Knob("cache_location", "Cache Folder"); folder.setFlag(nuke.READ_ONLY); folder.setValue(str(_cache_root(node))); _add_knob(nuke, node, folder)
    _add_knob(nuke, node, nuke.PyScript_Knob("create_result_read", "Create Result Read", "kyven_nuke.inpaint_node.create_read_from_current_result()"))
    _add_knob(nuke, node, nuke.PyScript_Knob("delete_node_cache", "Delete Node Cache", "kyven_nuke.inpaint_node.delete_this_node_cache()"), start_line=False)
    _add_knob(nuke, node, nuke.PyScript_Knob("delete_all_cache", "Delete All Kyven Cache", "kyven_nuke.node.delete_all_cache()"))
    job = nuke.String_Knob("kyven_job_id", "Job ID"); job.setVisible(False); node.addKnob(job)
    _add_section(nuke, node, "status_section", "STATUS"); status = nuke.String_Knob("kyven_status", "Status"); status.setFlag(nuke.READ_ONLY); status.setValue("Ready"); _add_knob(nuke, node, status)
    node["knobChanged"].setValue("kyven_nuke.inpaint_node.knob_changed()")
    node.begin()
    try:
        src = nuke.nodes.Input(name="Source"); src["number"].setValue(0)
        msk = nuke.nodes.Input(name="Mask"); msk["number"].setValue(1)
        result_switch = nuke.nodes.Switch(name="KyvenResultSwitch"); result_switch.setInput(0, src)
        processed_mask = nuke.nodes.Switch(name="KyvenProcessedMaskSwitch"); processed_mask.setInput(0, msk)
        result_alpha = nuke.nodes.Copy(name="KyvenResultSourceAlpha"); result_alpha.setInput(0, result_switch); result_alpha.setInput(1, src); result_alpha["from0"].setValue("rgba.alpha"); result_alpha["to0"].setValue("rgba.alpha")
        patch_alpha = nuke.nodes.Copy(name="KyvenPatchAlpha"); patch_alpha.setInput(0, result_switch); patch_alpha.setInput(1, processed_mask); patch_alpha["from0"].setValue("rgba.red"); patch_alpha["to0"].setValue("rgba.alpha")
        patch = nuke.nodes.Premult(name="KyvenPatch"); patch.setInput(0, patch_alpha)
        difference = nuke.nodes.Merge2(name="KyvenDifference", operation="difference"); difference.setInput(0, result_switch); difference.setInput(1, src)
        output_switch = nuke.nodes.Switch(name="KyvenOutputSwitch"); output_switch.setInput(0, result_alpha); output_switch.setInput(1, patch); output_switch.setInput(2, processed_mask); output_switch.setInput(3, difference); output_switch.setInput(4, src); output_switch["which"].setExpression("parent.output_mode")
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
    reset = source
    if reset is not None: node["processing_roi"].setValue([0, 0, float(reset.width()), float(reset.height())])
    node["processing_roi"].setVisible(False)
    _restyle_inpaint_cache(node)
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
    _restyle_inpaint_cache(node)
    if "kyven_status" in node.knobs():
        node["kyven_status"].setValue("Inpaint UI upgraded. Cached results were preserved.")
