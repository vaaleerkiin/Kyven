"""Kyven Refine Group node and asynchronous ViTMatte orchestration."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any

from kyven_nuke.node import (
    OUTPUT_MODES,
    _add_knob,
    _add_section,
    _cache_root,
    _ensure_cache_controls,
    _ensure_live_controls,
    _ensure_output_controls,
    _inside,
    _nuke,
    _nuke_file_path,
    _path_for_frame,
    _set_busy,
    _set_matte_read,
    _set_status,
)
from kyven_nuke.payload import REFINE_MODEL_LABELS, refine_payload
from kyven_nuke.runtime import ensure_server

_range_cancellations: set[str] = set()
_range_cancel_lock = threading.RLock()


def _cache_paths(node: Any, frame: int) -> tuple[Path, Path, Path]:
    root = _cache_root(node)
    return (
        root / f"refine_source.{frame:04d}.png",
        root / f"refine_mask.{frame:04d}.png",
        root / f"refined_matte.{frame:04d}.png",
    )


def _cache_patterns(node: Any) -> tuple[Path, Path, Path]:
    root = _cache_root(node)
    return (
        root / "refine_source.%04d.png",
        root / "refine_mask.%04d.png",
        root / "refined_matte.%04d.png",
    )


def _tile_size(node: Any) -> int:
    profile = str(node["profile"].value())
    automatic = {"low_memory": 512, "balanced": 1024, "quality": 0}[profile]
    custom = int(node["tile_size"].value())
    return custom if custom else automatic


def _payload(
    node: Any,
    source: Any,
    source_path: Path,
    mask_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    return refine_payload(
        source=str(source_path.resolve()),
        mask=str(mask_path.resolve()),
        output=str(output_path.resolve()),
        model_index=int(node["model"].getValue()),
        profile=str(node["profile"].value()),
        image_height=int(source.height()),
        roi_enabled=bool(node["roi_enabled"].value()),
        roi=tuple(node["processing_roi"].value()),
        generate_trimap=bool(node["generate_trimap"].value()),
        foreground_radius=int(node["foreground_radius"].value()),
        background_radius=int(node["background_radius"].value()),
        tile_size=_tile_size(node),
        tile_overlap=int(node["tile_overlap"].value()),
    )


def _apply_result(node_name: str, job: dict[str, Any]) -> None:
    node = _nuke().toNode(node_name)
    if node is None:
        return
    if job["status"] != "succeeded":
        error = job.get("error") or {}
        node["kyven_status"].setValue(f"Refine failed: {error.get('message', job['status'])}")
        return
    result = job["result"]
    _set_matte_read(node, _nuke_file_path(Path(result["output"])))
    metadata = result.get("metadata") or {}
    trimap = metadata.get("trimap") or {}
    mode = "auto trimap" if trimap.get("generated") else "input trimap"
    roi = metadata.get("processing_roi")
    roi_text = f" | ROI {roi['width']}x{roi['height']}" if roi else ""
    node["kyven_status"].setValue(
        f"Refined - {mode} | {int(metadata.get('tiles', 1))} tile(s){roi_text}"
    )


def _submit_and_wait(node_name: str, payload: dict[str, Any]) -> None:
    nuke = _nuke()
    try:
        client = ensure_server()
        job_id = client.submit_refine(payload)
        nuke.executeInMainThread(_set_job_id, args=(node_name, job_id))
        job = client.wait(job_id, timeout_seconds=1800.0)
        nuke.executeInMainThread(_apply_result, args=(node_name, job))
    except Exception as exc:  # noqa: BLE001
        nuke.executeInMainThread(_set_status, args=(node_name, f"Refine failed: {exc}"))
    finally:
        nuke.executeInMainThread(_set_busy, args=(node_name, False))


def _set_job_id(node_name: str, job_id: str) -> None:
    node = _nuke().toNode(node_name)
    if node is not None:
        node["kyven_job_id"].setValue(job_id)
        node["kyven_status"].setValue("ViTMatte refining...")


def _export(node: Any, frame: int, source_path: Path, mask_path: Path) -> bool:
    nuke = _nuke()
    source_writer = _inside(node, "KyvenRefineSourceWrite")
    mask_writer = _inside(node, "KyvenRefineMaskWrite")
    source_writer["file"].setValue(_nuke_file_path(source_path))
    mask_writer["file"].setValue(_nuke_file_path(mask_path))
    try:
        nuke.execute(source_writer, frame, frame)
        nuke.execute(mask_writer, frame, frame)
    except Exception as exc:  # noqa: BLE001
        node["kyven_status"].setValue(f"Refine input export failed: {exc}")
        return False
    return source_path.is_file() and mask_path.is_file()


def process_current_frame(node: Any | None = None, live: bool = False) -> None:
    nuke = _nuke()
    node = node or nuke.thisNode()
    source = node.input(0)
    mask = node.input(1)
    if source is None or mask is None:
        if not live:
            nuke.message("Kyven Refine requires Source and Mask/Trimap inputs.")
        return
    if bool(node["kyven_busy"].value()):
        return
    frame = int(nuke.frame())
    source_path, mask_path, output_path = _cache_paths(node, frame)
    node["kyven_status"].setValue("Exporting source and mask...")
    if not _export(node, frame, source_path, mask_path):
        return
    payload = _payload(node, source, source_path, mask_path, output_path)
    node["kyven_busy"].setValue(True)
    node["kyven_live_frame"].setValue(frame)
    threading.Thread(
        target=_submit_and_wait,
        args=(node.fullName(), payload),
        name="kyven-refine-submit",
        daemon=True,
    ).start()


def _apply_range(node_name: str, output: str, first: int, last: int) -> None:
    node = _nuke().toNode(node_name)
    if node is not None:
        _set_matte_read(node, output, first, last)
        node["kyven_status"].setValue(f"Refined range ready: {first}-{last}")


def _range_worker(
    node_name: str,
    payloads: list[tuple[int, dict[str, Any]]],
    output_pattern: str,
    first: int,
    last: int,
) -> None:
    nuke = _nuke()
    try:
        client = ensure_server()
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
            job = client.wait(job_id, timeout_seconds=1800.0)
            if job["status"] != "succeeded":
                error = job.get("error") or {}
                raise RuntimeError(error.get("message", job["status"]))
        nuke.executeInMainThread(
            _apply_range,
            args=(node_name, output_pattern, first, last),
        )
    except Exception as exc:  # noqa: BLE001
        nuke.executeInMainThread(_set_status, args=(node_name, f"Refine range failed: {exc}"))
    finally:
        with _range_cancel_lock:
            _range_cancellations.discard(node_name)
        nuke.executeInMainThread(_set_busy, args=(node_name, False))


def process_frame_range() -> None:
    nuke = _nuke()
    node = nuke.thisNode()
    source = node.input(0)
    if source is None or node.input(1) is None:
        nuke.message("Kyven Refine requires Source and Mask/Trimap inputs.")
        return
    first = int(node["range_first"].value())
    last = int(node["range_last"].value())
    if last < first:
        nuke.message("Range Last must be greater than or equal to Range First.")
        return
    source_pattern, mask_pattern, output_pattern = _cache_patterns(node)
    source_writer = _inside(node, "KyvenRefineSourceWrite")
    mask_writer = _inside(node, "KyvenRefineMaskWrite")
    source_writer["file"].setValue(_nuke_file_path(source_pattern))
    mask_writer["file"].setValue(_nuke_file_path(mask_pattern))
    node["kyven_status"].setValue(f"Exporting refine inputs {first}-{last}...")
    try:
        nuke.execute(source_writer, first, last)
        nuke.execute(mask_writer, first, last)
    except Exception as exc:  # noqa: BLE001
        node["kyven_status"].setValue(f"Refine range export failed: {exc}")
        return
    payloads = []
    for frame in range(first, last + 1):
        source_path = _path_for_frame(source_pattern, frame)
        mask_path = _path_for_frame(mask_pattern, frame)
        if not source_path.is_file() or not mask_path.is_file():
            node["kyven_status"].setValue(f"Missing exported refine input at frame {frame}.")
            return
        output_path = _path_for_frame(output_pattern, frame)
        payloads.append((frame, _payload(node, source, source_path, mask_path, output_path)))
    node_name = node.fullName()
    with _range_cancel_lock:
        _range_cancellations.discard(node_name)
    node["kyven_busy"].setValue(True)
    threading.Thread(
        target=_range_worker,
        args=(node_name, payloads, _nuke_file_path(output_pattern), first, last),
        name="kyven-refine-range",
        daemon=True,
    ).start()


def cancel_current_job() -> None:
    nuke = _nuke()
    node = nuke.thisNode()
    node_name = node.fullName()
    with _range_cancel_lock:
        _range_cancellations.add(node_name)
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
    source = node.input(0)
    if source is None:
        nuke.message("Connect a Source first.")
        return
    node["processing_roi"].setValue([0.0, 0.0, float(source.width()), float(source.height())])
    node["kyven_status"].setValue("Processing ROI reset to Source size.")


def knob_changed() -> None:
    nuke = _nuke()
    node = nuke.thisNode()
    name = nuke.thisKnob().name()
    generated = bool(node["generate_trimap"].value())
    node["foreground_radius"].setVisible(generated)
    node["background_radius"].setVisible(generated)
    node["processing_roi"].setVisible(bool(node["roi_enabled"].value()))
    if name not in {"kyven_status", "kyven_job_id", "kyven_live_frame", "kyven_busy"}:
        node["kyven_live_frame"].setValue(-2147483647)


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


def create_refine_node() -> Any:
    nuke = _nuke()
    selected = nuke.selectedNode() if nuke.selectedNodes() else None
    node = nuke.nodes.Group(name="KyvenRefine")
    node.setInput(0, selected)
    node["label"].setValue("[value kyven_status]")
    node.addKnob(nuke.Tab_Knob("kyven", "Kyven Refine"))
    _add_knob(
        nuke,
        node,
        nuke.Text_Knob(
            "kyven_title",
            "",
            '<font size="5" color="#dce9f2"><b>KYVEN / REFINE</b></font><br>'
            '<font color="#91a3b0">ViTMatte | Source + Mask | API 5</font>',
        ),
    )
    _add_section(nuke, node, "model_section", "MODEL AND PERFORMANCE")
    _add_knob(nuke, node, nuke.Enumeration_Knob("model", "Model", list(REFINE_MODEL_LABELS)))
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
        start_line=False,
    )
    tile_size = nuke.Int_Knob("tile_size", "Custom Tile Size (0 = Auto)")
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
        nuke.Enumeration_Knob("mask_channel", "Input 1 Channel", ["Alpha", "Red"]),
    )
    generate = nuke.Boolean_Knob("generate_trimap", "Generate Trimap from Mask")
    generate.setValue(True)
    _add_knob(nuke, node, generate)
    foreground = nuke.Int_Knob("foreground_radius", "Foreground Erosion (px)")
    foreground.setRange(0, 49)
    foreground.setValue(10)
    _add_knob(nuke, node, foreground)
    background = nuke.Int_Knob("background_radius", "Background Dilation (px)")
    background.setRange(0, 49)
    background.setValue(15)
    _add_knob(nuke, node, background, start_line=False)
    _add_knob(
        nuke,
        node,
        nuke.Text_Knob(
            "trimap_help",
            "",
            "On: Input 1 is a coarse mask and Kyven creates the trimap. "
            "Off: Input 1 must already be black / gray / white trimap.",
        ),
    )

    _add_section(nuke, node, "roi_section", "PROCESSING ROI / MODEL CROP")
    _add_knob(nuke, node, nuke.Boolean_Knob("roi_enabled", "Enable Processing ROI"))
    _add_knob(nuke, node, nuke.BBox_Knob("processing_roi", "Processing ROI"))
    _add_knob(
        nuke,
        node,
        nuke.PyScript_Knob(
            "reset_roi", "Reset ROI to Source", "kyven_nuke.refine_node.reset_roi_to_input()"
        ),
    )

    _add_section(nuke, node, "processing_section", "PROCESSING")
    _ensure_live_controls(node, "refine")
    _add_knob(
        nuke,
        node,
        nuke.Text_Knob(
            "live_help",
            "",
            "Live processes the current frame after the timeline changes; GPU work stays asynchronous.",
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
            "process_range", "Process Frame Range", "kyven_nuke.refine_node.process_frame_range()"
        ),
    )

    _add_section(nuke, node, "output_section", "OUTPUT")
    _add_knob(nuke, node, nuke.Enumeration_Knob("output_mode", "Output", list(OUTPUT_MODES)))
    node["output_mode"].setValue(1)
    internal_id = nuke.String_Knob("kyven_uuid", "UUID")
    internal_id.setValue(uuid.uuid4().hex)
    internal_id.setVisible(False)
    node.addKnob(internal_id)
    _ensure_cache_controls(node)
    node["create_matte_read"].setCommand("kyven_nuke.node.create_read_from_current_matte()")
    _add_section(nuke, node, "status_section", "STATUS")
    status = nuke.String_Knob("kyven_status", "Status")
    status.setFlag(nuke.READ_ONLY)
    status.setValue("Connect Source and Mask, then process a frame.")
    _add_knob(nuke, node, status)
    job_id = nuke.String_Knob("kyven_job_id", "Job ID")
    job_id.setVisible(False)
    node.addKnob(job_id)
    node["knobChanged"].setValue("kyven_nuke.refine_node.knob_changed()")

    node.begin()
    try:
        source = nuke.nodes.Input(name="Source")
        source["number"].setValue(0)
        mask = nuke.nodes.Input(name="MaskOrTrimap")
        mask["number"].setValue(1)
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
        source_writer["file_type"].setValue("png")
        source_writer["channels"].setValue("rgb")
        mask_writer = nuke.nodes.Write(name="KyvenRefineMaskWrite")
        mask_writer.setInput(0, mask_channel)
        mask_writer["file_type"].setValue("png")
        mask_writer["channels"].setValue("rgb")
    finally:
        node.end()
    _ensure_output_controls(node)
    reset_roi_to_input_for_node(node)
    knob_changed_for_node(node)
    return node


def reset_roi_to_input_for_node(node: Any) -> None:
    source = node.input(0)
    if source is not None:
        node["processing_roi"].setValue(
            [0.0, 0.0, float(source.width()), float(source.height())]
        )


def knob_changed_for_node(node: Any) -> None:
    generated = bool(node["generate_trimap"].value())
    node["foreground_radius"].setVisible(generated)
    node["background_radius"].setVisible(generated)
    node["processing_roi"].setVisible(bool(node["roi_enabled"].value()))
