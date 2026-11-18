"""Explicit input snapshots. Nuke export belongs to Cache Frames, never Process."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from kyven import input_index
from kyven.hashing import cached_file_sha256


def _receipt(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".input.json")


def _state(node: Any, frame: int) -> dict:
    # Inpaint exports a processed model mask; these edits invalidate that input snapshot.
    names = ("mask_grow", "mask_threshold", "invert_mask", "preprocess_mask", "input_colorspace",
             "in_colorspace", "generate_trimap")
    from kyven_nuke.node import _knob_value_at
    state = {name: _knob_value_at(node[name], frame) for name in names if name in node.knobs()}
    if "kyven_kind" in node.knobs() and node["kyven_kind"].value() in {"refine", "inpaint"}:
        state["mask_input_contract"] = "alpha-only-positive-v1"
        if node["kyven_kind"].value() == "refine":
            state["mask_input_contract"] = "alpha-only-cattery-average-v2"
    if "kyven_kind" in node.knobs() and node["kyven_kind"].value() == "dlss":
        from kyven_nuke.dlss_inputs import snapshot_state
        state["dlss_input"] = snapshot_state(node, frame)
    return state


def ready(node: Any, path: Path, frame: int | None = None) -> bool:
    try:
        receipt = input_index.read(path)
        if receipt is None:
            receipt = json.loads(_receipt(path).read_text(encoding="utf-8"))
        if not isinstance(receipt, dict) or type(receipt.get("frame")) is not int:
            return False
        recorded_frame = receipt["frame"]
        return ((frame is None or frame == recorded_frame)
                and receipt.get("state") == _state(node, recorded_frame)
                and path.stat().st_size > 0 and receipt.get("sha256") == cached_file_sha256(path))
    except (OSError, ValueError, TypeError):
        return False



def invalidate(path: Path) -> None:
    input_index.forget(path)
    _receipt(path).unlink(missing_ok=True)


def export_input(nuke: Any, node: Any, writer: Any, frame: int, path: Path,
                 refresh: bool = False) -> bool:
    if not refresh and not path.exists() and path.parent.name == "tracking_pool":
        # Adopt verified JPEGs from the previous per-range layout without rendering again.
        for legacy in path.parent.parent.glob(f"*_*/{path.name}"):
            if ready(node, legacy, frame):
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(legacy, path)
                except OSError:
                    from kyven.cache_layout import copy_if_changed
                    copy_if_changed(legacy, path)
                input_index.write(path, {"sha256": cached_file_sha256(path),
                                         "state": _state(node, frame), "frame": frame})
                break
    if not refresh and ready(node, path, frame):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    writer["file"].setValue(path.as_posix())
    if hasattr(writer, "knobs"):
        file_type = {".jpg": "jpeg", ".jpeg": "jpeg", ".tif": "tiff", ".tiff": "tiff", ".exr": "exr"}.get(path.suffix.lower())
        if file_type and "file_type" in writer.knobs():
            writer["file_type"].setValue(file_type)
        if path.suffix.lower() in {".tif", ".tiff"} and "compression" in writer.knobs():
            writer["compression"].setValue(0)
        if path.suffix.lower() in {".jpg", ".jpeg"} and "_jpeg_quality" in writer.knobs():
            writer["_jpeg_quality"].setValue(1.0)
    if path.suffix.lower() == ".png" and hasattr(writer, "knobs") and "compression" in writer.knobs():
        writer["compression"].setValue(0)
    # Invalidate first: a failed refresh must never leave an apparently valid old receipt.
    invalidate(path)
    from kyven_nuke import dlss_node
    guarded = dlss_node._suppress_node_callbacks(
        lambda current: nuke.execute(writer, frame, frame))
    guarded(node)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Input export did not create frame {frame}: {path.name}")
    payload = {"sha256": cached_file_sha256(path), "state": _state(node, frame), "frame": frame}
    input_index.write(path, payload)
    return True


def use_input(nuke: Any, node: Any, writer: Any, frame: int) -> None:
    pattern = str(writer["file"].value())
    path = Path(pattern % frame if "%" in pattern else pattern)
    if not ready(node, path, frame):
        raise RuntimeError(f"Input cache missing or outdated at frame {frame} ({path.name}). "
                           "Click Cache Frames; use Refresh Frames after upstream edits.")



def current_input(nuke: Any, node: Any, writer: Any, frame: int) -> None:
    """Prepare a missing current-frame snapshot; ready snapshots are reused."""
    pattern = str(writer["file"].value())
    path = Path(pattern % frame if "%" in pattern else pattern)
    export_input(nuke, node, writer, frame, path)
    use_input(nuke, node, writer, frame)


def ensure_controls(node: Any) -> None:
    from kyven_nuke.node import _nuke, _place_knob_after
    nuke = _nuke()
    definitions = [
        ("Enumeration_Knob", "input_cache_scope", "Input Frames", ["Frame Range", "Current Frame"]),
        ("PyScript_Knob", "cache_frames", "Cache Frames", None),
        ("PyScript_Knob", "refresh_frames", "Refresh Frames", None),
    ]
    for kind, name, label, choices in definitions:
        if name not in node.knobs():
            knob = getattr(nuke, kind)(name, label, choices) if choices else getattr(nuke, kind)(name, label)
            node.addKnob(knob)
        if choices is None:
            refresh = name == "refresh_frames"
            node[name].setCommand("__import__('kyven_nuke.input_cache', fromlist=['cache_frames'])"
                                  f".cache_frames(refresh={refresh})")
        node[name].setTooltip("Prepare a snapshot of input frames without running models. "
                              "Cache adds missing frames; Refresh updates upstream edits. "
                              "Process fills missing inputs automatically; ready frames are reused.")
    for name in ("process_frame", "process_range", "live_mode"):
        if name in node.knobs():
            node[name].setTooltip("Process fills missing input snapshots automatically. Ready frames are reused. "
                                  "Refresh Frames after upstream edits.")
    anchor = "range_last" if "range_last" in node.knobs() else "last_frame"
    for name in ("input_cache_scope", "cache_frames", "refresh_frames"):
        _place_knob_after(node, name, anchor)
        anchor = name
    from kyven_nuke.tooltips import apply_tooltips
    apply_tooltips(node)


def _specs(node: Any, frame: int, first: int, last: int) -> list:
    from kyven_nuke import node as segment
    kind = (str(node["kyven_kind"].value()) if "kyven_kind" in node.knobs()
            else "inpaint" if "mask_grow" in node.knobs()
            else "refine" if "generate_trimap" in node.knobs() else "segment")
    inside = segment._inside
    if kind == "refine":
        from kyven_nuke import refine_node as adapter
        adapter._ensure_input_order(node)
        paths = adapter._cache_paths(node, frame)
        writers = [inside(node, "KyvenRefineSourceWrite"), inside(node, "KyvenRefineMaskWrite")]
        adapter._configure_export_writers(*writers)
        return list(zip(writers, paths[:2]))
    if kind == "inpaint":
        from kyven_nuke import inpaint_node as adapter
        adapter._ensure_input_order(node)
        if inside(node, "KyvenInpaintModelMaskWrite") is None:
            adapter._ensure_inpaint_preview_graph(node)
        adapter._configure_color_io(node)
        source, mask, model, *_ = adapter._paths(node, frame)
        specs = [(inside(node, "KyvenInpaintSourceWrite"), source),
                 (inside(node, "KyvenInpaintModelMaskWrite"), model)]
        if inside(node, "KyvenInpaintCombined") is None:
            specs.append((inside(node, "KyvenInpaintMaskWrite"), mask))
        return specs
    if kind == "dlss":
        from kyven_nuke import dlss_node as adapter
        root = adapter._dlss_cache_root(node) / "_kyven_work" / "inputs"
        settings = adapter._request_settings(node)
        adapter._validate_inputs(node, settings)
        return [(adapter._packed_input_writer(node, settings), root / f"input.{frame:04d}.exr")]
    source = segment._cache_paths(node, frame)[0]
    return [(inside(node, "KyvenSourceWrite"), source)]



def prepare_range(nuke: Any, node: Any, first: int, last: int,
                  refresh: bool = False, cancelled=None) -> bool:
    """Shared preparation for the button and Process; never re-export ready inputs."""
    from kyven_nuke.node import _set_status
    if last < first or last - first >= 100000:
        raise RuntimeError("Input cache range must contain 1 to 100000 frames.")
    name = str(node.fullName())
    with input_index.batch():
        pending = []
        for frame in range(first, last + 1):
            if cancelled is not None and cancelled():
                _set_status(name, "Input caching cancelled; completed frames retained.")
                return False
            for writer, path in _specs(node, frame, first, last):
                if refresh or not ready(node, path, frame):
                    pending.append((frame, writer, path))
        if not pending:
            return True
        progress = nuke.ProgressTask("Kyven Cache Frames")
        original_frame = int(nuke.frame())
        try:
            for index, (frame, writer, path) in enumerate(pending):
                if progress.isCancelled() or (cancelled is not None and cancelled()):
                    _set_status(name, "Input caching cancelled; completed frames retained.")
                    return False
                progress.setMessage(f"Caching frame {frame}: {path.name}")
                _set_status(name, f"Caching missing input {frame}: {path.name}")
                previous = writer["file"].value()
                try:
                    export_input(nuke, node, writer, frame, path, refresh)
                finally:
                    writer["file"].setValue(previous)
                progress.setProgress(round((index + 1) * 100 / len(pending)))
            return True
        finally:
            del progress
            from kyven_nuke import dlss_node
            dlss_node._suppress_node_callbacks(lambda current: nuke.frame(original_frame))(node)


def cache_frames(refresh: bool = False) -> None:
    from kyven_nuke.node import _begin_node_operation, _nuke, _set_busy, _set_status
    nuke = _nuke()
    node = nuke.thisNode()
    name = str(node.fullName())
    if not _begin_node_operation(node):
        return
    try:
        if str(node["input_cache_scope"].value()) == "Current Frame":
            first = last = int(nuke.frame())
        else:
            first_name, last_name = (("range_first", "range_last") if "range_first" in node.knobs()
                                     else ("first_frame", "last_frame"))
            first, last = int(node[first_name].value()), int(node[last_name].value())
        if prepare_range(nuke, node, first, last, refresh):
            _set_status(name, f"Input cache ready {first}-{last}. Process is ready.")
    except Exception as exc:  # noqa: BLE001 - Nuke callback boundary
        _set_status(name, f"Input cache failed: {exc}")
        if "Choose Cache Folder" in str(exc):
            nuke.message(str(exc))
    finally:
        _set_busy(name, False)
