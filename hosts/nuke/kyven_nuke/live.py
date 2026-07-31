"""Debounced current-frame processing callbacks for Kyven Nuke nodes."""

from __future__ import annotations

import threading
import time

_last_scan = 0.0
_request_lock = threading.RLock()
_request_revisions: dict[str, int] = {}


def affects_live_result(knob_name: str, kind: str) -> bool:
    """Return whether a Group knob changes inference for the current frame."""

    if kind == "refine":
        return knob_name in {
            "model",
            "profile",
            "roi_enabled",
            "processing_roi",
            "tile_size",
            "tile_overlap",
        }
    return (
        knob_name.startswith(("positive_point", "negative_point"))
        or knob_name
        in {
            "model",
            "profile",
            "positive_enabled",
            "negative_enabled",
            "box_enabled",
            "prompt_box",
        }
    )


def request_live_update(node: object, delay_seconds: float = 0.25) -> None:
    """Debounce Viewer-handle and settings edits into one current-frame inference."""

    if not bool(node["live_mode"].value()):
        return
    node_name = str(node.fullName())
    with _request_lock:
        revision = _request_revisions.get(node_name, 0) + 1
        _request_revisions[node_name] = revision
    timer = threading.Timer(delay_seconds, _dispatch_requested_update, args=(node_name, revision))
    timer.daemon = True
    timer.start()


def _dispatch_requested_update(node_name: str, revision: int) -> None:
    import nuke

    nuke.executeInMainThread(_run_requested_update, args=(node_name, revision))


def _run_requested_update(node_name: str, revision: int) -> None:
    import nuke

    with _request_lock:
        if _request_revisions.get(node_name) != revision:
            return
    node = nuke.toNode(node_name)
    if node is None or not bool(node["live_mode"].value()):
        return
    if int(node["kyven_live_frame"].value()) == int(nuke.frame()):
        return
    if nuke.executing() or bool(node["kyven_busy"].value()):
        timer = threading.Timer(0.2, _dispatch_requested_update, args=(node_name, revision))
        timer.daemon = True
        timer.start()
        return
    kind = str(node["kyven_kind"].value()) if "kyven_kind" in node.knobs() else "segment"
    if kind == "refine":
        if node.input(0) is None or node.input(1) is None:
            return
        from kyven_nuke.refine_node import process_current_frame
    else:
        if node.input(0) is None:
            return
        from kyven_nuke.node import process_current_frame
    process_current_frame(node=node, live=True)


def update_live_nodes() -> None:
    """Submit the visible timeline frame once for every enabled idle Kyven node."""
    global _last_scan
    now = time.monotonic()
    if now - _last_scan < 0.15:
        return
    _last_scan = now
    import nuke

    if nuke.executing():
        return
    frame = int(nuke.frame())
    for node in nuke.allNodes("Group"):
        knobs = node.knobs()
        if not {"live_mode", "kyven_live_frame", "kyven_busy"}.issubset(knobs):
            continue
        kind = str(node["kyven_kind"].value()) if "kyven_kind" in knobs else "segment"
        if (
            kind == "refine"
            and "kyven_trimap_preview_frame" in knobs
            and int(node["kyven_trimap_preview_frame"].value()) != frame
        ):
            node["kyven_trimap_preview_frame"].setValue(frame)
            from kyven_nuke.refine_node import request_trimap_preview

            request_trimap_preview(node)
        if not bool(node["live_mode"].value()) or bool(node["kyven_busy"].value()):
            continue
        if int(node["kyven_live_frame"].value()) == frame:
            continue
        node["kyven_live_frame"].setValue(frame)
        if kind == "refine":
            if node.input(0) is None or node.input(1) is None:
                continue
            from kyven_nuke.refine_node import process_current_frame
        else:
            if node.input(0) is None:
                continue
            from kyven_nuke.node import process_current_frame
        process_current_frame(node=node, live=True)
