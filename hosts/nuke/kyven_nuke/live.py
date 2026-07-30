"""Debounced current-frame processing callbacks for Kyven Nuke nodes."""

from __future__ import annotations

import time

_last_scan = 0.0


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
        if not bool(node["live_mode"].value()) or bool(node["kyven_busy"].value()):
            continue
        if int(node["kyven_live_frame"].value()) == frame:
            continue
        node["kyven_live_frame"].setValue(frame)
        kind = str(node["kyven_kind"].value()) if "kyven_kind" in knobs else "segment"
        if kind == "refine":
            if node.input(0) is None or node.input(1) is None:
                continue
            from kyven_nuke.refine_node import process_current_frame
        else:
            if node.input(0) is None:
                continue
            from kyven_nuke.node import process_current_frame
        process_current_frame(node=node, live=True)
