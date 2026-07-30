"""Nuke menu bootstrap for Kyven."""

from __future__ import annotations

import sys
from pathlib import Path

import nuke

KYVEN_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = KYVEN_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import kyven_nuke

menu = nuke.menu("Nodes").addMenu("Kyven")
menu.addCommand("Segment", kyven_nuke.create_segment_node, icon="")
menu.addCommand("Refine", kyven_nuke.create_refine_node, icon="")
menu.addCommand("Upgrade Selected Segment Node", kyven_nuke.upgrade_selected_segment_node)
menu.addCommand("Start Server", kyven_nuke.start_server, icon="")
if not getattr(nuke, "_kyven_live_callback_installed", False):
    nuke.addUpdateUI(kyven_nuke.update_live_nodes)
    nuke._kyven_live_callback_installed = True
