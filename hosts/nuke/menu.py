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

ICONS_ROOT = Path(__file__).parent / "icons"
nuke.pluginAddPath(str(ICONS_ROOT))
LOGO_ICON = "kyven_logo_icon.png"

menu = nuke.menu("Nodes").addMenu("Kyven", icon=LOGO_ICON)
menu.addCommand("Model Manager...", kyven_nuke.show_model_manager, icon=LOGO_ICON)
menu.addSeparator()
menu.addCommand("Segment", kyven_nuke.create_segment_node, icon=LOGO_ICON)
menu.addCommand("Refine", kyven_nuke.create_refine_node, icon=LOGO_ICON)
menu.addCommand("Inpaint", kyven_nuke.create_inpaint_node, icon=LOGO_ICON)
menu.addCommand("Generative Fill", kyven_nuke.create_generative_fill_node, icon=LOGO_ICON)
menu.addCommand("DLSS (Optional RTX)", kyven_nuke.create_dlss_node, icon=LOGO_ICON)
if (KYVEN_ROOT / "plugins" / "Cattery" / "WAFT" / "WAFT.gizmo").is_file():
    menu.addCommand("WAFT SmartVector", 'nuke.createNode("WAFT")', icon=LOGO_ICON)
menu.addSeparator()
menu.addCommand("Upgrade Selected Segment Node", kyven_nuke.upgrade_selected_segment_node)
menu.addCommand("Upgrade Selected Refine Node", kyven_nuke.upgrade_selected_refine_node)
menu.addCommand("Upgrade Selected Inpaint Node", kyven_nuke.upgrade_selected_inpaint_node)
menu.addCommand("Upgrade Selected Generative Fill Node", kyven_nuke.upgrade_selected_generative_fill_node)
menu.addCommand("Upgrade Selected DLSS Node", kyven_nuke.upgrade_selected_dlss_node)
menu.addCommand("Start Kyven Server", kyven_nuke.start_server, icon=LOGO_ICON)
menu.addCommand("Stop Kyven Server", kyven_nuke.stop_server_from_menu, icon=LOGO_ICON)
if not getattr(nuke, "_kyven_live_callback_installed", False):
    nuke.addUpdateUI(kyven_nuke.update_live_nodes)
    nuke._kyven_live_callback_installed = True
if not getattr(nuke, "_kyven_segment_repair_callback_installed", False):
    nuke.addOnScriptLoad(kyven_nuke.repair_segment_nodes)
    nuke._kyven_segment_repair_callback_installed = True
kyven_nuke.repair_segment_nodes()
if not getattr(nuke, "_kyven_dlss_repair_callback_installed", False):
    nuke.addOnScriptLoad(kyven_nuke.repair_dlss_nodes)
    nuke._kyven_dlss_repair_callback_installed = True
kyven_nuke.repair_dlss_nodes()
