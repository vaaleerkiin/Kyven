"""Shared Kyven section layout for new and existing DLSS Groups."""

from __future__ import annotations

from typing import Any

from kyven_nuke.branding import add_node_branding
from kyven_nuke.node import _add_knob, _add_section, _nuke

INTERFACE_VERSION = 9
PROCESSING_SCALES = ["1", "0.5", "0.25", "0.125"]


def restyle_dlss_node(node: Any, *, upgrade: bool = False) -> None:
    """Reorder native knobs while preserving node identity, settings and connections."""
    nuke = _nuke()
    callback = str(node["knobChanged"].value())
    node["knobChanged"].setValue("")
    try:
        def ensure(kind: str, name: str, label: str, value: Any = None, choices=None):
            if name not in node.knobs():
                knob = (getattr(nuke, kind)(name, label, choices) if choices is not None
                        else getattr(nuke, kind)(name, label))
                if value is not None:
                    knob.setValue(value)
                _add_knob(nuke, node, knob)
            return node[name]

        old_version = int(node["dlss_ui_version"].value()) if "dlss_ui_version" in node.knobs() else 0
        ensure("Int_Knob", "dlss_ui_version", "Interface Version", INTERFACE_VERSION).setVisible(False)
        ensure("Text_Knob", "kyven_title", "",
               '<font size="5" color="#dce9f2"><b>KYVEN / DLSS</b></font><br>'
               '<font color="#91a3b0">Neural Reconstruction | External Motion | API 36</font>')
        node["kyven_title"].setValue(
            '<font size="5" color="#dce9f2"><b>KYVEN / DLSS</b></font><br>'
            '<font color="#91a3b0">Neural Reconstruction | External Motion | API 36</font>')
        ensure("Enumeration_Knob", "model", "Model", "Kyven DLSS Neural Reconstruction",
               choices=["Kyven DLSS Neural Reconstruction"])
        ensure("PyScript_Knob", "refresh_models", "Refresh Models").setCommand(
            "kyven_nuke.dlss_node.refresh_runtime()")
        ensure("PyScript_Knob", "open_model_manager", "Model Manager...").setCommand(
            "kyven_nuke.model_manager.show_model_manager()")
        ensure("Text_Knob", "model_help", "", "Install and check Kyven DLSS in Model Manager.")
        if "processing_scale" not in node.knobs():
            ensure("Enumeration_Knob", "processing_scale", "Processing Size", "1",
                   choices=PROCESSING_SCALES)
        node["processing_scale"].setTooltip(
            "Scale of Source width and height used for neural processing. "
            "1 = full size; 0.5 = half width and height. Output stays at Source resolution. "
            "Processing is limited to 8192 pixels on the longest side.")
        node["max_dimension"].setVisible(False)
        ensure("Boolean_Knob", "hdr_rolloff", "HDR Rolloff", True)
        ensure("Double_Knob", "rolloff_knee", "Rolloff Knee", 0.5).setRange(0.01, 0.95)
        ensure("Enumeration_Knob", "history_frames", "History Buffer", "3",
               choices=["1", "2", "3", "4"]).setTooltip(
                   "Previous raw correction frames, excluding the current frame. "
                   "External vectors align them; the CPU memory budget can reduce the effective count.")
        ensure("Int_Knob", "history_budget_mb", "History Budget (MiB)", 512).setRange(64, 1024)
        ensure("Text_Knob", "history_status", "History", "No temporal result yet.")
        ensure("Double_Knob", "history_clamp", "History Clamp", 2.0).setRange(0, 10)
        node["history_clamp"].setTooltip("Limit old corrections relative to the current correction. 0 disables clamping.")
        ensure("Boolean_Knob", "motion_consistency", "Check Forward / Backward", False).setTooltip(
            "Requires forward channels from the external vector generator, on the same Motion input.")
        ensure("String_Knob", "forward_x", "Forward X Channel", "forward.u")
        ensure("String_Knob", "forward_y", "Forward Y Channel", "forward.v")
        ensure("Double_Knob", "consistency_threshold", "Vector Error (Source px)", 1.5).setRange(0.01, 100)
        ensure("Double_Knob", "depth_rejection", "Depth Rejection", 0.1).setRange(0.001, 10)
        ensure("PyScript_Knob", "use_nuke_vectors", "Use Nuke Backward Channels").setCommand(
            "kyven_nuke.dlss_node.use_nuke_vectors()")
        node["hdr_rolloff"].setTooltip(
            "Compress highlights and lift negative RGB for inference, then restore "
            "the neural change to the original scene-linear range. Alpha is untouched.")
        ensure("Enumeration_Knob", "output_mode", "Output", "Result", choices=["Result", "Source"])
        ensure("Text_Knob", "output_help", "", "Generated RGB with original alpha and auxiliary channels.")
        ensure("Text_Knob", "cache_help", "", "Choose Cache Folder before caching or processing DLSS.")
        ensure("Text_Knob", "mask_help", "")
        node["mask_help"].setValue(
            "White: allow DLSS changes. Black: preserve Source. Grey: soft neural guide.<br>"
            "Source Alpha or Mask Input guides DLSS inference; output alpha is preserved.")
        ensure("Text_Knob", "advanced_help", "")
        node["advanced_help"].setValue(
            "Generate vectors upstream: VectorGenerator, WAFT (Motion Vectors), or SmartVector > VectorToMotion.<br>"
            "Backward channels: current to previous. Forward channels: current to next, for validation.<br>"
            "Vectors must match Source dimensions; use data/raw channels.")
        ensure("PyScript_Knob", "create_result_read", "Create Result Read").setCommand(
            "kyven_nuke.dlss_node.create_result_read()")
        ensure("PyScript_Knob", "new_cache_version", "New Cache Version").setCommand(
            "kyven_nuke.dlss_node.new_cache_version()")
        sections = {
            "model_section": "MODEL AND PERFORMANCE",
            "reconstruction_section": "NEURAL RECONSTRUCTION",
            "mask_section": "MASK / ALPHA INPUT",
            "temporal_section": "TEMPORAL STABILITY",
            "processing_section": "PROCESSING",
            "output_section": "OUTPUT",
            "cache_section": "CACHE",
            "status_section": "STATUS",
            "motion_section": "EXTERNAL MOTION VECTORS",
            "advanced_section": "ADVANCED CONTROLS",
        }
        for name, title in sections.items():
            if name not in node.knobs():
                _add_section(nuke, node, name, title)
        ensure("Tab_Knob", "dlss_advanced", "Advanced")
        mask = node["mask_source"]
        selected = str(mask.value())
        if old_version < 3 or "Full Frame" not in mask.values():
            ensure("String_Knob", "legacy_mask_source", "Previous Mask Selection", selected).setVisible(False)
            mask.setValues(["Full Frame", "Mask Input", "Source Alpha"])
            mask.setValue("Full Frame")
        mask.setLabel("Mask Alpha Input")
        mask.setEnabled(True)
        node["mask_channel"].setValue("A")
        node["mask_channel"].setVisible(False)
        node["protect_outside"].setVisible(mask.value() != "Full Frame")
        node["protect_outside"].setTooltip(
            "Also protect the full-resolution RGB outside the guide after restoring the neural change.")
        if "mask_section" in node.knobs():
            from kyven_nuke.node import _section_markup

            node["mask_section"].setValue(_section_markup("MASK / ALPHA INPUT"))
        node["worker"].setVisible(False)
        node["guide_help"].setVisible(False)
        if upgrade and old_version < 2:
            node["live_mode"].setValue(True)
        node["dlss_ui_version"].setValue(INTERFACE_VERSION)
        node["cache_help"].setValue("Choose Cache Folder before caching or processing DLSS.")
        node["cache_root"].setTooltip("Required. Input frames and results are stored under Cache Folder / Name / Version.")
        for name, label in (("process_frame", "Process Current Frame"),
                            ("process_range", "Process Frame Range"),
                            ("first_frame", "Range First"), ("last_frame", "Range Last"),
                            ("temporal", "Enable Temporal (external vectors)"),
                            ("temporal_strength", "Temporal Stability"),
                            ("live_mode", "Live Current Frame")):
            node[name].setLabel(label)
        for name in ("cancel", "open_model_manager", "last_frame", "reset_history", "new_cache_version"):
            node[name].clearFlag(nuke.STARTLINE)
        primary = [
            "kyven_logo", "kyven_title", "model_section", "model", "processing_scale",
            "refresh_models", "open_model_manager", "model_help",
            "reconstruction_section", "hdr_rolloff", "intensity", "local_tone", "local_structure", "style", "preset",
            "mask_section", "mask_source", "mask_channel", "protect_outside", "mask_help",
            "temporal_section", "temporal", "history_frames", "temporal_strength", "history_status",
            "processing_section", "live_mode", "process_frame", "cancel", "first_frame", "last_frame",
            "process_range", "reset_history", "output_section", "output_mode", "output_help",
            "cache_section", "cache_root", "dlss_name", "cache_version", "load_cache",
            "create_result_read", "new_cache_version", "cache_help",
            "status_section", "kyven_status", "start_server_control", "stop_server_control",
        ]
        from kyven_nuke.input_cache import ensure_controls
        ensure_controls(node)
        primary[primary.index("process_range"):primary.index("process_range")] = [
            "input_cache_scope", "cache_frames", "refresh_frames"]
        advanced = ["dlss_advanced", "motion_section", "advanced_help", "use_nuke_vectors", "motion_x", "motion_y",
                    "motion_units", "motion_y_up", "motion_scale_x", "motion_scale_y",
                    "motion_consistency", "forward_x", "forward_y", "consistency_threshold",
                    "advanced_section", "history_budget_mb", "history_clamp", "depth_rejection",
                    "rolloff_knee", "depth_channel", "skin_structure", "rejection", "scene_cut", "live_delay"]
        add_node_branding(node, nuke)
        names = list(node.knobs())
        tail = [node[name] for name in names[names.index("kyven") + 1:]]
        for knob in tail:
            node.removeKnob(knob)
        by_name = {knob.name(): knob for knob in tail}
        order = primary + [name for name in by_name if name not in primary + advanced] + advanced
        for name in order:
            if name in by_name:
                node.addKnob(by_name[name])
        if "label" in node.knobs():
            node["label"].setValue("[value kyven_status]")
    finally:
        node["knobChanged"].setValue(callback)
    from kyven_nuke.tooltips import apply_tooltips
    apply_tooltips(node, "dlss")
