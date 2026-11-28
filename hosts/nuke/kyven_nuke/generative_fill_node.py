"""Native Nuke Generative Fill Group; inference belongs to the Kyven worker."""
from __future__ import annotations

import threading
import time
from typing import Any

from kyven_nuke import generative_fill_post as post
from kyven_nuke import inpaint_node
from kyven_nuke.node import (
    _add_knob,
    _add_section,
    _begin_node_operation,
    _job_error_text,
    _nuke,
    _operation_node,
    _place_knob_after,
    _set_busy,
    _set_status,
)
from kyven_nuke.runtime import ensure_server

PROVIDER_ID = "flux2-klein-4b-q4-lanpaint"
MAGE_PROVIDER_ID = "mage-flow-edit-turbo-q6"
BASE_PROVIDER_ID = "flux2-klein-base-4b-q4-lanpaint"
BASE_LABEL = "FLUX.2 Klein 4B Base Q4 + LanPaint (Legacy)"
MODEL_IDS = (PROVIDER_ID, MAGE_PROVIDER_ID)
MODEL_LABELS = ("FLUX.2 Klein 4B Q4 + LanPaint", "Mage-Flow Edit Turbo Q6")


def _model_id(node: Any) -> str:
    selected = str(node["model"].value())
    if selected.startswith("Mage-Flow Edit Turbo"):
        return MAGE_PROVIDER_ID
    if selected.startswith("FLUX.2 Klein 4B Base"):
        return BASE_PROVIDER_ID
    return MODEL_IDS[MODEL_LABELS.index(selected)] if selected in MODEL_LABELS else PROVIDER_ID


def upgrade_selected_generative_fill_node() -> None:
    nuke = _nuke()
    selected = nuke.selectedNodes()
    if len(selected) != 1 or "qwen_system_prompt" not in selected[0].knobs():
        nuke.message("Select one Kyven Generative Fill node.")
        return
    node = selected[0]
    _ensure_fill_graph(node)
    node["knobChanged"].setValue("kyven_nuke.generative_fill_node.knob_changed()")
    _set_status(str(node.fullName()),
                "Controls updated; caches preserved. Generate once to prepare live Lab and edge controls."
                if not post.native_ready(node) else "Live Lab and edge controls ready.")


def _ensure_model_control(node: Any) -> None:
    if "model" not in node.knobs():
        return
    if "moss_guidance" not in node.knobs():
        knob = _nuke().Double_Knob("moss_guidance", "MOSS Guidance")
        knob.setRange(1, 10)
        knob.setValue(4.0)
        node.addKnob(knob)
        if "steps" in node.knobs():
            _place_knob_after(node, "moss_guidance", "steps")
    if "moss_guidance" in node.knobs():
        node["moss_guidance"].setLabel("Base Guidance")
        node["moss_guidance"].setVisible(_model_id(node) == BASE_PROVIDER_ID)
        node["moss_guidance"].setEnabled(_model_id(node) == BASE_PROVIDER_ID)
    for name in ("zimage_guidance", "zimage_control_scale", "zimage_negative_prompt"):
        if name in node.knobs():
            node[name].setVisible(False)
    mage = _model_id(node) == MAGE_PROVIDER_ID
    if "fill_is_mage" not in node.knobs():
        flag = _nuke().Boolean_Knob("fill_is_mage", "Mage Backend")
        flag.setVisible(False)
        node.addKnob(flag)
    node["fill_is_mage"].setValue(mage)
    help_text = {
        "model": (
            "Selects Mage or FLUX for local masked image editing. Mage is the factory "
            "default; FLUX uses LanPaint, and Base is a slower preview option. "
            "Install / Repair Model prepares the selected backend."
        ),
        "invert_mask": "Invert the input alpha before interpreting the edit permission mask.",
        "mask_threshold": (
            "Ignore alpha below this cutoff. Retained fractional alpha controls the "
            "final blend; the model's sampling mask is binary."
        ),
        "preview_model_mask": (
            "Preview the exact exported binary model mask, including Model Mask Grow. "
            "FLUX may add Generation Overlap in the worker."
        ),
        "mask_grow": (
            "Grow the FLUX inference region only. Does not expand the result blend alpha. "
            "Mage uses input alpha directly and hides this control."
        ),
        "edge_softness": (
            "Feather the FLUX patch inward within the result permission mask. "
            "Generate again to apply."
        ),
        "output_mode": (
            "Result keeps Source alpha and auxiliary channels. Result + Mask Alpha "
            "uses the result permission alpha; Result Premult also premultiplies RGB. "
            "Generated Patch exposes replacement RGB with permission alpha. Difference "
            "shows the RGB change; Source bypasses generation."
        ),
        "fill_mask_overlap": (
            "Generate extra source pixels beyond the sampling region to move model "
            "transitions away from the composite edge. This does not expand result permission."
        ),
    }
    for name, tip in help_text.items():
        if name in node.knobs():
            node[name].setTooltip(tip)
    if "fill_smooth_removal" in node.knobs():
        node["fill_smooth_removal"].setVisible(False)
    for name in ("mask_grow", "mask_threshold", "fill_mask_overlap"):
        if name in node.knobs():
            node[name].setVisible(not mage)
    if mage:
        node["preview_model_mask"].setTooltip(
            "Exact binary alpha exported to Mage, before any result mask adjustments. "
            "Only crop and working-resolution resampling are applied in the worker.")
    if "working_size" in node.knobs():
        node["working_size"].setRange(256, 2048 if mage else 1024)
        if not mage and node["working_size"].value() > 1024:
            node["working_size"].setValue(1024)
    if "fill_color_radius" not in node.knobs() and "fill_color_match" in node.knobs():
        radius = _nuke().Int_Knob("fill_color_radius", "Color Match Radius (px)")
        radius.setRange(0, 256)
        radius.setValue(64)
        radius.setTooltip("Local background color matching around removal edges, in source pixels. 0 disables the local step. Texture and pixel positions are not warped.")
        node.addKnob(radius)
        _place_knob_after(node, "fill_color_radius", "fill_color_match")
    if "fill_seam_repair" not in node.knobs():
        repair = _nuke().Enumeration_Knob("fill_seam_repair", "Boundary Repair", ["Auto", "Neural", "Blend", "Off"])
        repair.setTooltip("Auto/Blend anchors an inward boundary band to Source. Neural adds a Mage seam pass first and can change edge detail. Pixels outside the original alpha remain unchanged.")
        node.addKnob(repair)
        _place_knob_after(node, "fill_seam_repair", "fill_seamless_blend")
        width = _nuke().Int_Knob("fill_seam_width", "Boundary Width")
        width.setRange(4, 64)
        width.setValue(24)
        width.setTooltip("Width of the inward repair band in source crop pixels. Larger values allow more structural repair and may change nearby detail.")
        node.addKnob(width)
        _place_knob_after(node, "fill_seam_width", "fill_seam_repair")
    for name in ("fill_seam_repair", "fill_seam_width"):
        node[name].setVisible(False)
    if "fill_seamless_blend" in node.knobs():
        node["fill_seamless_blend"].setVisible(not mage)
    for name in ("thinking_steps", "lanpaint_early_stop", "lanpaint_step_size", "lanpaint_friction", "lanpaint_strength", "lanpaint_beta"):
        if name in node.knobs():
            node[name].setEnabled(not mage)
            node[name].setVisible(not mage)
    if "vae_tile_size" in node.knobs():
        node["vae_tile_size"].setVisible(not mage)
    if "install_fill_runtime" in node.knobs():
        node["install_fill_runtime"].setLabel("Install / Repair Model")
        node["install_fill_runtime"].setTooltip("Install or verify the selected model and all its required components. Existing verified weights are reused.")
    if "open_model_manager" in node.knobs():
        node["open_model_manager"].setValue("kyven_nuke.generative_fill_node.open_model_manager()")
    if "fill_ui_version" not in node.knobs():
        version = _nuke().Int_Knob("fill_ui_version", "Fill UI Version")
        version.setValue(2)
        version.setVisible(False)
        node.addKnob(version)
        for name, anchor in (("working_size", "model"), ("edge_softness", "invert_mask"),
                             ("fill_blend_margin", "edge_softness"), ("fill_smooth_removal", "fill_blend_margin"),
                             ("fill_color_mode", "mask_help"), ("fill_color_match", "fill_color_mode"),
                             ("fill_color_radius", "fill_color_match")):
            if name in node.knobs() and anchor in node.knobs():
                _place_knob_after(node, name, anchor)
        if "fill_color_section" in node.knobs():
            node["fill_color_section"].setVisible(False)
    for name, label in (("process_frame", "Generate Current Frame"), ("process_range", "Generate Frame Range")):
        if name in node.knobs():
            node[name].setLabel(label)
    if "model_help" in node.knobs():
        node["model_help"].setValue(
            "Mage: 4 steps. Start at 512; up to 2048 is available. Result mask controls run after generation. Install / Repair Model prepares everything."
            if mage else "Base preview: 20 steps and guidance. Validate each shot. Install / Repair Model prepares everything."
            if _model_id(node) == BASE_PROVIDER_ID else "FLUX: 4 steps with LanPaint. Up to 1024. Install / Repair Model prepares everything.")
    if mage:
        node["mask_help"].setValue("White input alpha selects model edits. Result Grow / Erode and Mask Blur run after generation and do not change the model mask.")
        node["output_help"].setValue("Result keeps Source alpha and channels. RGB outside the final result blend mask remains Source. Result mask adjustments update immediately.")
    selected = str(node["model"].value())
    desired_labels = MODEL_LABELS + ((BASE_LABEL,) if _model_id(node) == BASE_PROVIDER_ID else ())
    post.ensure_controls(node, mage)
    try:
        if tuple(node["model"].values()) == desired_labels:
            node["model"].setEnabled(True)
            return
    except (AttributeError, TypeError):
        pass
    callback = node["knobChanged"] if "knobChanged" in node.knobs() else None
    script = callback.value() if callback is not None else ""
    try:
        if callback is not None:
            callback.setValue("")
        node["model"].setValues(list(desired_labels))
        previous_id = (MAGE_PROVIDER_ID if selected.startswith("Mage-Flow Edit Turbo") else
                       BASE_PROVIDER_ID if selected.startswith("FLUX.2 Klein 4B Base") else PROVIDER_ID)
        node["model"].setValue(2 if previous_id == BASE_PROVIDER_ID else MODEL_IDS.index(previous_id))
        if selected not in MODEL_LABELS and previous_id not in (MAGE_PROVIDER_ID, BASE_PROVIDER_ID):
            node["steps"].setValue(4)
            node["fill_mask_overlap"].setValue(32)
        node["model"].setEnabled(True)
    finally:
        if callback is not None:
            callback.setValue(script)


def _payload(node: Any, *args: Any) -> dict[str, Any]:
    payload = inpaint_node._payload(node, *args, model_id=_model_id(node))
    payload["model_id"] = _model_id(node)
    payload["processing_size"] = 0
    payload["quality_mode"] = "standard"
    payload["fill"] = {
        "prompt": str(node["prompt"].value()), "seed": int(node["seed"].value()),
        "steps": int(node["steps"].value()), "working_size": int(node["working_size"].value()),
        "thinking_steps": int(node["thinking_steps"].value()),
        "step_size": float(node["lanpaint_step_size"].value()),
        "friction": float(node["lanpaint_friction"].value()),
        "strength": float(node["lanpaint_strength"].value()),
        "beta": float(node["lanpaint_beta"].value()),
        "early_stop": int(node["lanpaint_early_stop"].value()),
        "vae_tile_size": int(node["vae_tile_size"].value()),
        "timeout_seconds": int(node["generation_timeout"].value()),
        "oom_retry": bool(node["oom_retry"].value()),
        "color_match": float(node["fill_color_match"].value()) if "fill_color_match" in node.knobs() else 1.0,
        "color_match_mode": str(node["fill_color_mode"].value()).lower() if "fill_color_mode" in node.knobs() else "auto",
        "color_match_radius": int(node["fill_color_radius"].value()) if "fill_color_radius" in node.knobs() else 64,
        "blend_margin": int(node["fill_blend_margin"].value()) if "fill_blend_margin" in node.knobs() and _model_id(node) == MAGE_PROVIDER_ID else 0,
        "mask_overlap": int(node["fill_mask_overlap"].value()) if "fill_mask_overlap" in node.knobs() else 32,
        "seamless_blend": str(node["fill_seamless_blend"].value()).lower() if "fill_seamless_blend" in node.knobs() else "auto",
        "moss_guidance": float(node["moss_guidance"].value()) if "moss_guidance" in node.knobs() else 4.0,
        "seam_repair": str(node["fill_seam_repair"].value()).lower() if "fill_seam_repair" in node.knobs() else "auto",
        "seam_width": int(node["fill_seam_width"].value()) if "fill_seam_width" in node.knobs() else 24,
    }
    if payload["model_id"] == MAGE_PROVIDER_ID:
        # Color and edge sliders operate only on the cached native graph.
        payload["edge_softness"] = 0
        payload["mask_grow"] = 0
        payload["mask_threshold"] = 0
        payload["preprocess_mask"] = True
        payload["fill"].update(native_postprocess=True, blend_margin=0, mask_overlap=0, color_match=1.0,
                               color_match_mode="off", color_match_radius=64)
        payload["fill"]["seam_repair"] = "off"
        payload["fill"]["seamless_blend"] = "off"
    return payload


def process_current_frame(node: Any | None = None, live: bool = False) -> None:
    node = node or _nuke().thisNode()
    if not str(node["prompt"].value()).strip():
        _set_status(str(node.fullName()), "Enter a prompt before generating.")
        return
    inpaint_node.process_current_frame_for_node(
        node, live=live, payload_builder=_payload,
        submit_method="submit_generative_fill", display_name="Generative Fill",
    )


def process_frame_range() -> None:
    node = _nuke().thisNode()
    if not str(node["prompt"].value()).strip():
        _set_status(str(node.fullName()), "Enter a prompt before generating.")
        return
    inpaint_node.process_frame_range_for_node(
        node, payload_builder=_payload, submit_method="submit_generative_fill",
        display_name="Generative Fill",
    )


def _apply_improved_prompt(node_name: str, original: str, improved: str, job_id: str) -> None:
    node = _operation_node(node_name)
    if node is None or str(node["kyven_job_id"].value()) != job_id:
        return
    _set_busy(node_name, False)
    if inpaint_node._export_cancelled(node_name):
        _set_status(node_name, "Prompt improvement cancelled; original text kept.")
        return
    # A delayed response must never overwrite a newer artist edit.
    if str(node["prompt"].value()) != original:
        node["improved_prompt"].setValue(improved)
        _set_status(node_name, "Improved text saved in Advanced; your newer prompt was kept.")
        return
    node["original_prompt"].setValue(original)
    node["prompt"].setValue(improved)
    node["improved_prompt"].setValue(improved)
    _set_status(node_name, "Prompt improved locally. Restore Original can undo this change.")


def improve_prompt() -> None:
    nuke = _nuke()
    node = nuke.thisNode()
    original = str(node["prompt"].value())
    if not original.strip():
        _set_status(str(node.fullName()), "Enter a prompt to improve.")
        return
    if not _begin_node_operation(node):
        return
    name = str(node.fullName())
    with inpaint_node._export_cancel_lock:
        inpaint_node._export_cancellations.discard(name)
    settings = {
        "provider_id": _model_id(node),
        "prompt": original, "system_prompt": str(node["qwen_system_prompt"].value()),
        "max_new_tokens": int(node["qwen_max_tokens"].value()),
        "temperature": float(node["qwen_temperature"].value()),
        "top_p": float(node["qwen_top_p"].value()), "seed": int(node["seed"].value()),
        "timeout_seconds": int(node["qwen_timeout"].value()),
    }
    _set_status(name, "Improving prompt with the selected model's Qwen encoder...")

    def work() -> None:
        try:
            client = ensure_server()
            if inpaint_node._export_cancelled(name):
                raise RuntimeError("Prompt improvement cancelled.")
            job_id = client.submit_prompt_enhance(settings)
            nuke.executeInMainThread(inpaint_node._set_node_job_id, args=(name, job_id))
            while True:
                if inpaint_node._export_cancelled(name):
                    client.cancel(job_id)
                job = client.job(job_id)
                if job["status"] in {"succeeded", "failed", "cancelled"}:
                    break
                nuke.executeInMainThread(_set_status, args=(name,
                    str(job.get("progress_message") or "Improving prompt...")))
                time.sleep(0.25)
            if job["status"] != "succeeded":
                raise RuntimeError(_job_error_text(job))
            nuke.executeInMainThread(_apply_improved_prompt,
                args=(name, original, str(job["result"]["prompt"]), job_id))
        except Exception as exc:  # noqa: BLE001 - background operation boundary
            nuke.executeInMainThread(_set_busy, args=(name, False))
            nuke.executeInMainThread(_set_status, args=(name, f"Prompt improvement failed: {exc}"))

    threading.Thread(target=work, name="kyven-prompt-enhance", daemon=True).start()


def restore_prompt() -> None:
    node = _nuke().thisNode()
    if node["original_prompt"].value():
        node["prompt"].setValue(node["original_prompt"].value())


def reset_system_prompt() -> None:
    from kyven.generative_fill.prompts import DEFAULT_SYSTEM_PROMPT
    _nuke().thisNode()["qwen_system_prompt"].setValue(DEFAULT_SYSTEM_PROMPT)


def install_runtime() -> None:
    from kyven_nuke.model_manager import install_model
    install_model(_model_id(_nuke().thisNode()))


def open_model_manager() -> None:
    from kyven_nuke.model_manager import show_model_manager
    show_model_manager(_model_id(_nuke().thisNode()))


def smooth_removal_edges() -> None:
    node = _nuke().thisNode()
    if _model_id(node) != MAGE_PROVIDER_ID:
        _set_status(str(node.fullName()), "Select Mage before applying the removal edge preset.")
        return
    # Compatibility for old serialized buttons; never alter generation settings.
    node["fill_blend_margin"].setValue(32)
    node["edge_softness"].setValue(24)
    _set_status(str(node.fullName()), "Result mask adjusted after generation.")


def knob_changed() -> None:
    nuke = _nuke()
    node = nuke.thisNode()
    knob = nuke.thisKnob().name()
    from kyven_nuke.input_cache import scope_changed
    if scope_changed(node, knob):
        return
    if knob in post.LIVE_CONTROLS and _model_id(node) == MAGE_PROVIDER_ID:
        enabled = node["fill_color_mode"].value() != "Off"
        for name in ("fill_color_match", "fill_match_luminance", "fill_match_chroma"):
            node[name].setEnabled(enabled)
        _set_status(str(node.fullName()),
                    "Live Color Match / edge adjustment. Write this Group to save the final look."
                    if post.native_ready(node) else
                    "Generate once to enable live Lab Color Match and edge adjustment for this frame.")
        return
    if knob == "working_resolution":
        node["working_size"].setValue(int(node["working_resolution"].value()))
        _set_status(str(node.fullName()), "Processing resolution changed. Generate again to apply.")
        return
    if knob == "model":
        model = _model_id(node)
        base = model == BASE_PROVIDER_ID
        node["steps"].setValue(20 if base else 4)
        node["moss_guidance"].setEnabled(base)
        _ensure_model_control(node)
    if knob in {"inputChange", "crop_mode", "processing_roi"}:
        from kyven_nuke.node import _clamp_bbox_knob
        _clamp_bbox_knob(node, "processing_roi", inpaint_node.SOURCE_INPUT)
    node["processing_roi"].setVisible(node["crop_mode"].value() == "manual")
    node["context_padding"].setVisible(node["crop_mode"].value() == "auto")
    if knob == "fill_color_mode":
        node["fill_color_match"].setEnabled(node["fill_color_mode"].value() != "Off")
        if "fill_color_radius" in node.knobs():
            node["fill_color_radius"].setEnabled(node["fill_color_mode"].value() in {"Auto", "Background"})
    if knob in {"fill_blend_margin", "fill_mask_overlap", "edge_softness", "fill_color_mode",
                "fill_color_match", "fill_color_radius", "context_padding", "working_size",
                "prompt", "seed", "steps", "mask_threshold", "invert_mask", "mask_grow",
                "crop_mode", "processing_roi", "in_colorspace", "profile"}:
        _set_status(str(node.fullName()), "Fill settings changed. Generate Current Frame to update the cached result.")
    if knob == "in_colorspace":
        inpaint_node._configure_color_io(node)
    if knob in {"cache_root", "inpaint_name", "cache_version"}:
        from kyven_nuke.node import _defer_ui_action
        _defer_ui_action(inpaint_node._apply_inpaint_cache_fields, str(node.fullName()))


def _ensure_fill_graph(node: Any) -> None:
    """Keep original float Source and alpha outside the artist mask in Nuke."""
    nuke = _nuke()
    _ensure_model_control(node)
    node.begin()
    try:
        source = nuke.toNode("Source")
        mask = nuke.toNode("KyvenInpaintMaskChannel")
        permission = nuke.toNode("KyvenFillPermission") or nuke.nodes.Expression(name="KyvenFillPermission")
        permission.setInput(0, mask)
        for i, channel in enumerate(("r", "g", "b", "a")):
            value = f"(parent.invert_mask ? 1-{channel} : {channel})"
            permission[f"expr{i}"].setValue(
                f"({value} > 0 && (parent.fill_is_mage || {value} >= parent.mask_threshold)) ? clamp({value},0,1) : 0")
        effective = nuke.toNode("KyvenEffectiveMask")
        effective.setInput(0, permission)
        effective["which"].clearAnimated()
        effective["which"].setValue(0)
        nuke.toNode("KyvenProcessedMaskSwitch").setInput(0, permission)
        support = nuke.toNode("KyvenFillMaskSupport") or nuke.nodes.Expression(name="KyvenFillMaskSupport")
        support.setInput(0, permission)
        for i, channel in enumerate(("r", "g", "b", "a")):
            support[f"expr{i}"].setValue(f"{channel}>0 ? 1 : 0")
        expanded_support = nuke.toNode("KyvenFillBlendSupport") or nuke.nodes.FilterErode(name="KyvenFillBlendSupport")
        expanded_support.setInput(0, support)
        expanded_support["channels"].setValue("rgba")
        expanded_support["size"].setExpression("parent.fill_is_mage ? -parent.fill_blend_margin : 0")
        clipped_mask = nuke.toNode("KyvenFillClippedMask") or nuke.nodes.Merge2(
            name="KyvenFillClippedMask", operation="multiply")
        clipped_mask.setInput(0, nuke.toNode("KyvenProcessedMaskSwitch"))
        clipped_mask.setInput(1, expanded_support)
        nuke.toNode("KyvenPatchMaskAlpha").setInput(1, clipped_mask)
        public_mask = nuke.toNode("KyvenFillPublicMask") or nuke.nodes.Switch(name="KyvenFillPublicMask")
        public_mask.setInput(0, permission)
        public_mask.setInput(1, clipped_mask)
        public_mask["which"].setExpression("parent.fill_is_mage && parent.fill_blend_margin>0")
        nuke.toNode("KyvenGeneratedPatch").setInput(1, public_mask)
        nuke.toNode("KyvenResultSourceAlpha").setInput(1, public_mask)
        sampling_union = nuke.toNode("KyvenFillSamplingUnion") or nuke.nodes.Merge2(name="KyvenFillSamplingUnion", operation="max")
        sampling_union.setInput(0, nuke.toNode("KyvenModelMaskGrow"))
        sampling_union.setInput(1, expanded_support)
        sampling_preview = nuke.toNode("KyvenFillSamplingPreview") or nuke.nodes.FilterErode(name="KyvenFillSamplingPreview")
        sampling_preview.setInput(0, sampling_union)
        sampling_preview["channels"].setValue("rgba")
        sampling_preview["size"].setExpression("-parent.fill_mask_overlap")
        model_mask = nuke.toNode("KyvenFillModelMask") or nuke.nodes.Switch(name="KyvenFillModelMask")
        model_mask.setInput(0, nuke.toNode("KyvenModelMaskGrow"))
        model_mask.setInput(1, support)
        model_mask["which"].setExpression("parent.fill_is_mage")
        nuke.toNode("KyvenInpaintModelMaskWrite").setInput(0, model_mask)
        composite = nuke.toNode("KyvenResultComposite")
        source_alpha = nuke.toNode("KyvenFillSafeSourceAlpha") or nuke.nodes.Expression(
            name="KyvenFillSafeSourceAlpha")
        source_alpha.setInput(0, source)
        source_alpha["channel0"].setValue("rgba.alpha")
        source_alpha["expr0"].setValue("1")
        # Channel availability can change upstream after this group is created.
        # Evaluate once as a knob expression, never as per-pixel Python.
        source_alpha["disable"].setExpression(
            '[python {int("rgba.alpha" in nuke.thisNode().input(0).channels())}]')
        preserve = nuke.toNode("KyvenFillSourceAlpha") or nuke.nodes.Copy(name="KyvenFillSourceAlpha")
        preserve.setInput(0, composite)
        preserve.setInput(1, source_alpha)
        preserve["from0"].setValue("rgba.alpha")
        preserve["to0"].setValue("rgba.alpha")
        output = nuke.toNode("KyvenOutputSwitch")
        output.setInput(0, preserve)
        output.setInput(6, model_mask)
        post.ensure_graph(node)
        # Retain the established output selector's other modes and cache plumbing.
    finally:
        node.end()


def create_generative_fill_node() -> Any:
    from kyven.generative_fill.prompts import DEFAULT_SYSTEM_PROMPT
    from kyven_nuke.preferences import default_model_index
    nuke = _nuke()
    node = inpaint_node.create_inpaint_node()
    node["knobChanged"].setValue("")
    node.setName("KyvenGenerativeFill")
    node["kyven"].setLabel("Generative Fill")
    node["kyven_title"].setValue('<font size="5"><b>KYVEN / GENERATIVE FILL</b></font><br>Mage / FLUX Generative Fill')
    node["model"].setValues(list(MODEL_LABELS))
    node["model"].setValue(default_model_index("generative_fill", MODEL_IDS))
    node["model"].setEnabled(True)
    node["model_help"].setValue("Connect Source and Mask, enter an edit, then Generate Current Frame. Install / Repair Model prepares the selected backend.")
    node["mask_help"].setValue("White alpha selects edits. Model Grow affects generation only; the original mask limits the result.")
    node["output_help"].setValue("Result keeps Source alpha and channels. RGB outside the input mask comes from the original Nuke Source.")
    node["roi_help"].setValue("Auto crops to the mask plus Context Padding. Working Size limits GPU memory; the patch returns at Source resolution.")
    node["inpaint_name"].setValue("generative_fill")
    node["inpaint_name"].setLabel("Fill Name")
    for name in ("refresh_models", "quality_mode", "refinement_steps", "refinement_strength",
                 "refinement_scales", "refinement_help", "preprocess_mask", "live_mode",
                 "live_interval", "live_help"):
        if name in node.knobs():
            node[name].setVisible(False)
    node["live_mode"].setValue(False)
    node["live_mode"].setEnabled(False)
    node["preprocess_mask"].setValue(True)
    node["quality_mode"].setValue(0)
    node["context_padding"].setValue(256)
    node["edge_softness"].setValue(0)

    prompt = nuke.Multiline_Eval_String_Knob("prompt", "Prompt")
    prompt.setTooltip("Describe the edit: remove, add, replace or modify. Russian is translated automatically. The mask limits where pixels may change.")
    node.addKnob(prompt)
    _place_knob_after(node, "prompt", "kyven_title")
    for name, label, action, anchor in (
        ("improve_prompt", "Improve Prompt", "improve_prompt", "prompt"),
        ("restore_prompt", "Restore Original", "restore_prompt", "improve_prompt"),
        ("install_fill_runtime", "Install Runtime", "install_runtime", "open_model_manager"),
    ):
        node.addKnob(nuke.PyScript_Knob(name, label, f"kyven_nuke.generative_fill_node.{action}()"))
        _place_knob_after(node, name, anchor)
    seed = nuke.Double_Knob("seed", "Seed")
    seed.setRange(0, 2**32 - 1)
    seed.setValue(42)
    node.addKnob(seed)
    _place_knob_after(node, "seed", "restore_prompt")
    original = nuke.String_Knob("original_prompt", "Original Prompt")
    original.setVisible(False)
    node.addKnob(original)
    node["process_frame"].setLabel("Generate Current Frame")
    node["process_frame"].setValue("kyven_nuke.generative_fill_node.process_current_frame()")
    node["process_range"].setLabel("Generate Frame Range")
    node["process_range"].setValue("kyven_nuke.generative_fill_node.process_frame_range()")

    node.addKnob(nuke.Tab_Knob("advanced", "Advanced"))
    _add_section(nuke, node, "fill_generation_section", "GENERATION / 8 GB MEMORY")
    for name, label, default, low, high, tip in (
        ("working_size", "Processing Resolution", 512, 256, 1024, "Maximum side of the cropped region in pixels; the output retains Source resolution. Lower values are faster and use less VRAM."),
        ("fill_mask_overlap", "Generation Overlap", 32, 0, 128, "Source pixels generated beyond the artist mask to keep model transitions away from the compositing boundary. Output remains limited to the original mask."),
        ("steps", "Diffusion Steps", 4, 2, 50, "Mage and distilled Klein use 4 steps. Base starts at 20. More steps cost time."),
        ("thinking_steps", "LanPaint Thinking Steps", 2, 1, 8, "Extra conditional sampling iterations. More improves some edits but increases generation time."),
        ("lanpaint_early_stop", "Early Stop", 1, 0, 29, "Skip thinking iterations during this many final diffusion steps. Must be smaller than Diffusion Steps."),
        ("vae_tile_size", "VAE Tile Size", 256, 128, 512, "Smaller decode tiles reduce temporary GPU memory."),
        ("generation_timeout", "Generation Timeout (s)", 1800, 30, 7200, "Cancel and unload the worker if a generation exceeds this limit."),
    ):
        knob = nuke.Int_Knob(name, label)
        knob.setRange(low, high)
        knob.setValue(default)
        knob.setTooltip(tip)
        _add_knob(nuke, node, knob)
    retry = nuke.Boolean_Knob("oom_retry", "Retry Smaller After Memory Error")
    retry.setValue(True)
    retry.setTooltip("Retry once with a smaller working crop. The original output mask stays unchanged.")
    _add_knob(nuke, node, retry)
    for name, label, default, low, high in (
        ("moss_guidance", "MOSS Guidance", 4.0, 1.0, 10.0),
        ("lanpaint_step_size", "Langevin Step Size", 0.2, 0.001, 1),
        ("lanpaint_friction", "Friction", 15, 0.1, 50),
        ("lanpaint_strength", "Constraint Strength", 8, 0.1, 32),
        ("lanpaint_beta", "Noise Beta", 1, 0.01, 4),
    ):
        knob = nuke.Double_Knob(name, label)
        knob.setRange(low, high)
        knob.setValue(default)
        knob.setTooltip("Advanced LanPaint solver parameter. Restore defaults if the result becomes unstable.")
        _add_knob(nuke, node, knob)
    for name in ("profile", "mask_threshold", "mask_grow", "edge_softness", "edge_color_match"):
        knob = node[name]
        node.removeKnob(knob)
        _add_knob(nuke, node, knob)
    node["profile"].setValue("balanced")
    node["mask_grow"].setLabel("Model Mask Grow")
    node["mask_grow"].setRange(0, 128)
    node["mask_grow"].setTooltip("Grow the inference region only. Pixels outside the original artist mask remain unchanged.")
    node["edge_color_match"].setVisible(False)
    _add_section(nuke, node, "fill_color_section", "COLOR MATCHING")
    mode = nuke.Enumeration_Knob("fill_color_mode", "Color Matching",
                                 ["Auto", "Context", "Background", "Off"])
    mode.setTooltip("Auto calibrates to clean context and matches flat background surfaces for removals. Context preserves new subject colors. Background forces removal matching.")
    _add_knob(nuke, node, mode)
    strength = nuke.Double_Knob("fill_color_match", "Color Match Strength")
    strength.setRange(0, 1)
    strength.setValue(1)
    strength.setTooltip("0 keeps generated colors; 1 applies full calibration from clean Source context. Source pixels inside the removal mask are never used as a color reference.")
    _add_knob(nuke, node, strength)
    stitch = nuke.Enumeration_Knob("fill_seamless_blend", "Seamless Blend", ["Auto", "On", "Off"])
    stitch.setTooltip("Auto matches boundaries for removals and all Base edits. On also applies it to other additions/replacements and may change their colors. Off keeps standard masked compositing.")
    _add_knob(nuke, node, stitch)
    _add_section(nuke, node, "qwen_section", "QWEN PROMPT IMPROVEMENT")
    system = nuke.Multiline_Eval_String_Knob("qwen_system_prompt", "System Prompt")
    system.setValue(DEFAULT_SYSTEM_PROMPT)
    system.setTooltip("Instruction used only by Improve Prompt. Qwen is text-only and cannot inspect the source image.")
    _add_knob(nuke, node, system)
    for name, label, default, low, high in (
        ("qwen_max_tokens", "Maximum New Tokens", 192, 16, 512),
        ("qwen_timeout", "Prompt Timeout (s)", 600, 30, 1800),
    ):
        knob = nuke.Int_Knob(name, label)
        knob.setRange(low, high)
        knob.setValue(default)
        _add_knob(nuke, node, knob)
    for name, label, default, low, high in (
        ("qwen_temperature", "Temperature", 0.6, 0, 2),
        ("qwen_top_p", "Top P", 0.9, 0.01, 1),
    ):
        knob = nuke.Double_Knob(name, label)
        knob.setRange(low, high)
        knob.setValue(default)
        _add_knob(nuke, node, knob)
    _add_knob(nuke, node, nuke.PyScript_Knob("reset_qwen_system", "Reset System Prompt",
                 "kyven_nuke.generative_fill_node.reset_system_prompt()"))
    _add_knob(nuke, node, nuke.Multiline_Eval_String_Knob("improved_prompt", "Last Improved Prompt"))
    for name, tip in {
        "improve_prompt": "Rewrite the editing instruction locally with the selected model's Qwen encoder. No source image is sent to prompt improvement.",
        "restore_prompt": "Restore the text that was present before the last successful Improve Prompt action.",
        "seed": "A fixed integer seed makes repeated generation reproducible with the same settings and runtime.",
        "install_fill_runtime": "Install or verify the selected model and all required components. Existing verified weights are reused.",
        "qwen_max_tokens": "Limit how much text Qwen can generate. Lower values are faster on CPU.",
        "qwen_timeout": "Stop prompt improvement and unload its worker when this time limit is reached.",
        "qwen_temperature": "Text sampling randomness. Zero selects deterministic greedy decoding.",
        "qwen_top_p": "Restrict text sampling to the smallest token set reaching this cumulative probability.",
        "reset_qwen_system": "Restore the default prompt-refinement instruction without changing the image prompt.",
        "improved_prompt": "Last Qwen result, including results saved when a newer artist edit prevented automatic replacement.",
        "edge_softness": "Feather the generated patch inward. Pixels outside the original input mask are preserved.",
    }.items():
        node[name].setTooltip(tip)
    _ensure_fill_graph(node)
    node["fill_color_mode"].setTooltip("Auto matches clean Source context and, for explicit removal prompts, continuous flat backgrounds. Context preserves the new object's colors. Background forces removal matching. Off disables color correction.")
    if _model_id(node) == BASE_PROVIDER_ID:
        node["steps"].setValue(20)
    post.ensure_controls(node, _model_id(node) == MAGE_PROVIDER_ID)
    node["output_mode"].setValue(0)
    node["knobChanged"].setValue("kyven_nuke.generative_fill_node.knob_changed()")
    return node
