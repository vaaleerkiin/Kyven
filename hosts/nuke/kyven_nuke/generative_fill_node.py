"""Native Nuke Generative Fill Group; inference belongs to the Kyven worker."""
from __future__ import annotations

import threading
import time
from typing import Any

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
MODEL_IDS = (PROVIDER_ID, "flux2-klein-base-4b-q4-lanpaint")
MODEL_LABELS = ("FLUX.2 Klein 4B Q4 + LanPaint", "FLUX.2 Klein 4B Base Q4 + LanPaint (Preview)")


def _model_id(node: Any) -> str:
    return MODEL_IDS[1] if str(node["model"].value()) == MODEL_LABELS[1] else PROVIDER_ID


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
        node["moss_guidance"].setVisible(True)
        node["moss_guidance"].setEnabled(_model_id(node) == MODEL_IDS[1])
    for name in ("zimage_guidance", "zimage_control_scale", "zimage_negative_prompt"):
        if name in node.knobs():
            node[name].setVisible(False)
    selected = str(node["model"].value())
    try:
        if tuple(node["model"].values()) == MODEL_LABELS:
            node["model"].setEnabled(True)
            return
    except (AttributeError, TypeError):
        pass
    callback = node["knobChanged"] if "knobChanged" in node.knobs() else None
    script = callback.value() if callback is not None else ""
    try:
        if callback is not None:
            callback.setValue("")
        node["model"].setValues(list(MODEL_LABELS))
        node["model"].setValue(1 if selected == MODEL_LABELS[1] else 0)
        if selected not in MODEL_LABELS:
            node["steps"].setValue(4)
            node["fill_mask_overlap"].setValue(32)
        node["model"].setEnabled(True)
    finally:
        if callback is not None:
            callback.setValue(script)


def _payload(node: Any, *args: Any) -> dict[str, Any]:
    payload = inpaint_node._payload(node, *args)
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
        "mask_overlap": int(node["fill_mask_overlap"].value()) if "fill_mask_overlap" in node.knobs() else 32,
        "seamless_blend": str(node["fill_seamless_blend"].value()).lower() if "fill_seamless_blend" in node.knobs() else "auto",
        "moss_guidance": float(node["moss_guidance"].value()) if "moss_guidance" in node.knobs() else 4.0,
    }
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
    _set_status(node_name, "Prompt improved with Qwen3-4B. Restore Original can undo this change.")


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
    from kyven_nuke.model_manager import install_required_runtime
    install_required_runtime(_model_id(_nuke().thisNode()))


def knob_changed() -> None:
    nuke = _nuke()
    node = nuke.thisNode()
    knob = nuke.thisKnob().name()
    if knob == "model":
        moss = _model_id(node) != PROVIDER_ID
        node["steps"].setValue(20 if moss else 4)
        node["moss_guidance"].setEnabled(moss)
        node["model_help"].setValue(
            "Base preview: higher step count and CFG; performance and image quality require shot validation."
            if moss else "Qwen3-4B encodes and improves text in RAM. Generation uses the GPU in separate stages.")
    if knob in {"inputChange", "crop_mode", "processing_roi"}:
        from kyven_nuke.node import _clamp_bbox_knob
        _clamp_bbox_knob(node, "processing_roi", inpaint_node.SOURCE_INPUT)
    node["processing_roi"].setVisible(node["crop_mode"].value() == "manual")
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
                f"({value} > 0 && {value} >= parent.mask_threshold) ? clamp({value},0,1) : 0")
        effective = nuke.toNode("KyvenEffectiveMask")
        effective.setInput(0, permission)
        effective["which"].clearAnimated()
        effective["which"].setValue(0)
        nuke.toNode("KyvenProcessedMaskSwitch").setInput(0, permission)
        support = nuke.toNode("KyvenFillMaskSupport") or nuke.nodes.Expression(name="KyvenFillMaskSupport")
        support.setInput(0, permission)
        for i, channel in enumerate(("r", "g", "b", "a")):
            support[f"expr{i}"].setValue(f"{channel}>0 ? 1 : 0")
        clipped_mask = nuke.toNode("KyvenFillClippedMask") or nuke.nodes.Merge2(
            name="KyvenFillClippedMask", operation="multiply")
        clipped_mask.setInput(0, nuke.toNode("KyvenProcessedMaskSwitch"))
        clipped_mask.setInput(1, support)
        nuke.toNode("KyvenPatchMaskAlpha").setInput(1, clipped_mask)
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
        # Retain the established output selector's other modes and cache plumbing.
    finally:
        node.end()


def create_generative_fill_node() -> Any:
    from kyven.generative_fill.prompts import DEFAULT_SYSTEM_PROMPT
    nuke = _nuke()
    node = inpaint_node.create_inpaint_node()
    node["knobChanged"].setValue("")
    node.setName("KyvenGenerativeFill")
    node["kyven"].setLabel("Generative Fill")
    node["kyven_title"].setValue('<font size="5"><b>KYVEN / GENERATIVE FILL</b></font><br>FLUX.2 Klein / Klein Base + LanPaint')
    node["model"].setValues(list(MODEL_LABELS))
    node["model"].setValue(0)
    node["model"].setEnabled(True)
    node["model_help"].setValue("Qwen3-4B encodes and improves text in RAM. Generation uses the GPU in separate stages.")
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
    node["edge_softness"].setValue(4)

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
    node["process_frame"].setValue("kyven_nuke.generative_fill_node.process_current_frame()")
    node["process_range"].setValue("kyven_nuke.generative_fill_node.process_frame_range()")

    node.addKnob(nuke.Tab_Knob("advanced", "Advanced"))
    _add_section(nuke, node, "fill_generation_section", "GENERATION / 8 GB MEMORY")
    for name, label, default, low, high, tip in (
        ("working_size", "Processing Resolution", 768, 256, 1024, "Maximum side of the cropped region in pixels; the output retains Source resolution. Lower values are faster and use less VRAM."),
        ("fill_mask_overlap", "Generation Overlap", 32, 0, 128, "Source pixels generated beyond the artist mask to keep model transitions away from the compositing boundary. Output remains limited to the original mask."),
        ("steps", "Diffusion Steps", 4, 2, 50, "Klein distilled normally uses 4 scheduler steps. More steps cost time."),
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
    retry = nuke.Boolean_Knob("oom_retry", "Retry Smaller After GPU OOM")
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
        "improve_prompt": "Rewrite Prompt locally using the shared Qwen3-4B encoder weights. Generation waits in the background queue.",
        "restore_prompt": "Restore the text that was present before the last successful Improve Prompt action.",
        "seed": "A fixed integer seed makes repeated generation reproducible with the same settings and runtime.",
        "install_fill_runtime": "Prepare the optional private Generative Fill environment. Install model weights separately in Model Manager.",
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
    node["output_mode"].setValue(0)
    node["knobChanged"].setValue("kyven_nuke.generative_fill_node.knob_changed()")
    return node
