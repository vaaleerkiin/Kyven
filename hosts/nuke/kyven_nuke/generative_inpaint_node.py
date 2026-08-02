"""Separate prompt-guided SDXL Inpainting node for Nuke."""

from __future__ import annotations

import secrets
import threading
from pathlib import Path
from typing import Any

from kyven_nuke import inpaint_node
from kyven_nuke.node import _add_knob, _add_section, _inside, _nuke, _place_knob_after, _set_status
from kyven_nuke.runtime import ensure_server

MODEL_IDS = ("sdxl-inpainting-1.0",)
MODEL_LABELS = ("SDXL Inpainting 1.0 (optional, ~7 GB, 8 GB+ VRAM)",)
GENERATION_MODE_IDS = ("clean_plate", "replace")
OUTPUT_LABELS = (
    "Result",
    "Result + Mask Alpha",
    "Result Premult",
    "Raw Patch (shows ROI seams)",
    "Difference",
    "Source",
)


def _payload(node: Any, source: Any, source_path: Path, mask_path: Path, model_mask_path: Path,
             output_path: Path, processed_mask_path: Path, patch_path: Path,
             mask_channel: str) -> dict[str, Any]:
    payload = inpaint_node._payload(node, source, source_path, mask_path, model_mask_path,
                                    output_path, processed_mask_path, patch_path, mask_channel)
    payload.update({
        "model_id": MODEL_IDS[min(int(node["model"].getValue()), len(MODEL_IDS) - 1)],
        "prompt": str(node["prompt"].value()),
        "negative_prompt": str(node["negative_prompt"].value()),
        "generation_mode": (
            GENERATION_MODE_IDS[int(node["generation_mode"].getValue())]
            if "generation_mode" in node.knobs()
            else "clean_plate"
        ),
        "seam_blend": (
            int(node["seam_blend"].value()) if "seam_blend" in node.knobs() else 16
        ),
        "seed": int(node["seed"].value()),
        "steps": int(node["steps"].value()),
        "guidance_scale": float(node["guidance_scale"].value()),
        "strength": float(node["strength"].value()),
        "low_memory": bool(node["low_memory"].value()),
        "render_quality": str(node["render_quality"].value()),
    })
    return payload


def process_current_frame() -> None:
    inpaint_node.process_current_frame_for_node(
        _nuke().thisNode(), payload_builder=_payload,
        submit_method="submit_generative_inpaint", display_name="Generative Inpaint")


def process_frame_range() -> None:
    inpaint_node.process_frame_range_for_node(
        _nuke().thisNode(), payload_builder=_payload,
        submit_method="submit_generative_inpaint", display_name="Generative Inpaint")


def randomize_seed() -> None:
    _nuke().thisNode()["seed"].setValue(secrets.randbelow(2**31))


def cancel_current_job() -> None:
    inpaint_node.cancel_current_job()


def create_read_from_current_result() -> None:
    inpaint_node.create_read_from_current_result()


def delete_this_node_cache() -> None:
    inpaint_node.delete_this_node_cache()


def reset_roi_to_input() -> None:
    inpaint_node.reset_roi_to_input()


def knob_changed() -> None:
    inpaint_node.knob_changed()
    node = _nuke().thisNode()
    _update_mode_ui(node)
    node["model_help"].setValue(
        "Prompt-guided 1024-class inpainting. Preview uses up to 768 px / 12 steps; "
        "Final uses up to 1024 px and the selected Steps. Use a tight ROI on 8 GB GPUs.")


def refresh_models() -> None:
    nuke = _nuke(); node_name = nuke.thisNode().fullName()

    def refresh() -> None:
        try:
            models = [m for m in ensure_server().models() if m.get("task") == "generative_inpaint"]
            labels = []
            for model in models:
                suffix = "installed" if model["installed"] else "not installed"
                if model["compatible"] is False: suffix += ", VRAM warning"
                labels.append(f"{model['display_name']} [{suffix}]")
            nuke.executeInMainThread(inpaint_node._apply_model_labels, args=(node_name, labels))
        except Exception as exc:  # noqa: BLE001
            nuke.executeInMainThread(_set_status, args=(node_name, f"Model refresh failed: {exc}"))

    threading.Thread(target=refresh, name="kyven-generative-models", daemon=True).start()


def _multiline_knob(nuke: Any, name: str, label: str) -> Any:
    return getattr(nuke, "Multiline_Eval_String_Knob", nuke.String_Knob)(name, label)


def _update_mode_ui(node: Any) -> None:
    if "generation_mode" not in node.knobs():
        return
    clean_plate = int(node["generation_mode"].getValue()) == 0
    node["prompt"].setLabel("Scene Hint (optional)" if clean_plate else "Replacement Prompt")
    node["negative_prompt"].setLabel(
        "Additional Exclusions" if clean_plate else "Negative Prompt"
    )
    node["generation_help"].setValue(
        (
            "Clean Plate uses a protected background-only prompt and exclusions for people, "
            "objects, text, and duplicates. Seam Blend softens only the RGB inside the mask."
        )
        if clean_plate
        else (
            "Replace follows your prompt. The same Seed is repeatable; changing it creates "
            "another variation. Seam Blend affects only the final RGB composite."
        )
    )
    if "output_mode" in node.knobs():
        current_labels = tuple(node["output_mode"].values())
        if current_labels != OUTPUT_LABELS:
            previous = int(node["output_mode"].getValue())
            node["output_mode"].setValues(list(OUTPUT_LABELS))
            node["output_mode"].setValue(min(previous, len(OUTPUT_LABELS) - 1))
    if "output_help" in node.knobs():
        node["output_help"].setValue(
            "Use Result or Result + Mask Alpha for the finished composite. Raw Patch is a "
            "diagnostic full-ROI model image and intentionally shows rectangular ROI seams."
        )


def create_generative_inpaint_node() -> Any:
    nuke = _nuke(); node = inpaint_node.create_inpaint_node()
    node.setName("KyvenGenerativeInpaint")
    node["kyven"].setLabel("Kyven Generative Inpaint")
    node["kyven_title"].setValue(
        '<font size="5" color="#dce9f2"><b>KYVEN / GENERATIVE INPAINT</b></font><br>'
        '<font color="#91a3b0">SDXL | Clean Plate or Replace | API 22</font>')
    node["model"].setValues(list(MODEL_LABELS)); node["model"].setValue(0)
    node["model_help"].setValue(
        "Optional ~7 GB download. Best quality at Final; Preview is faster. "
        "CreativeML Open RAIL++-M license applies to this model.")
    if "kyven_kind" not in node.knobs():
        kind = nuke.String_Knob("kyven_kind", "Kind"); kind.setVisible(False); node.addKnob(kind)
    node["kyven_kind"].setValue("generative_inpaint")

    _add_section(nuke, node, "generation_section", "GENERATION")
    mode = nuke.Enumeration_Knob(
        "generation_mode", "Mode", ["Remove / Clean Plate", "Replace / Prompt"]
    )
    _add_knob(nuke, node, mode)
    prompt = _multiline_knob(nuke, "prompt", "Scene Hint (optional)")
    _add_knob(nuke, node, prompt)
    _add_knob(nuke, node, _multiline_knob(nuke, "negative_prompt", "Additional Exclusions"))
    seed = nuke.Int_Knob("seed", "Seed"); seed.setValue(0); _add_knob(nuke, node, seed)
    _add_knob(nuke, node, nuke.PyScript_Knob(
        "randomize_seed", "Randomize Seed", "kyven_nuke.generative_inpaint_node.randomize_seed()"),
        start_line=False)
    steps = nuke.Int_Knob("steps", "Steps"); steps.setRange(1, 100); steps.setValue(25)
    _add_knob(nuke, node, steps)
    guidance = nuke.Double_Knob("guidance_scale", "Guidance"); guidance.setRange(0, 20); guidance.setValue(4)
    _add_knob(nuke, node, guidance)
    strength = nuke.Double_Knob("strength", "Strength"); strength.setRange(0.01, 0.99); strength.setValue(0.99)
    _add_knob(nuke, node, strength)
    quality = nuke.Enumeration_Knob("render_quality", "Quality", ["preview", "final"]); quality.setValue(1)
    _add_knob(nuke, node, quality)
    low_memory = nuke.Boolean_Knob("low_memory", "Low Memory (8 GB)"); low_memory.setValue(True)
    _add_knob(nuke, node, low_memory)
    seam = nuke.Int_Knob("seam_blend", "Seam Blend (px)")
    seam.setRange(0, 128); seam.setValue(16); _add_knob(nuke, node, seam)
    _add_knob(nuke, node, nuke.Text_Knob(
        "generation_help", "", ""))
    previous = "model_help"
    for name in ("generation_section", "generation_mode", "prompt", "negative_prompt", "seed", "randomize_seed",
                 "steps", "guidance_scale", "strength", "render_quality", "low_memory",
                 "seam_blend", "generation_help"):
        _place_knob_after(node, name, previous); previous = name

    for name in ("live_mode", "live_help"):
        if name in node.knobs():
            node[name].setVisible(False)
            if name == "live_mode": node[name].setValue(False)
    commands = {
        "refresh_models": "kyven_nuke.generative_inpaint_node.refresh_models()",
        "process_frame": "kyven_nuke.generative_inpaint_node.process_current_frame()",
        "process_range": "kyven_nuke.generative_inpaint_node.process_frame_range()",
        "cancel": "kyven_nuke.generative_inpaint_node.cancel_current_job()",
        "reset_roi": "kyven_nuke.generative_inpaint_node.reset_roi_to_input()",
        "create_result_read": "kyven_nuke.generative_inpaint_node.create_read_from_current_result()",
        "delete_node_cache": "kyven_nuke.generative_inpaint_node.delete_this_node_cache()",
    }
    for name, command in commands.items(): node[name].setCommand(command)
    node["knobChanged"].setValue("kyven_nuke.generative_inpaint_node.knob_changed()")
    node["mask_help"].setValue(
        "The preprocessed Model Mask is sent to SDXL and drives Mask Alpha / Premult output.")
    if int(node["context_padding"].value()) == 128:
        node["context_padding"].setValue(256)
    _update_mode_ui(node)
    node["kyven_status"].setValue("Ready — install SDXL from Model Manager before first use")
    if _inside(node, "KyvenOutputSwitch") is not None: inpaint_node._ensure_inpaint_preview_graph(node)
    return node


def upgrade_selected_generative_inpaint_node() -> None:
    nuke = _nuke(); selected = nuke.selectedNodes()
    if len(selected) != 1 or selected[0].Class() != "Group":
        nuke.message("Select one Kyven Generative Inpaint Group node first.")
        return
    node = selected[0]
    if "prompt" not in node.knobs() or "low_memory" not in node.knobs():
        nuke.message("The selected Group is not a Kyven Generative Inpaint node.")
        return
    if "generation_mode" not in node.knobs():
        mode = nuke.Enumeration_Knob(
            "generation_mode", "Mode", ["Remove / Clean Plate", "Replace / Prompt"]
        )
        node.addKnob(mode); _place_knob_after(node, "generation_mode", "generation_section")
    if "seam_blend" not in node.knobs():
        seam = nuke.Int_Knob("seam_blend", "Seam Blend (px)")
        seam.setRange(0, 128); seam.setValue(16); node.addKnob(seam)
        _place_knob_after(node, "seam_blend", "low_memory")
    if str(node["prompt"].value()) == "clean background matching the surrounding scene":
        node["prompt"].setValue("")
    if float(node["guidance_scale"].value()) == 6.0:
        node["guidance_scale"].setValue(4.0)
    if int(node["context_padding"].value()) == 128:
        node["context_padding"].setValue(256)
    node["kyven_title"].setValue(
        '<font size="5" color="#dce9f2"><b>KYVEN / GENERATIVE INPAINT</b></font><br>'
        '<font color="#91a3b0">SDXL | Clean Plate or Replace | API 22</font>'
    )
    _update_mode_ui(node)
    if int(node["output_mode"].getValue()) == 3:
        node["output_mode"].setValue(1)
    node["kyven_status"].setValue("Updated — Clean Plate and Seam Blend are ready")
