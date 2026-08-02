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


def _payload(node: Any, source: Any, source_path: Path, mask_path: Path, model_mask_path: Path,
             output_path: Path, processed_mask_path: Path, patch_path: Path,
             mask_channel: str) -> dict[str, Any]:
    payload = inpaint_node._payload(node, source, source_path, mask_path, model_mask_path,
                                    output_path, processed_mask_path, patch_path, mask_channel)
    payload.update({
        "model_id": MODEL_IDS[min(int(node["model"].getValue()), len(MODEL_IDS) - 1)],
        "prompt": str(node["prompt"].value()),
        "negative_prompt": str(node["negative_prompt"].value()),
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


def create_generative_inpaint_node() -> Any:
    nuke = _nuke(); node = inpaint_node.create_inpaint_node()
    node.setName("KyvenGenerativeInpaint")
    node["kyven"].setLabel("Kyven Generative Inpaint")
    node["kyven_title"].setValue(
        '<font size="5" color="#dce9f2"><b>KYVEN / GENERATIVE INPAINT</b></font><br>'
        '<font color="#91a3b0">SDXL | Prompt + Source + Mask | API 21</font>')
    node["model"].setValues(list(MODEL_LABELS)); node["model"].setValue(0)
    node["model_help"].setValue(
        "Optional ~7 GB download. Best quality at Final; Preview is faster. "
        "CreativeML Open RAIL++-M license applies to this model.")
    if "kyven_kind" not in node.knobs():
        kind = nuke.String_Knob("kyven_kind", "Kind"); kind.setVisible(False); node.addKnob(kind)
    node["kyven_kind"].setValue("generative_inpaint")

    _add_section(nuke, node, "generation_section", "GENERATION")
    prompt = _multiline_knob(nuke, "prompt", "Prompt")
    prompt.setValue("clean background matching the surrounding scene"); _add_knob(nuke, node, prompt)
    _add_knob(nuke, node, _multiline_knob(nuke, "negative_prompt", "Negative Prompt"))
    seed = nuke.Int_Knob("seed", "Seed"); seed.setValue(0); _add_knob(nuke, node, seed)
    _add_knob(nuke, node, nuke.PyScript_Knob(
        "randomize_seed", "Randomize Seed", "kyven_nuke.generative_inpaint_node.randomize_seed()"),
        start_line=False)
    steps = nuke.Int_Knob("steps", "Steps"); steps.setRange(1, 100); steps.setValue(25)
    _add_knob(nuke, node, steps)
    guidance = nuke.Double_Knob("guidance_scale", "Guidance"); guidance.setRange(0, 20); guidance.setValue(6)
    _add_knob(nuke, node, guidance)
    strength = nuke.Double_Knob("strength", "Strength"); strength.setRange(0.01, 0.99); strength.setValue(0.99)
    _add_knob(nuke, node, strength)
    quality = nuke.Enumeration_Knob("render_quality", "Quality", ["preview", "final"]); quality.setValue(1)
    _add_knob(nuke, node, quality)
    low_memory = nuke.Boolean_Knob("low_memory", "Low Memory (8 GB)"); low_memory.setValue(True)
    _add_knob(nuke, node, low_memory)
    _add_knob(nuke, node, nuke.Text_Knob(
        "generation_help", "", "Strength stays below 1.0 so SDXL preserves Source context. "
        "The same Seed is repeatable; changing it creates another variation."))
    previous = "model_help"
    for name in ("generation_section", "prompt", "negative_prompt", "seed", "randomize_seed",
                 "steps", "guidance_scale", "strength", "render_quality", "low_memory",
                 "generation_help"):
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
    node["kyven_status"].setValue("Ready — install SDXL from Model Manager before first use")
    if _inside(node, "KyvenOutputSwitch") is not None: inpaint_node._ensure_inpaint_preview_graph(node)
    return node
