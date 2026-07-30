"""Kyven Segment Group node and background job orchestration."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any

from kyven_nuke import config
from kyven_nuke.payload import MODEL_LABELS, segment_payload
from kyven_nuke.runtime import ensure_server


def _nuke() -> Any:
    import nuke

    return nuke


def _inside(group: Any, name: str) -> Any:
    group.begin()
    try:
        return _nuke().toNode(name)
    finally:
        group.end()


def _set_status(node_name: str, status: str) -> None:
    node = _nuke().toNode(node_name)
    if node is not None:
        node["kyven_status"].setValue(status)


def _apply_result(node_name: str, job: dict[str, Any]) -> None:
    nuke = _nuke()
    node = nuke.toNode(node_name)
    if node is None:
        return
    if job["status"] != "succeeded":
        error = job.get("error") or {}
        node["kyven_status"].setValue(f"Failed: {error.get('message', job['status'])}")
        return
    output = str(job["result"]["output"])
    node.begin()
    try:
        matte = nuke.toNode("KyvenMatteRead")
        if matte is None:
            matte = nuke.nodes.Read(name="KyvenMatteRead", file=output)
            nuke.toNode("KyvenMatteSwitch").setInput(1, matte)
        else:
            matte["file"].setValue(output)
            matte["reload"].execute()
        nuke.toNode("KyvenMatteSwitch")["which"].setValue(1)
    finally:
        node.end()
    node["kyven_status"].setValue(
        f"Ready — score {float(job['result']['score']):.3f}"
    )


def _submit_and_wait(node_name: str, payload: dict[str, Any]) -> None:
    nuke = _nuke()
    try:
        client = ensure_server()
        job_id = client.submit_segment(payload)
        nuke.executeInMainThread(_set_job_id, args=(node_name, job_id))
        job = client.wait(job_id)
        nuke.executeInMainThread(_apply_result, args=(node_name, job))
    except Exception as exc:  # noqa: BLE001 - background boundary must report to the host UI
        nuke.executeInMainThread(_set_status, args=(node_name, f"Failed: {exc}"))


def _set_job_id(node_name: str, job_id: str) -> None:
    node = _nuke().toNode(node_name)
    if node is not None:
        node["kyven_job_id"].setValue(job_id)
        node["kyven_status"].setValue("Segmenting…")


def _cache_paths(node: Any, frame: int) -> tuple[Path, Path]:
    root = config.cache_dir() / node["kyven_uuid"].value()
    root.mkdir(parents=True, exist_ok=True)
    return root / f"source.{frame}.png", root / f"matte.{frame}.png"


def process_current_frame() -> None:
    nuke = _nuke()
    node = nuke.thisNode()
    source = node.input(0)
    if source is None:
        nuke.message("Kyven Segment requires a Source input.")
        return
    if not node["positive_enabled"].value() and not node["negative_enabled"].value() and not node["box_enabled"].value():
        nuke.message("Enable at least one positive point, negative point, or box prompt.")
        return

    frame = int(nuke.frame())
    source_path, matte_path = _cache_paths(node, frame)
    payload = segment_payload(
        source=str(source_path.resolve()),
        output=str(matte_path.resolve()),
        model_index=int(node["model"].getValue()),
        profile=str(node["profile"].value()),
        image_height=int(source.height()),
        positive_enabled=bool(node["positive_enabled"].value()),
        positive_xy=tuple(node["positive_point"].value()),
        negative_enabled=bool(node["negative_enabled"].value()),
        negative_xy=tuple(node["negative_point"].value()),
        box_enabled=bool(node["box_enabled"].value()),
        box=tuple(node["prompt_box"].value()),
    )

    writer = _inside(node, "KyvenSourceWrite")
    writer["file"].setValue(str(source_path))
    node["kyven_status"].setValue("Exporting source frame...")
    try:
        nuke.execute(writer, frame, frame)
    except Exception as exc:  # noqa: BLE001 - host boundary must report useful context
        node["kyven_status"].setValue(f"Source export failed: {exc}")
        nuke.message(f"Kyven source export failed:\n{exc}")
        return
    if not source_path.is_file():
        node["kyven_status"].setValue("Source export failed: PNG was not created.")
        return
    node["kyven_status"].setValue("Starting Kyven server...")
    threading.Thread(
        target=_submit_and_wait,
        args=(node.fullName(), payload),
        name="kyven-nuke-submit",
        daemon=True,
    ).start()


def cancel_current_job() -> None:
    nuke = _nuke()
    node = nuke.thisNode()
    job_id = node["kyven_job_id"].value()
    if not job_id:
        node["kyven_status"].setValue("No submitted Kyven job to cancel.")
        return

    def cancel() -> None:
        try:
            ensure_server().cancel(job_id)
            nuke.executeInMainThread(_set_status, args=(node.fullName(), "Cancellation requested."))
        except Exception as exc:  # noqa: BLE001 - background boundary must report to the host UI
            nuke.executeInMainThread(_set_status, args=(node.fullName(), f"Cancel failed: {exc}"))

    threading.Thread(target=cancel, name="kyven-nuke-cancel", daemon=True).start()


def refresh_models() -> None:
    nuke = _nuke()
    node = nuke.thisNode()
    node_name = node.fullName()

    def refresh() -> None:
        try:
            models = ensure_server().models()
            labels = []
            for model in models:
                suffix = "installed" if model["installed"] else "not installed"
                if model["compatible"] is False:
                    suffix += ", VRAM warning"
                labels.append(f"{model['display_name']} [{suffix}]")
            nuke.executeInMainThread(_apply_model_labels, args=(node_name, labels))
        except Exception as exc:  # noqa: BLE001 - background boundary must report to the host UI
            nuke.executeInMainThread(_set_status, args=(node_name, f"Model refresh failed: {exc}"))

    threading.Thread(target=refresh, name="kyven-model-refresh", daemon=True).start()


def _apply_model_labels(node_name: str, labels: list[str]) -> None:
    node = _nuke().toNode(node_name)
    if node is None or not labels:
        return
    previous = int(node["model"].getValue())
    node["model"].setValues(labels)
    node["model"].setValue(min(previous, len(labels) - 1))
    node["kyven_status"].setValue("Model list refreshed.")


def start_server() -> None:
    nuke = _nuke()

    def start() -> None:
        try:
            ensure_server()
            nuke.executeInMainThread(nuke.message, args=("Kyven Server is ready.",))
        except Exception as exc:  # noqa: BLE001 - background boundary must report to the host UI
            nuke.executeInMainThread(nuke.message, args=(f"Kyven Server failed: {exc}",))

    threading.Thread(target=start, name="kyven-server-start", daemon=True).start()


def create_segment_node() -> Any:
    nuke = _nuke()
    selected = nuke.selectedNode() if nuke.selectedNodes() else None
    node = nuke.nodes.Group(name="KyvenSegment")
    node.setInput(0, selected)
    node["label"].setValue("[value kyven_status]")
    node.addKnob(nuke.Tab_Knob("kyven", "Kyven Segment"))
    node.addKnob(nuke.Enumeration_Knob("model", "Model", list(MODEL_LABELS)))
    node["model"].setValue(1)
    node.addKnob(nuke.Enumeration_Knob("profile", "Execution Profile", ["low_memory", "balanced", "quality"]))
    node["profile"].setValue(1)
    node.addKnob(nuke.PyScript_Knob("refresh_models", "Refresh Models", "kyven_nuke.node.refresh_models()"))
    node.addKnob(nuke.Boolean_Knob("positive_enabled", "Use Positive Point"))
    node["positive_enabled"].setValue(True)
    node.addKnob(nuke.XY_Knob("positive_point", "Positive Point"))
    node.addKnob(nuke.Boolean_Knob("negative_enabled", "Use Negative Point"))
    node.addKnob(nuke.XY_Knob("negative_point", "Negative Point"))
    node.addKnob(nuke.Boolean_Knob("box_enabled", "Use Box"))
    node.addKnob(nuke.BBox_Knob("prompt_box", "Prompt Box"))
    node.addKnob(nuke.PyScript_Knob("process_frame", "Process Current Frame", "kyven_nuke.node.process_current_frame()"))
    node.addKnob(nuke.PyScript_Knob("cancel", "Cancel", "kyven_nuke.node.cancel_current_job()"))
    status = nuke.String_Knob("kyven_status", "Status")
    status.setFlag(nuke.READ_ONLY)
    status.setValue("Ready")
    node.addKnob(status)
    internal_id = nuke.String_Knob("kyven_uuid", "UUID")
    internal_id.setValue(uuid.uuid4().hex)
    internal_id.setVisible(False)
    node.addKnob(internal_id)
    job_id = nuke.String_Knob("kyven_job_id", "Job ID")
    job_id.setVisible(False)
    node.addKnob(job_id)

    node.begin()
    try:
        source = nuke.nodes.Input(name="Source")
        source["number"].setValue(0)
        black = nuke.nodes.Constant(name="KyvenEmptyMatte")
        black["color"].setValue([0.0, 0.0, 0.0, 0.0])
        switch = nuke.nodes.Switch(name="KyvenMatteSwitch")
        switch.setInput(0, black)
        switch["which"].setValue(0)
        output = nuke.nodes.Output(name="Output")
        output.setInput(0, switch)
        writer = nuke.nodes.Write(name="KyvenSourceWrite")
        writer.setInput(0, source)
        writer["file_type"].setValue("png")
        writer["channels"].setValue("rgb")
    finally:
        node.end()
    return node
