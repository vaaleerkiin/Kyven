"""Nuke UI for installing and removing trusted Kyven models."""

from __future__ import annotations

import os
import threading
import time

from kyven_nuke import config
from kyven_nuke.runtime import ensure_server

_tasks = {}


def _nuke():
    import nuke

    return nuke


def _format_size(size_bytes):
    return f"{float(size_bytes) / (1024 ** 2):.0f} MB"


def _choice_label(model):
    state = "installed" if model.get("installed") else "not installed"
    warning = " | VRAM warning" if model.get("compatible") is False else ""
    return (
        f"{str(model.get('task', '')).replace('_', ' ').title()} | {model['display_name']} | "
        f"{_format_size(model['size_bytes'])} | {state}{warning}"
    )


def show_model_manager():
    """Show one safe catalog action at a time in a native modal panel."""

    nuke = _nuke()
    try:
        models = ensure_server().models()
    except Exception as exc:  # noqa: BLE001
        nuke.message(f"Could not open Kyven Model Manager:\n{exc}")
        return
    labels = [_choice_label(model) for model in models]
    if not labels:
        nuke.message("The Kyven trusted model catalog is empty.")
        return
    panel = nuke.Panel("Kyven Model Manager")
    panel.addEnumerationPulldown(
        "Action",
        "{Install Selected} {Remove Selected} {Open Models Folder}",
    )
    panel.addEnumerationPulldown(
        "Model",
        " ".join("{" + label.replace("}", "") + "}" for label in labels),
    )
    panel.addNotepad(
        "Safety",
        "Downloads come only from the trusted Kyven catalog. Single-file models are verified "
        "by exact size and SHA-256; repository models are pinned to an audited revision. "
        "All model files stay inside the repository.",
    )
    if not panel.show():
        return
    action = str(panel.value("Action"))
    if action == "Open Models Folder":
        path = config.models_dir()
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    selected_label = str(panel.value("Model"))
    try:
        index = labels.index(selected_label)
    except ValueError:
        nuke.message("The selected catalog model could not be resolved.")
        return
    model = models[index]
    if action == "Install Selected":
        if model.get("installed"):
            nuke.message(f"{model['display_name']} is already installed.")
            return
        if model.get("license_acceptance_required") and not nuke.ask(
            f"Install {model['display_name']}?\n\n"
            f"License: {model.get('license', 'See model card')}\n"
            "Commercial use is conditional on following the model license and use restrictions.\n\n"
            f"Model card: {model.get('license_url', '')}\n\n"
            "Continue only if you reviewed and accept these terms."
        ):
            return
        _start_operation("download", model)
        return
    if not model.get("installed"):
        nuke.message(f"{model['display_name']} is not installed.")
        return
    if not nuke.ask(
        f"Remove this Kyven model?\n\n{model['display_name']}\n"
        f"{_format_size(model['size_bytes'])}\n\nIt can be installed again later."
    ):
        return
    _start_operation("remove", model)


def _start_operation(action, model):
    nuke = _nuke()
    key = str(model["model_id"])
    title = "Install" if action == "download" else "Remove"
    task = nuke.ProgressTask(f"Kyven Models | {title} {model['display_name']}")
    task.setMessage("Submitting model operation")
    task.setProgress(0)
    _tasks[key] = task

    def work():
        operation_id = ""
        try:
            client = ensure_server()
            operation_id = (
                client.start_model_download(key)
                if action == "download"
                else client.start_model_remove(key)
            )
            while True:
                operation = client.model_operation(operation_id)
                nuke.executeInMainThread(
                    _update_task,
                    args=(key, int(float(operation.get("progress") or 0) * 100), str(operation.get("message") or "Working")),
                )
                cancelled = nuke.executeInMainThreadWithResult(_task_cancelled, args=(key,))
                if cancelled and operation["status"] not in ("succeeded", "failed", "cancelled"):
                    client.cancel_model_operation(operation_id)
                if operation["status"] in ("succeeded", "failed", "cancelled"):
                    break
                time.sleep(0.2)
            nuke.executeInMainThread(_finish_operation, args=(key, model, operation))
        except Exception as exc:  # noqa: BLE001
            nuke.executeInMainThread(_finish_error, args=(key, str(exc)))

    threading.Thread(target=work, name=f"kyven-model-{action}", daemon=True).start()


def _update_task(key, percent, message):
    task = _tasks.get(key)
    if task is not None:
        task.setMessage(message)
        task.setProgress(max(0, min(100, int(percent))))


def _task_cancelled(key):
    task = _tasks.get(key)
    return bool(task is not None and task.isCancelled())


def _finish_operation(key, model, operation):
    _tasks.pop(key, None)
    nuke = _nuke()
    if operation["status"] == "succeeded":
        _refresh_all_nodes()
        verb = "installed" if operation["action"] == "download" else "removed"
        nuke.message(f"{model['display_name']} was {verb} successfully.")
        return
    error = operation.get("error") or {}
    nuke.message(f"Model operation {operation['status']}: {error.get('message', operation.get('message', 'Unknown error'))}")


def _finish_error(key, message):
    _tasks.pop(key, None)
    _nuke().message(f"Kyven Model Manager failed:\n{message}")


def _refresh_all_nodes():
    nuke = _nuke()
    try:
        models = ensure_server().models()
    except Exception:  # noqa: BLE001 - refresh failure should not hide a successful install
        return
    by_task = {
        task: [model for model in models if model.get("task") == task]
        for task in ("segment", "refine", "inpaint")
    }
    for node in nuke.allNodes("Group"):
        if "model" not in node.knobs():
            continue
        kind = str(node["kyven_kind"].value()) if "kyven_kind" in node.knobs() else "segment"
        labels = [_choice_label(model) for model in by_task.get(kind, [])]
        if labels:
            previous = int(node["model"].getValue())
            node["model"].setValues(labels)
            node["model"].setValue(min(previous, len(labels) - 1))
            if "kyven_status" in node.knobs():
                node["kyven_status"].setValue("Model catalog refreshed.")
