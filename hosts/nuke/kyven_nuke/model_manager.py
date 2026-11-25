"""Nuke UI for installing and removing trusted Kyven models."""

from __future__ import annotations

import json
import os
import threading
import time

from kyven_nuke import config
from kyven_nuke.preferences import default_model_id, set_default_model
from kyven_nuke.runtime import (
    RuntimeInstallCancelled,
    ensure_model_runtime,
    ensure_server,
    stop_server,
)

_tasks = {}
_tasks_lock = threading.RLock()


def _nuke():
    import nuke

    return nuke


def _format_size(size_bytes):
    return f"{float(size_bytes) / (1024**2):.0f} MB"


def _manager_models():
    core = {"model_id": "kyven-core-runtime", "display_name": "Kyven Base Runtime",
            "task": "setup", "runtime": "core", "core_runtime": True,
            "installed": config.python_executable().is_file(), "size_bytes": 0}
    catalog = config.root() / "src/kyven/resources/models/catalog.json"
    local = json.loads(catalog.read_text(encoding="utf-8"))["models"] if catalog.is_file() else []
    runtimes = {item["model_id"]: item.get("runtime", "") for item in local}
    if config.python_executable().is_file():
        try:
            models = [dict(item, runtime=runtimes.get(item["model_id"], item.get("runtime", "")))
                      for item in ensure_server().models()]
            return [*models, core], True
        except Exception as exc:  # noqa: BLE001 - the offline catalog still supports setup
            core["setup_detail"] = str(exc)
    return [core, *[dict(model, installed=False, needs_core=True) for model in local]], False


def install_required_runtime(model_id):
    try:
        models, _ = _manager_models()
    except (OSError, ValueError) as exc:
        _nuke().message(f"Could not read the trusted model catalog:\n{exc}")
        return
    model = next((item for item in models if item["model_id"] == model_id), None)
    if model is None:
        _nuke().message("The selected model is not in the trusted catalog.")
        return
    _start_operation("runtime", model)


def _choice_label(model):
    if model.get("core_runtime"):
        return "Setup | Kyven Base Runtime | install or verify base components"
    state = "installed" if model.get("installed") else "not installed"
    warning = " | VRAM warning" if model.get("compatible") is False else ""
    default = (
        " | default for new nodes"
        if default_model_id(str(model.get("task", ""))) == str(model.get("model_id", ""))
        else ""
    )
    if model.get("local_runtime"):
        return f"DLSS | {model['display_name']} | {model.get('status_message', state)}"
    return (
        f"{str(model.get('task', '')).replace('_', ' ').title()} | {model['display_name']} | "
        f"{_format_size(model['size_bytes'])} | {state}{warning}{default}"
    )


def show_model_manager():
    """Show one safe catalog action at a time in a native modal panel."""

    nuke = _nuke()
    try:
        models, server_ready = _manager_models()
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
        "{Install Selected} {Install Required Components} {Import NVIDIA Component} {Remove Selected} {Set as Default} {Open Models Folder}",
    )
    panel.addEnumerationPulldown(
        "Model",
        " ".join("{" + label.replace("}", "") + "}" for label in labels),
    )
    panel.addNotepad(
        "Safety",
        ("Server setup is not complete. Installing a model also installs its required components. "
         "A private Python is downloaded automatically if needed. DLSS appears after base setup.\n\n" if not server_ready else "") +
        "Downloads come only from the trusted Kyven catalog. Single-file models are verified "
        "by exact size and SHA-256; repository models are pinned to an audited revision. "
        "Required model runtimes are installed automatically into Kyven's private environment. "
        "DLSS installs Kyven's portable worker and its bundled NVIDIA component locally, with "
        "size/SHA-256 verification. OpenEXR is installed into Kyven's private environment when needed. "
        "A distribution without the bundled component can use the pinned community download.",
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
    if model.get("core_runtime"):
        if action in {"Install Selected", "Install Required Components"}:
            _start_operation("runtime", model)
        else:
            nuke.message("The base runtime supports all Kyven nodes. Use Install Selected to install or verify it.")
        return
    if action == "Install Required Components":
        _start_operation("runtime", model)
        return
    if action == "Import NVIDIA Component":
        if not model.get("local_runtime"):
            nuke.message("Import NVIDIA Component applies to the DLSS entry.")
            return
        source = nuke.getFilename("Choose licensed nvngx_dlssnr.dll", "nvngx_dlssnr.dll")
        if source:
            _start_operation("download", model, source=str(source))
        return
    if action == "Set as Default":
        if model.get("local_runtime"):
            nuke.message("DLSS automatically uses the runtime selected in Model Manager.")
            return
        task = str(model.get("task", ""))
        try:
            set_default_model(task, str(model["model_id"]))
        except (OSError, ValueError) as exc:
            nuke.message(f"Could not save the default model:\n{exc}")
            return
        nuke.message(
            f"{model['display_name']} is now the default for new {task.title()} nodes.\n\n"
            "Existing nodes were not changed."
        )
        return
    if action == "Install Selected":
        if model.get("installed") and not model.get("runtime"):
            nuke.message(f"{model['display_name']} is already installed.")
            return
        if model.get("local_runtime") and model.get("license_acceptance_required"):
            if not nuke.ask(
                "Install the DLSS NVIDIA component from the pinned community release?\n\n"
                f"Source: {model.get('source_url', '')}\n\n"
                f"Terms: {model.get('license', '')}\n{model.get('license_url', '')}\n\n"
                "Kyven's worker is built separately. Only the NVIDIA NR DLL is extracted. "
                "This download is not presented as an official NVIDIA SDK or a redistribution grant.\n\n"
                "Continue if you have reviewed the source and applicable terms."):
                return
        elif model.get("license_acceptance_required") and not nuke.ask(
            f"Install {model['display_name']}?\n\n"
            f"License: {model.get('license', 'See model card')}\n" +
            ("Commercial permission for these weights is not confirmed. This is an experimental entry.\n\n"
             if not model.get("commercial_use") else
             "Commercial use is conditional on following the model license and use restrictions.\n\n") +
            f"Model card: {model.get('license_url', '')}\n\n"
            "Continue only if you reviewed and accept these terms."
        ):
            return
        _start_operation("download", model)
        return
    if not model.get("installed"):
        nuke.message(f"{model['display_name']} is not installed.")
        return
    if model.get("local_runtime"):
        _start_operation("remove", model)
        return
    if not nuke.ask(
        f"Remove this Kyven model?\n\n{model['display_name']}\n"
        f"{_format_size(model['size_bytes'])}\n\nIt can be installed again later."
    ):
        return
    _start_operation("remove", model)


def _start_operation(action, model, source=None):
    nuke = _nuke()
    key = str(model["model_id"])
    with _tasks_lock:
        if key in _tasks:
            nuke.message(f"An operation for {model['display_name']} is already running.")
            return
        # Reserve before creating UI or starting a thread so rapid repeated actions cannot race.
        _tasks[key] = None
    title = "Install Components" if action == "runtime" else "Install" if action == "download" else "Remove"
    try:
        task = nuke.ProgressTask(f"Kyven Models | {title} {model['display_name']}")
        task.setMessage("Submitting model operation")
        task.setProgress(0)
        with _tasks_lock:
            _tasks[key] = task
    except Exception:
        with _tasks_lock:
            _tasks.pop(key, None)
        raise

    def work():
        operation_id = ""
        try:
            runtime_changed = False
            if action in {"download", "runtime"}:
                if model.get("needs_core"):
                    ensure_model_runtime("core", progress_callback=lambda message:
                        nuke.executeInMainThread(_update_task, args=(key, 2, message)),
                        cancel_requested=lambda: nuke.executeInMainThreadWithResult(_task_cancelled, args=(key,)))
                runtime_changed = ensure_model_runtime(
                    str(model.get("runtime") or ""),
                    progress_callback=lambda message: nuke.executeInMainThread(
                        _update_task, args=(key, 5, message)
                    ),
                    cancel_requested=lambda: nuke.executeInMainThreadWithResult(
                        _task_cancelled, args=(key,)
                    ),
                )
                if runtime_changed:
                    nuke.executeInMainThread(
                        _update_task, args=(key, 10, "Restarting Kyven Server with the new runtime")
                    )
                    stop_server()
            if action == "runtime":
                nuke.executeInMainThread(_finish_runtime, args=(key, model))
                return
            if nuke.executeInMainThreadWithResult(_task_cancelled, args=(key,)):
                raise RuntimeInstallCancelled("Installation cancelled before downloading model weights.")
            client = ensure_server()
            operation_id = (
                (client.start_model_download(key, source=source) if source else
                 client.start_model_download(key, license_accepted=True) if model.get("local_runtime") else
                 client.start_model_download(key))
                if action == "download"
                else client.start_model_remove(key)
            )
            while True:
                operation = client.model_operation(operation_id)
                nuke.executeInMainThread(
                    _update_task,
                    args=(
                        key,
                        int(float(operation.get("progress") or 0) * 100),
                        str(operation.get("message") or "Working"),
                    ),
                )
                cancelled = nuke.executeInMainThreadWithResult(_task_cancelled, args=(key,))
                if cancelled and operation["status"] not in ("succeeded", "failed", "cancelled"):
                    client.cancel_model_operation(operation_id)
                if operation["status"] in ("succeeded", "failed", "cancelled"):
                    break
                time.sleep(0.2)
            nuke.executeInMainThread(
                _finish_operation, args=(key, model, operation, runtime_changed)
            )
        except RuntimeInstallCancelled:
            nuke.executeInMainThread(_finish_cancelled, args=(key,))
        except Exception as exc:  # noqa: BLE001
            nuke.executeInMainThread(_finish_error, args=(key, str(exc)))

    threading.Thread(target=work, name=f"kyven-model-{action}", daemon=True).start()


def _update_task(key, percent, message):
    with _tasks_lock:
        task = _tasks.get(key)
    if task is not None:
        task.setMessage(message)
        task.setProgress(max(0, min(100, int(percent))))


def _task_cancelled(key):
    with _tasks_lock:
        task = _tasks.get(key)
    return bool(task is not None and task.isCancelled())


def _finish_runtime(key, model):
    with _tasks_lock:
        _tasks.pop(key, None)
    _refresh_all_nodes()
    _nuke().message(f"Required components for {model['display_name']} are ready.")
    if model.get("core_runtime"):
        show_model_manager()


def _finish_operation(key, model, operation, runtime_changed=False):
    with _tasks_lock:
        _tasks.pop(key, None)
    nuke = _nuke()
    if operation["status"] == "succeeded":
        _refresh_all_nodes()
        verb = "installed" if operation["action"] == "download" else "removed"
        runtime_note = "\nRequired runtime was installed automatically." if runtime_changed else ""
        nuke.message(f"{model['display_name']} was {verb} successfully.{runtime_note}")
        return
    error = operation.get("error") or {}
    message = str(error.get("message") or operation.get("message") or "Unknown error")
    detail = str(error.get("technical_detail") or "").strip()
    suggested = str(error.get("suggested_action") or "").strip()
    lines = [f"Model operation {operation['status']}: {message}"]
    if detail and detail not in message:
        lines.extend(("", f"Technical detail: {detail}"))
    if suggested:
        lines.extend(("", suggested))
    nuke.message("\n".join(lines))


def _finish_error(key, message):
    with _tasks_lock:
        _tasks.pop(key, None)
    _nuke().message(f"Kyven Model Manager failed:\n{message}")


def _finish_cancelled(key):
    with _tasks_lock:
        _tasks.pop(key, None)
    _nuke().message("Kyven model installation was cancelled.")


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
        if "qwen_system_prompt" in node.knobs():
            # This single-model node must never receive the LaMa model list.
            from kyven_nuke.generative_fill_node import _ensure_fill_graph
            _ensure_fill_graph(node)
            if "kyven_status" in node.knobs():
                node["kyven_status"].setValue("Generative Fill components and model catalog refreshed.")
            continue
        if "kyven_kind" in node.knobs() and node["kyven_kind"].value() == "dlss":
            from kyven_nuke.dlss_node import refresh_runtime

            refresh_runtime(node)
            continue
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
