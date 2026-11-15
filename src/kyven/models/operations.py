"""Asynchronous trusted-model installation and removal operations."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kyven.dlss.runtime import (
    MODEL_ID,
    bind_runtime,
    bundled_vendor_available,
    runtime_root,
    unbind_runtime,
)
from kyven.errors import ErrorCode, KyvenError
from kyven.models.catalog import ModelCatalog
from kyven.segment.providers.registry import ProviderRegistry


@dataclass(slots=True)
class ModelOperation:
    operation_id: str
    action: str
    model_id: str
    status: str = "queued"
    progress: float = 0.0
    message: str = "Queued"
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: dict[str, Any] | None = None
    source: str = ""
    license_accepted: bool = False
    cancelled: threading.Event = field(default_factory=threading.Event)

    def snapshot(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "action": self.action,
            "model_id": self.model_id,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }


class ModelOperationManager:
    """Run model file mutations outside host and HTTP threads."""

    def __init__(
        self,
        catalog: ModelCatalog,
        models_dir: Path,
        registry: ProviderRegistry,
        unload_provider: Callable[[str], None] | None = None,
    ) -> None:
        self._catalog = catalog
        self._models_dir = models_dir
        self._registry = registry
        self._unload_provider = unload_provider or registry.unload
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kyven-models")
        self._operations: dict[str, ModelOperation] = {}
        self._active: dict[tuple[str, str], str] = {}
        self._history_limit = 100
        self._lock = threading.RLock()

    def submit(self, action: str, model_id: str, source: str | None = None,
               license_accepted: bool = False) -> str:
        if model_id != MODEL_ID:
            self._catalog.get(model_id)
        if source is not None and (model_id != MODEL_ID or not isinstance(source, str)
                                   or not Path(source).is_absolute()):
            raise KyvenError(ErrorCode.INVALID_REQUEST, "Local runtime source must be an absolute path.")
        if (model_id == MODEL_ID and action == "download" and source is None
                and license_accepted is not True and not bundled_vendor_available()):
            raise KyvenError(ErrorCode.INVALID_REQUEST,
                             "Review the DLSS community component source and terms in Model Manager first.")
        if action not in {"download", "remove"}:
            raise KyvenError(ErrorCode.INVALID_REQUEST, f"Unknown model action: {action}")
        key = (action, model_id)
        with self._lock:
            active_id = self._active.get(key)
            if active_id is not None:
                active = self._operations.get(active_id)
                if active is not None and active.status in {"queued", "running"}:
                    return active_id
                self._active.pop(key, None)
            finished = sorted(
                (item for item in self._operations.values() if item.finished_at is not None),
                key=lambda item: item.finished_at or item.created_at,
            )
            for item in finished[: max(0, len(self._operations) - self._history_limit + 1)]:
                self._operations.pop(item.operation_id, None)
            operation = ModelOperation(uuid.uuid4().hex, action, model_id)
            operation.source = source or ""
            operation.license_accepted = license_accepted
            self._operations[operation.operation_id] = operation
            self._active[key] = operation.operation_id
        self._executor.submit(self._run, operation)
        return operation.operation_id

    def get(self, operation_id: str) -> dict[str, Any]:
        with self._lock:
            operation = self._operations.get(operation_id)
            if operation is None:
                raise KyvenError(ErrorCode.JOB_NOT_FOUND, f"Model operation was not found: {operation_id}")
            return operation.snapshot()

    def cancel(self, operation_id: str) -> dict[str, Any]:
        with self._lock:
            operation = self._operations.get(operation_id)
            if operation is None:
                raise KyvenError(ErrorCode.JOB_NOT_FOUND, f"Model operation was not found: {operation_id}")
            operation.cancelled.set()
            if operation.status == "queued":
                operation.status = "cancelled"
                operation.message = "Cancelled"
                operation.finished_at = time.time()
                self._active.pop((operation.action, operation.model_id), None)
            return operation.snapshot()

    def _run(self, operation: ModelOperation) -> None:
        with self._lock:
            if operation.cancelled.is_set():
                return
            operation.status = "running"
            operation.message = "Preparing model operation"
        try:
            if operation.action == "download":
                self._download(operation)
            else:
                self._remove(operation)
            with self._lock:
                operation.status = "succeeded"
                operation.progress = 1.0
                operation.message = "Model installed" if operation.action == "download" else "Model removed"
        except KyvenError as exc:
            with self._lock:
                operation.status = "cancelled" if exc.code is ErrorCode.CANCELLED else "failed"
                operation.message = exc.message
                operation.error = exc.to_dict()
        except Exception as exc:  # noqa: BLE001
            detail = str(exc)
            suggested_action = (
                "Allow Kyven's Python runtime through Windows Firewall, then retry the download."
                if "WinError 10013" in detail
                else "Retry the operation; if it fails again, review the technical detail."
            )
            with self._lock:
                operation.status = "failed"
                operation.message = "Model operation failed"
                operation.error = KyvenError(
                    ErrorCode.SERVER_ERROR,
                    "Model operation failed unexpectedly.",
                    technical_detail=detail,
                    recoverable=True,
                    suggested_action=suggested_action,
                ).to_dict()
        finally:
            with self._lock:
                operation.finished_at = time.time()
                key = (operation.action, operation.model_id)
                if self._active.get(key) == operation.operation_id:
                    self._active.pop(key, None)

    def _download(self, operation: ModelOperation) -> None:
        if operation.model_id == MODEL_ID:
            self._unload_provider(MODEL_ID)
            bind_runtime(runtime_root(self._models_dir),
                         Path(operation.source) if operation.source else None,
                         cancelled=operation.cancelled.is_set,
                         download_vendor=operation.license_accepted,
                         progress=lambda downloaded, total: self._report_runtime_download(
                             operation, downloaded, total))
            return
        def report(downloaded: int, total: int) -> None:
            with self._lock:
                operation.progress = min(0.95, downloaded / max(1, total) * 0.95)
                operation.message = f"Downloading {downloaded / (1024**2):.0f} / {total / (1024**2):.0f} MB"

        self._catalog.download(
            operation.model_id,
            self._models_dir,
            progress=report,
            cancelled=operation.cancelled.is_set,
        )
        with self._lock:
            operation.progress = 0.98
            operation.message = "Checksum verified"

    def _report_runtime_download(self, operation: ModelOperation, downloaded: int, total: int) -> None:
        with self._lock:
            operation.progress = min(0.95, downloaded / max(1, total) * 0.95)
            operation.message = f"Downloading DLSS component {downloaded / 1024**2:.0f}/{total / 1024**2:.0f} MB"

    def _remove(self, operation: ModelOperation) -> None:
        with self._lock:
            operation.progress = 0.25
            operation.message = "Unloading model"
        self._unload_provider(operation.model_id)
        if operation.model_id == MODEL_ID:
            unbind_runtime(runtime_root(self._models_dir))
            return
        self._catalog.remove(operation.model_id, self._models_dir)

    def shutdown(self) -> None:
        with self._lock:
            for operation in self._operations.values():
                if operation.status in {"queued", "running"}:
                    operation.cancelled.set()
        self._executor.shutdown(wait=True, cancel_futures=True)
