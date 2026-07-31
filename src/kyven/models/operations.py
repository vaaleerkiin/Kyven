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
        self._lock = threading.RLock()

    def submit(self, action: str, model_id: str) -> str:
        self._catalog.get(model_id)
        if action not in {"download", "remove"}:
            raise KyvenError(ErrorCode.INVALID_REQUEST, f"Unknown model action: {action}")
        operation = ModelOperation(uuid.uuid4().hex, action, model_id)
        with self._lock:
            self._operations[operation.operation_id] = operation
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
            with self._lock:
                operation.status = "failed"
                operation.message = "Model operation failed"
                operation.error = KyvenError(
                    ErrorCode.SERVER_ERROR,
                    "Model operation failed unexpectedly.",
                    technical_detail=str(exc),
                    recoverable=True,
                ).to_dict()
        finally:
            with self._lock:
                operation.finished_at = time.time()

    def _download(self, operation: ModelOperation) -> None:
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

    def _remove(self, operation: ModelOperation) -> None:
        with self._lock:
            operation.progress = 0.25
            operation.message = "Unloading model"
        self._unload_provider(operation.model_id)
        self._catalog.remove(operation.model_id, self._models_dir)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
