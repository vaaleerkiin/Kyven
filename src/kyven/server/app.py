"""Loopback-only JSON HTTP server for host adapters."""

from __future__ import annotations

import hmac
import json
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from kyven.errors import ErrorCode, KyvenError
from kyven.models.catalog import ModelCatalog
from kyven.models.operations import ModelOperationManager
from kyven.preview import (
    postprocess_mask_preview,
    prepare_inpaint_mask_preview,
    prepare_trimap_preview,
)
from kyven.segment.providers.registry import ProviderRegistry
from kyven.server.jobs import JobManager

MAX_REQUEST_BYTES = 1024 * 1024
SERVER_API_VERSION = 21


@dataclass(frozen=True, slots=True)
class ServerConfig:
    token: str
    models_dir: Path
    port: int = 8765
    available_vram_mb: int | None = None

    def __post_init__(self) -> None:
        if len(self.token) < 32:
            raise ValueError("The server token must contain at least 32 characters.")
        if not 0 <= self.port <= 65535:
            raise ValueError("The server port is outside the valid range.")


def _handler_type(
    manager: JobManager,
    registry: ProviderRegistry,
    catalog: ModelCatalog,
    config: ServerConfig,
    model_operations: ModelOperationManager,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "Kyven/0.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _authorized(self) -> bool:
            supplied = self.headers.get("Authorization", "")
            return hmac.compare_digest(supplied, f"Bearer {config.token}")

        def _send(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                return

        def _error(self, error: KyvenError) -> None:
            status = {
                ErrorCode.AUTHENTICATION_FAILED: HTTPStatus.UNAUTHORIZED,
                ErrorCode.JOB_NOT_FOUND: HTTPStatus.NOT_FOUND,
                ErrorCode.INVALID_REQUEST: HTTPStatus.BAD_REQUEST,
                ErrorCode.MODEL_NOT_FOUND: HTTPStatus.BAD_REQUEST,
            }.get(error.code, HTTPStatus.INTERNAL_SERVER_ERROR)
            self._send(status, {"error": error.to_dict()})

        def _require_auth(self) -> bool:
            if self._authorized():
                return True
            self._error(
                KyvenError(
                    code=ErrorCode.AUTHENTICATION_FAILED,
                    message="A valid Kyven server token is required.",
                )
            )
            return False

        def _json_body(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise KyvenError(
                    code=ErrorCode.INVALID_REQUEST,
                    message="Invalid Content-Length header.",
                ) from exc
            if length < 0 or length > MAX_REQUEST_BYTES:
                raise KyvenError(
                    code=ErrorCode.INVALID_REQUEST,
                    message="The request body is too large.",
                )
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise KyvenError(
                    code=ErrorCode.INVALID_REQUEST,
                    message="The request body must be valid UTF-8 JSON.",
                    technical_detail=str(exc),
                ) from exc
            if not isinstance(payload, dict):
                raise KyvenError(
                    code=ErrorCode.INVALID_REQUEST,
                    message="The JSON request body must be an object.",
                )
            return payload

        def do_GET(self) -> None:
            if not self._require_auth():
                return
            path = urlparse(self.path).path
            try:
                if path == "/v1/health":
                    self._send(
                        HTTPStatus.OK,
                        {
                            "status": "ok",
                            "service": "kyven",
                            "api_version": SERVER_API_VERSION,
                        },
                    )
                    return
                if path == "/v1/models":
                    self._send(
                        HTTPStatus.OK,
                        {
                            "models": [
                                spec.snapshot(config.models_dir, config.available_vram_mb)
                                for spec in catalog.list()
                            ]
                        },
                    )
                    return
                if path.startswith("/v1/model-operations/"):
                    operation_id = path.removeprefix("/v1/model-operations/")
                    self._send(HTTPStatus.OK, model_operations.get(operation_id))
                    return
                if path.startswith("/v1/jobs/"):
                    job_id = path.removeprefix("/v1/jobs/")
                    self._send(HTTPStatus.OK, manager.get(job_id))
                    return
                self._send(HTTPStatus.NOT_FOUND, {"error": {"message": "Not found."}})
            except KyvenError as exc:
                self._error(exc)

        def do_POST(self) -> None:
            if not self._require_auth():
                return
            path = urlparse(self.path).path
            try:
                payload = self._json_body()
                if path == "/v1/jobs/segment":
                    model_id = str(payload.get("model_id", "sam2.1-small"))
                    catalog.get(model_id)
                    job_id = manager.submit_segment(payload)
                    self._send(HTTPStatus.ACCEPTED, {"job_id": job_id, "status": "queued"})
                    return
                if path == "/v1/jobs/segment-video":
                    model_id = str(payload.get("model_id", "sam2.1-small"))
                    catalog.get(model_id)
                    job_id = manager.submit_video(payload)
                    self._send(HTTPStatus.ACCEPTED, {"job_id": job_id, "status": "queued"})
                    return
                if path == "/v1/jobs/refine":
                    model_id = str(payload.get("model_id", "vitmatte-small-composition-1k"))
                    if catalog.get(model_id).task != "refine":
                        raise KyvenError(
                            code=ErrorCode.INVALID_REQUEST,
                            message="The selected model is not a refinement model.",
                        )
                    job_id = manager.submit_refine(payload)
                    self._send(HTTPStatus.ACCEPTED, {"job_id": job_id, "status": "queued"})
                    return
                if path == "/v1/jobs/inpaint":
                    model_id = str(payload.get("model_id", "lama-2025jan-onnx"))
                    if catalog.get(model_id).task != "inpaint":
                        raise KyvenError(ErrorCode.INVALID_REQUEST, "The selected model is not an inpaint model.")
                    job_id = manager.submit_inpaint(payload)
                    self._send(HTTPStatus.ACCEPTED, {"job_id": job_id, "status": "queued"})
                    return
                if path == "/v1/jobs/generative-inpaint":
                    model_id = str(payload.get("model_id", "sdxl-inpainting-1.0"))
                    if catalog.get(model_id).task != "generative_inpaint":
                        raise KyvenError(
                            ErrorCode.INVALID_REQUEST,
                            "The selected model is not a generative inpaint model.",
                        )
                    job_id = manager.submit_generative_inpaint(payload)
                    self._send(HTTPStatus.ACCEPTED, {"job_id": job_id, "status": "queued"})
                    return
                if path == "/v1/models/download":
                    operation_id = model_operations.submit("download", str(payload.get("model_id", "")))
                    self._send(HTTPStatus.ACCEPTED, {"operation_id": operation_id})
                    return
                if path == "/v1/models/remove":
                    operation_id = model_operations.submit("remove", str(payload.get("model_id", "")))
                    self._send(HTTPStatus.ACCEPTED, {"operation_id": operation_id})
                    return
                if path.startswith("/v1/model-operations/") and path.endswith("/cancel"):
                    operation_id = path.removeprefix("/v1/model-operations/").removesuffix("/cancel")
                    self._send(HTTPStatus.OK, model_operations.cancel(operation_id))
                    return
                if path == "/v1/preview/trimap":
                    self._send(HTTPStatus.OK, prepare_trimap_preview(payload))
                    return
                if path == "/v1/preview/mask-postprocess":
                    self._send(HTTPStatus.OK, postprocess_mask_preview(payload))
                    return
                if path == "/v1/preview/inpaint-mask":
                    self._send(HTTPStatus.OK, prepare_inpaint_mask_preview(payload))
                    return
                if path.startswith("/v1/jobs/") and path.endswith("/cancel"):
                    job_id = path.removeprefix("/v1/jobs/").removesuffix("/cancel")
                    self._send(HTTPStatus.OK, manager.cancel(job_id))
                    return
                if path == "/v1/providers/unload-all":
                    manager.unload_all(registry)
                    self._send(HTTPStatus.OK, {"status": "unloaded"})
                    return
                if path == "/v1/server/shutdown":
                    self._send(HTTPStatus.ACCEPTED, {"status": "stopping"})
                    threading.Thread(
                        target=self.server.shutdown,
                        name="kyven-http-shutdown",
                        daemon=True,
                    ).start()
                    return
                self._send(HTTPStatus.NOT_FOUND, {"error": {"message": "Not found."}})
            except KyvenError as exc:
                self._error(exc)

    return Handler


class KyvenServer:
    """Own the loopback HTTP server and GPU job manager."""

    def __init__(
        self,
        config: ServerConfig,
        manager: JobManager,
        registry: ProviderRegistry,
        catalog: ModelCatalog,
    ) -> None:
        self._manager = manager
        self._model_operations = ModelOperationManager(
            catalog,
            config.models_dir,
            registry,
            unload_provider=lambda provider_id: manager.unload_provider(registry, provider_id),
        )
        self._httpd = ThreadingHTTPServer(
            ("127.0.0.1", config.port),
            _handler_type(manager, registry, catalog, config, self._model_operations),
        )
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return int(self._httpd.server_address[1])

    def serve_forever(self) -> None:
        self._httpd.serve_forever(poll_interval=0.2)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Kyven server is already running.")
        self._thread = threading.Thread(target=self.serve_forever, name="kyven-http", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._httpd.shutdown()
        self.close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def close(self) -> None:
        self._httpd.server_close()
        self._model_operations.shutdown()
        self._manager.shutdown()
