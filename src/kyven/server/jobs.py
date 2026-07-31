"""Single-GPU asynchronous job scheduling."""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from kyven.cancellation import CancellationToken
from kyven.errors import ErrorCode, KyvenError
from kyven.inpaint.models import InpaintRequest
from kyven.inpaint.service import InpaintService
from kyven.refine.models import RefineRequest
from kyven.refine.service import RefineService
from kyven.segment.models import (
    BoxPrompt,
    ExecutionProfile,
    PointLabel,
    PointPrompt,
    SegmentRequest,
)
from kyven.segment.providers.registry import ProviderRegistry
from kyven.segment.service import SegmentService
from kyven.segment.video import (
    VideoDirection,
    VideoSegmentRequest,
    VideoSegmentService,
)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class JobRecord:
    job_id: str
    request: SegmentRequest | VideoSegmentRequest | RefineRequest | InpaintRequest
    cancellation: CancellationToken = field(default_factory=CancellationToken)
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    def snapshot(self) -> dict[str, Any]:
        progress, progress_message = self.cancellation.progress_snapshot()
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
            "progress": progress,
            "progress_message": progress_message,
        }


class JobManager:
    """Serialize GPU work while keeping HTTP and host UIs responsive."""

    def __init__(
        self,
        service: SegmentService,
        video_service: VideoSegmentService | None = None,
        refine_service: RefineService | None = None,
        inpaint_service: InpaintService | None = None,
    ) -> None:
        self._service = service
        self._video_service = video_service
        self._refine_service = refine_service
        self._inpaint_service = inpaint_service
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kyven-gpu")
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _box_from_payload(payload: dict[str, Any], field: str) -> BoxPrompt | None:
        value = payload.get(field)
        if value is None:
            return None
        return BoxPrompt(
            float(value["x0"]),
            float(value["y0"]),
            float(value["x1"]),
            float(value["y1"]),
        )

    @staticmethod
    def request_from_payload(payload: dict[str, Any]) -> SegmentRequest:
        source = Path(str(payload["source"]))
        output = Path(str(payload["output"]))
        raw_output_value = payload.get("raw_output")
        raw_output = Path(str(raw_output_value)) if raw_output_value else None
        if (
            not source.is_absolute()
            or not output.is_absolute()
            or (raw_output is not None and not raw_output.is_absolute())
        ):
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="Server job paths must be absolute.",
                suggested_action="Resolve source and output paths in the host adapter.",
            )
        points = tuple(
            PointPrompt(
                x=float(item["x"]),
                y=float(item["y"]),
                label=PointLabel(str(item.get("label", "positive"))),
            )
            for item in payload.get("points", [])
        )
        box = JobManager._box_from_payload(payload, "box")
        roi = JobManager._box_from_payload(payload, "roi")
        return SegmentRequest(
            source=source,
            output=output,
            raw_output=raw_output,
            points=points,
            box=box,
            roi=roi,
            provider_id=str(payload.get("model_id", "sam2.1-small")),
            profile=ExecutionProfile(str(payload.get("profile", "balanced"))),
            multimask_output=bool(payload.get("multimask_output", True)),
            fill_holes=bool(payload.get("fill_holes", True)),
            max_hole_area=int(payload.get("max_hole_area", 2_048)),
        )

    def submit_segment(self, payload: dict[str, Any]) -> str:
        try:
            request = self.request_from_payload(payload)
            request.validate()
        except (KeyError, TypeError, ValueError) as exc:
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="The segment job payload is invalid.",
                technical_detail=str(exc),
                suggested_action="Check source, output, points, box, ROI, model, and profile fields.",
            ) from exc
        record = JobRecord(job_id=uuid.uuid4().hex, request=request)
        with self._lock:
            self._jobs[record.job_id] = record
        self._executor.submit(self._run, record)
        return record.job_id

    @staticmethod
    def video_request_from_payload(payload: dict[str, Any]) -> VideoSegmentRequest:
        frames_dir = Path(str(payload["frames_dir"]))
        output_pattern = Path(str(payload["output_pattern"]))
        raw_output_pattern_value = payload.get("raw_output_pattern")
        raw_output_pattern = (
            Path(str(raw_output_pattern_value)) if raw_output_pattern_value else None
        )
        if (
            not frames_dir.is_absolute()
            or not output_pattern.is_absolute()
            or (raw_output_pattern is not None and not raw_output_pattern.is_absolute())
        ):
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="Video job paths must be absolute.",
            )
        points = tuple(
            PointPrompt(
                x=float(item["x"]),
                y=float(item["y"]),
                label=PointLabel(str(item.get("label", "positive"))),
            )
            for item in payload.get("points", [])
        )
        box = JobManager._box_from_payload(payload, "box")
        roi = JobManager._box_from_payload(payload, "roi")
        rois = tuple(
            (
                int(item["frame"]),
                BoxPrompt(
                    float(item["x0"]),
                    float(item["y0"]),
                    float(item["x1"]),
                    float(item["y1"]),
                ),
            )
            for item in payload.get("rois", [])
        )
        return VideoSegmentRequest(
            frames_dir=frames_dir,
            output_pattern=output_pattern,
            first_frame=int(payload["first_frame"]),
            last_frame=int(payload["last_frame"]),
            key_frame=int(payload["key_frame"]),
            direction=VideoDirection(str(payload.get("direction", "both"))),
            points=points,
            box=box,
            roi=roi,
            rois=rois,
            provider_id=str(payload.get("model_id", "sam2.1-small")),
            profile=ExecutionProfile(str(payload.get("profile", "balanced"))),
            offload_video_to_cpu=bool(payload.get("offload_video_to_cpu", True)),
            offload_state_to_cpu=bool(payload.get("offload_state_to_cpu", True)),
            fill_holes=bool(payload.get("fill_holes", True)),
            max_hole_area=int(payload.get("max_hole_area", 2_048)),
            raw_output_pattern=raw_output_pattern,
        )

    def submit_video(self, payload: dict[str, Any]) -> str:
        if self._video_service is None:
            raise KyvenError(
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                message="Video segmentation service is unavailable.",
            )
        try:
            request = self.video_request_from_payload(payload)
            request.validate()
        except (KeyError, TypeError, ValueError) as exc:
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="The video segmentation payload is invalid.",
                technical_detail=str(exc),
            ) from exc
        record = JobRecord(job_id=uuid.uuid4().hex, request=request)
        with self._lock:
            self._jobs[record.job_id] = record
        self._executor.submit(self._run_video, record)
        return record.job_id

    @staticmethod
    def refine_request_from_payload(payload: dict[str, Any]) -> RefineRequest:
        source = Path(str(payload["source"]))
        mask = Path(str(payload["mask"]))
        output = Path(str(payload["output"]))
        trimap_output_value = payload.get("trimap_output")
        trimap_output = Path(str(trimap_output_value)) if trimap_output_value else None
        if (
            not source.is_absolute()
            or not mask.is_absolute()
            or not output.is_absolute()
            or (trimap_output is not None and not trimap_output.is_absolute())
        ):
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="Refinement job paths must be absolute.",
            )
        roi = JobManager._box_from_payload(payload, "roi")
        profile = ExecutionProfile(str(payload.get("profile", "balanced")))
        default_tile = {
            ExecutionProfile.LOW_MEMORY: 512,
            ExecutionProfile.BALANCED: 1024,
            ExecutionProfile.QUALITY: 0,
        }[profile]
        return RefineRequest(
            source=source,
            mask=mask,
            output=output,
            trimap_output=trimap_output,
            provider_id=str(payload.get("model_id", "vitmatte-small-composition-1k")),
            profile=profile,
            roi=roi,
            generate_trimap=bool(payload.get("generate_trimap", True)),
            foreground_radius=int(payload.get("foreground_radius", 10)),
            background_radius=int(payload.get("background_radius", 15)),
            tile_size=int(payload.get("tile_size", default_tile)),
            tile_overlap=int(payload.get("tile_overlap", 64)),
        )

    def submit_refine(self, payload: dict[str, Any]) -> str:
        if self._refine_service is None:
            raise KyvenError(
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                message="Refinement service is unavailable.",
            )
        try:
            request = self.refine_request_from_payload(payload)
            request.validate()
        except (KeyError, TypeError, ValueError) as exc:
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="The refinement job payload is invalid.",
                technical_detail=str(exc),
            ) from exc
        record = JobRecord(job_id=uuid.uuid4().hex, request=request)
        with self._lock:
            self._jobs[record.job_id] = record
        self._executor.submit(self._run_refine, record)
        return record.job_id

    @staticmethod
    def inpaint_request_from_payload(payload: dict[str, Any]) -> InpaintRequest:
        source = Path(str(payload["source"]))
        mask = Path(str(payload["mask"]))
        output = Path(str(payload["output"]))
        if not source.is_absolute() or not mask.is_absolute() or not output.is_absolute():
            raise KyvenError(ErrorCode.INVALID_REQUEST, "Inpaint job paths must be absolute.")
        return InpaintRequest(
            source=source,
            mask=mask,
            output=output,
            provider_id=str(payload.get("model_id", "lama-2025jan-onnx")),
            profile=ExecutionProfile(str(payload.get("profile", "balanced"))),
            crop_mode=str(payload.get("crop_mode", "auto")),
            roi=JobManager._box_from_payload(payload, "roi"),
            context_padding=int(payload.get("context_padding", 128)),
            mask_grow=int(payload.get("mask_grow", 8)),
            mask_feather=float(payload.get("mask_feather", 4.0)),
            processing_size=int(payload.get("processing_size", 0)),
        )

    def submit_inpaint(self, payload: dict[str, Any]) -> str:
        if self._inpaint_service is None:
            raise KyvenError(ErrorCode.PROVIDER_UNAVAILABLE, "Inpaint service is unavailable.")
        try:
            request = self.inpaint_request_from_payload(payload)
            request.validate()
        except (KeyError, TypeError, ValueError) as exc:
            raise KyvenError(ErrorCode.INVALID_REQUEST, "The inpaint job payload is invalid.", technical_detail=str(exc)) from exc
        record = JobRecord(job_id=uuid.uuid4().hex, request=request)
        with self._lock:
            self._jobs[record.job_id] = record
        self._executor.submit(self._run_inpaint, record)
        return record.job_id

    def _run(self, record: JobRecord) -> None:
        with self._lock:
            if record.cancellation.is_cancelled:
                record.status = JobStatus.CANCELLED
                record.finished_at = time.time()
                return
            record.status = JobStatus.RUNNING
            record.started_at = time.time()
        try:
            if not isinstance(record.request, SegmentRequest):
                raise KyvenError(
                    code=ErrorCode.INVALID_REQUEST,
                    message="Image worker received a non-image request.",
                )
            result = self._service.run(record.request, record.cancellation)
            with self._lock:
                record.status = JobStatus.SUCCEEDED
                record.result = {
                    "output": str(result.output),
                    "raw_output": (
                        str(record.request.raw_output)
                        if record.request.raw_output is not None
                        else None
                    ),
                    "score": result.score,
                    "cache_key": result.cache_key,
                    "metadata": result.metadata,
                }
        except KyvenError as exc:
            with self._lock:
                record.status = (
                    JobStatus.CANCELLED if exc.code is ErrorCode.CANCELLED else JobStatus.FAILED
                )
                record.error = exc.to_dict()
        except Exception as exc:  # noqa: BLE001 - worker boundary converts failures to job state
            with self._lock:
                record.status = JobStatus.FAILED
                record.error = KyvenError(
                    code=ErrorCode.SERVER_ERROR,
                    message="The Kyven worker failed unexpectedly.",
                    technical_detail=str(exc),
                    recoverable=True,
                    suggested_action="Inspect the server log and retry the job.",
                ).to_dict()
        finally:
            with self._lock:
                record.finished_at = time.time()

    def _run_video(self, record: JobRecord) -> None:
        with self._lock:
            if record.cancellation.is_cancelled:
                record.status = JobStatus.CANCELLED
                record.finished_at = time.time()
                return
            record.status = JobStatus.RUNNING
            record.started_at = time.time()
        try:
            if self._video_service is None or not isinstance(record.request, VideoSegmentRequest):
                raise KyvenError(
                    code=ErrorCode.PROVIDER_UNAVAILABLE,
                    message="Video segmentation service is unavailable.",
                )
            result = self._video_service.run(record.request, record.cancellation)
            with self._lock:
                record.status = JobStatus.SUCCEEDED
                record.result = {
                    "output_pattern": str(record.request.output_pattern),
                    "output_count": len(result.outputs),
                    "first_frame": result.first_frame,
                    "last_frame": result.last_frame,
                    "key_frame": result.key_frame,
                    "direction": result.direction.value,
                    "metadata": result.metadata,
                }
        except KyvenError as exc:
            with self._lock:
                record.status = (
                    JobStatus.CANCELLED if exc.code is ErrorCode.CANCELLED else JobStatus.FAILED
                )
                record.error = exc.to_dict()
        except Exception as exc:  # noqa: BLE001 - worker boundary converts failures to job state
            with self._lock:
                record.status = JobStatus.FAILED
                record.error = KyvenError(
                    code=ErrorCode.SERVER_ERROR,
                    message="The Kyven video worker failed unexpectedly.",
                    technical_detail=str(exc),
                    recoverable=True,
                    suggested_action="Inspect the server log and retry with a shorter range.",
                ).to_dict()
        finally:
            with self._lock:
                record.finished_at = time.time()

    def _run_refine(self, record: JobRecord) -> None:
        with self._lock:
            if record.cancellation.is_cancelled:
                record.status = JobStatus.CANCELLED
                record.finished_at = time.time()
                return
            record.status = JobStatus.RUNNING
            record.started_at = time.time()
        try:
            if self._refine_service is None or not isinstance(record.request, RefineRequest):
                raise KyvenError(
                    code=ErrorCode.PROVIDER_UNAVAILABLE,
                    message="Refinement service is unavailable.",
                )
            result = self._refine_service.run(record.request, record.cancellation)
            with self._lock:
                record.status = JobStatus.SUCCEEDED
                record.result = {
                    "output": str(result.output),
                    "trimap_output": (
                        str(result.trimap_output) if result.trimap_output is not None else None
                    ),
                    "cache_key": result.cache_key,
                    "metadata": result.metadata,
                }
        except KyvenError as exc:
            with self._lock:
                record.status = (
                    JobStatus.CANCELLED if exc.code is ErrorCode.CANCELLED else JobStatus.FAILED
                )
                record.error = exc.to_dict()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                record.status = JobStatus.FAILED
                record.error = KyvenError(
                    code=ErrorCode.SERVER_ERROR,
                    message="The Kyven refinement worker failed unexpectedly.",
                    technical_detail=str(exc),
                    recoverable=True,
                    suggested_action="Inspect the server log and retry with Low Memory.",
                ).to_dict()
        finally:
            with self._lock:
                record.finished_at = time.time()

    def _run_inpaint(self, record: JobRecord) -> None:
        with self._lock:
            if record.cancellation.is_cancelled:
                record.status = JobStatus.CANCELLED
                record.finished_at = time.time()
                return
            record.status = JobStatus.RUNNING
            record.started_at = time.time()
        try:
            if self._inpaint_service is None or not isinstance(record.request, InpaintRequest):
                raise KyvenError(ErrorCode.PROVIDER_UNAVAILABLE, "Inpaint service is unavailable.")
            result = self._inpaint_service.run(record.request, record.cancellation)
            with self._lock:
                record.status = JobStatus.SUCCEEDED
                record.result = {
                    "output": str(result.output),
                    "cache_key": result.cache_key,
                    "metadata": result.metadata,
                }
        except KyvenError as exc:
            with self._lock:
                record.status = (
                    JobStatus.CANCELLED if exc.code is ErrorCode.CANCELLED else JobStatus.FAILED
                )
                record.error = exc.to_dict()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                record.status = JobStatus.FAILED
                record.error = KyvenError(
                    ErrorCode.SERVER_ERROR,
                    "The Kyven inpaint worker failed unexpectedly.",
                    technical_detail=str(exc),
                    recoverable=True,
                ).to_dict()
        finally:
            with self._lock:
                record.finished_at = time.time()

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KyvenError(
                    code=ErrorCode.JOB_NOT_FOUND,
                    message=f"Kyven job was not found: {job_id}",
                )
            return record.snapshot()

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KyvenError(
                    code=ErrorCode.JOB_NOT_FOUND,
                    message=f"Kyven job was not found: {job_id}",
                )
            record.cancellation.cancel()
            if record.status is JobStatus.QUEUED:
                record.status = JobStatus.CANCELLED
                record.finished_at = time.time()
            return record.snapshot()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def unload_all(self, registry: ProviderRegistry) -> None:
        """Queue unloading behind active inference to avoid model races."""

        self._executor.submit(registry.unload_all).result(timeout=120)
