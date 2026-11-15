"""DLSS orchestration, temporal safety, deterministic history keys and durable cache."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kyven.cache_manifest import read_valid_manifest, write_manifest_atomic
from kyven.cancellation import CancellationToken
from kyven.dlss.color import prepare, restore_residual
from kyven.dlss.history import TemporalBuffer, TemporalFrame
from kyven.dlss.images import channel, read_channels, resize, write_exr
from kyven.dlss.models import DlssRequest
from kyven.dlss.temporal import difference, reproject
from kyven.errors import ErrorCode, KyvenError
from kyven.hashing import cached_file_sha256
from kyven.segment.providers.registry import ProviderRegistry


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


@dataclass
class _History:
    configuration: str
    frame: int
    key: str
    provider: Any
    generation: int


class DlssService:
    """Single GPU queue; one bounded shot/session buffer, with no future-frame requests."""

    def __init__(self, registry: ProviderRegistry) -> None:
        self.registry = registry
        self._history: _History | None = None
        self._buffer = TemporalBuffer()

    def clear_history(self) -> None:
        """Release retained CPU frames on the serialized queue when unloading DLSS."""
        self._history = None
        self._buffer.clear()

    def run(self, request: DlssRequest,
            cancellation: CancellationToken | None = None) -> dict[str, Any]:
        token = cancellation or CancellationToken()
        try:
            return self._run(request, token)
        except Exception:
            self.clear_history()
            # A failed/cancelled output must never leave a partially advanced GPU history.
            if "dlss-external" in self.registry.loaded_ids:
                self.registry.unload("dlss-external")
            raise

    def _run(self, request: DlssRequest, token: CancellationToken) -> dict[str, Any]:
        request.validate()
        token.raise_if_cancelled()
        token.report_progress(0.05, "Reading linear DLSS inputs")
        paths = {name: getattr(request, name) for name in ("source", "motion", "mask", "depth")}
        hashes = {}
        for name, path in paths.items():
            if path is not None:
                if not path.is_file():
                    raise KyvenError(ErrorCode.INVALID_REQUEST, f"DLSS {name} is missing: {path}")
                hashes[name] = cached_file_sha256(path)
        if not request.worker.is_file():
            raise KyvenError(
                ErrorCode.PROVIDER_UNAVAILABLE, "DLSS worker executable was not found.",
                suggested_action="Select your locally built DLSS_Nuke_Worker.exe.",
            )
        # Runtime identity participates in cache/history invalidation, including replaced DLLs.
        runtime = {p.name: cached_file_sha256(p)
                   for p in sorted(request.worker.parent.glob("*.dll")) if p.is_file()}
        runtime[request.worker.name] = cached_file_sha256(request.worker)
        inputs = read_channels(request.source)
        rgba = np.stack([channel(inputs, c) for c in ("R", "G", "B")]
                        + [inputs.get("A", np.ones_like(channel(inputs, "R")))], axis=-1)
        height, width = rgba.shape[:2]
        if height * width > 33_554_432 or max(height, width) > 16384:
            raise KyvenError(ErrorCode.INVALID_REQUEST, "DLSS source exceeds the 32 MP safety limit.")
        ratio = min(1.0, request.max_dimension / max(width, height))
        size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
        small = resize(rgba, size)
        source = small[..., :3].copy()
        # Alpha is explicitly a guide, never an implicit premultiply operation.
        small[..., 3] = 1.0
        full_mask = np.ones((height, width), dtype=np.float32)

        def guide(path: Path, name: str) -> np.ndarray:
            values = channel(read_channels(path), name)
            if values.shape != (height, width):
                raise KyvenError(
                    ErrorCode.INVALID_REQUEST, "DLSS guides must match Source dimensions.",
                    suggested_action="Reformat vectors/depth/mask to Source before connecting them.",
                )
            return values

        if request.mask_source == "source_alpha":
            full_mask = channel(inputs, "A").copy()
        elif request.mask_source == "input":
            full_mask = guide(request.mask, request.mask_channel)
        full_mask = np.clip(full_mask, 0, 1)
        mask = resize(full_mask, size)
        depth = resize(guide(request.depth, request.depth_channel), size) if request.depth else None
        motion = forward = None
        if request.motion is not None:
            vectors = read_channels(request.motion)
            sx = size[0] if request.motion_units == "uv" else size[0] / width
            sy = size[1] if request.motion_units == "uv" else size[1] / height

            def motion_field(x: str, y: str) -> np.ndarray:
                mx, my = channel(vectors, x), channel(vectors, y)
                if mx.shape != (height, width) or my.shape != (height, width):
                    raise KyvenError(ErrorCode.INVALID_REQUEST, "External vectors must match Source dimensions.")
                field = resize(np.stack((mx, my), axis=-1), size)
                field[..., 0] *= sx * request.motion_scale_x
                field[..., 1] *= sy * request.motion_scale_y * (-1 if request.motion_y_up else 1)
                if np.abs(field).max() > 65504:
                    raise KyvenError(ErrorCode.INVALID_REQUEST, "Motion vectors exceed float16 range.")
                return field

            motion = motion_field(request.motion_x, request.motion_y)
            if request.temporal and request.motion_consistency:
                forward = motion_field(request.forward_x, request.forward_y)
        budget = request.history_budget_mb * 1024**2
        frame_bytes = (source.nbytes * 2 + mask.nbytes + (motion.nbytes if motion is not None else 0)
                       + (forward.nbytes if forward is not None else 0)
                       + (depth.nbytes if depth is not None else 0))
        if request.temporal and frame_bytes > budget:
            raise KyvenError(
                ErrorCode.INVALID_REQUEST, "One DLSS history frame exceeds the CPU history budget.",
                suggested_action="Lower Processing Size or increase History Budget in Advanced.")
        if not request.hdr_rolloff and np.abs(small).max() > 65504:
            raise KyvenError(ErrorCode.INVALID_REQUEST, "Source exceeds DLSS float16 range.")
        small[..., :3], inverse_gain = prepare(source, enabled=request.hdr_rolloff,
                                              knee=request.rolloff_knee)
        provider = self.registry.activate("dlss-external")
        config = digest({"version": 5, "settings": request.settings(), "runtime": runtime,
                         "size": size, "source_size": (width, height),
                         "has_motion": motion is not None, "has_depth": depth is not None})
        history = self._history
        reason = "single_frame"
        continuous = False
        if request.temporal:
            reason = "first_frame"
            if request.reset:
                reason = "artist_reset"
            elif history is not None:
                if history.configuration != config or history.provider is not provider:
                    reason = "settings_or_session_changed"
                elif history.generation != getattr(provider, "generation", 0):
                    reason = "worker_restarted"
                elif request.frame != history.frame + 1:
                    reason = "nonconsecutive_frame"
                else:
                    warped, valid = reproject(self._buffer.frames[-1].source, motion, token)
                    error = difference(source, warped)
                    # A large invalid area or a photometric discontinuity denotes a cut.
                    if valid.mean() < 0.1 or np.median(error[valid]) > request.scene_cut:
                        reason = "scene_cut"
                    else:
                        continuous, reason = True, "continuous"
        if not continuous:
            self._buffer.clear()
        key = digest({"configuration": config, "inputs": hashes, "frame": request.frame,
                      "predecessor": history.key if continuous else None})
        # Temporal cache hits must replay inference to reconstruct private DLSS state.
        # Skipping cached frames would silently reuse the wrong GPU history.
        if not request.temporal:
            cached = read_valid_manifest(request.output, key, (request.output,))
            if (cached is not None and cached.get("metadata", {}).get("output_sha256")
                    == cached_file_sha256(request.output)):
                self._history = None
                self._buffer.clear()
                return {"output": str(request.output), "cache_key": key,
                        "metadata": {**cached["metadata"], "cache_hit": True}}
        token.raise_if_cancelled()
        token.report_progress(0.25, "DLSS reconstruction with external guides")
        predicted = provider.predict(
            request, small, motion,
            mask if request.mask_source != "none" else None,
            depth, not continuous, token,
        )
        if predicted.shape != small.shape or not np.isfinite(predicted).all():
            raise KyvenError(ErrorCode.INFERENCE_FAILED, "DLSS returned invalid pixels.")
        token.raise_if_cancelled()
        token.report_progress(0.8, "Rejecting unstable history and publishing EXR")
        raw_residual = restore_residual(predicted[..., :3], small[..., :3], inverse_gain)
        if not np.isfinite(raw_residual).all():
            raise KyvenError(ErrorCode.INFERENCE_FAILED, "DLSS produced a nonfinite scene-linear correction.")
        residual = raw_residual
        accumulation = {"temporal_samples_used": 0, "temporal_reused_fraction": 0.0,
                        "temporal_mean_weight": 0.0}
        if continuous and request.temporal_strength > 0:
            residual, accumulation = self._buffer.accumulate(
                raw_residual, source, mask, motion, strength=request.temporal_strength,
                rejection=request.rejection, clamp=request.history_clamp,
                depth=depth, depth_rejection=request.depth_rejection,
                consistency=request.motion_consistency,
                consistency_threshold=request.consistency_threshold * max(size[0] / width, size[1] / height),
                cancellation=token)
        correction = resize(residual, (width, height))
        if request.protect_outside and request.mask_source != "none":
            # Soft weights already condition NR itself. Do not apply them twice;
            # only remove corrections that interpolation/history spread into black.
            correction[full_mask <= 0] = 0
        result = rgba.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            result[..., :3] += correction
        if not np.isfinite(result).all():
            raise KyvenError(ErrorCode.INFERENCE_FAILED,
                             "Restored DLSS RGB exceeds the finite float32 EXR range.")
        metadata = {"provider": "dlss-external", "cache_hit": False,
                    "history_reset": not continuous, "history_reason": reason,
                    "processing_size": size, "control_mask": request.mask_source,
                    "external_motion": motion is not None, "runtime": runtime,
                    "alpha": "preserved", "resolution": "native_residual"}
        metadata["guide_convention"] = "artist_white_allows; native_rgba_white_allows"
        metadata["hdr_rolloff"] = "anchored_peak_shoulder_v1" if request.hdr_rolloff else "off"
        metadata["rolloff_knee"] = request.rolloff_knee
        metadata.update(accumulation)
        if request.temporal:
            self._buffer.append(TemporalFrame(source, raw_residual, mask, motion, forward, depth),
                                request.history_frames, budget)
        metadata["history_capacity"] = min(request.history_frames, budget // frame_bytes) if request.temporal else 0
        metadata["history_cpu_bytes"] = self._buffer.nbytes
        metadata["history_frames_buffered"] = len(self._buffer.frames)
        metadata["motion_consistency"] = request.temporal and request.motion_consistency
        token.raise_if_cancelled()
        try:
            write_exr(request.output, result)
            token.raise_if_cancelled()
            metadata["output_sha256"] = cached_file_sha256(request.output)
            write_manifest_atomic(request.output, key, metadata)
        except OSError as exc:
            raise KyvenError(ErrorCode.OUTPUT_FAILED, "Could not publish the DLSS cache.",
                             technical_detail=str(exc)) from exc
        self._history = _History(config, request.frame, key,
                                 provider, getattr(provider, "generation", 0)) if request.temporal else None
        token.report_progress(1.0, "DLSS complete")
        return {"output": str(request.output), "cache_key": key, "metadata": metadata}
