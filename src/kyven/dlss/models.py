"""Validated, host-independent DLSS job contract."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kyven.errors import ErrorCode, KyvenError


@dataclass(frozen=True)
class DlssRequest:
    source: Path
    output: Path
    worker: Path
    motion: Path | None = None
    mask: Path | None = None
    depth: Path | None = None
    session: str = "default"
    frame: int = 1
    temporal: bool = False
    reset: bool = False
    mask_source: str = "none"  # none, source_alpha, input
    mask_channel: str = "A"
    motion_x: str = "R"
    motion_y: str = "G"
    motion_units: str = "pixels"  # pixels or uv
    motion_y_up: bool = True
    motion_scale_x: float = 1.0
    motion_scale_y: float = 1.0
    depth_channel: str = "depth.Z"
    intensity: float = 1.0
    local_tone: float = 1.0
    local_structure: float = 1.0
    skin_structure: float = -1.0
    style: int = 0
    preset: int = 0
    temporal_strength: float = 0.35
    history_frames: int = 3
    history_budget_mb: int = 512
    history_clamp: float = 2.0
    motion_consistency: bool = False
    forward_x: str = "forward.u"
    forward_y: str = "forward.v"
    consistency_threshold: float = 1.5
    depth_rejection: float = 0.1
    rejection: float = 0.1
    scene_cut: float = 0.45
    protect_outside: bool = True
    max_dimension: int = 1024
    hdr_rolloff: bool = True
    rolloff_knee: float = 0.5
    timeout: float = 120.0

    def validate(self) -> None:
        def invalid(message: str) -> None:
            raise KyvenError(ErrorCode.INVALID_REQUEST, message)

        for name in ("source", "output", "worker", "motion", "mask", "depth"):
            path = getattr(self, name)
            if path is not None and (not isinstance(path, Path) or not path.is_absolute()):
                invalid(f"DLSS {name} must be an absolute path.")
            if path is None and name in {"source", "output", "worker"}:
                invalid(f"DLSS {name} is required.")
        for name in ("mask_source", "motion_units", "session", "motion_x", "motion_y",
                     "mask_channel", "depth_channel", "forward_x", "forward_y"):
            if not isinstance(getattr(self, name), str):
                invalid(f"DLSS {name} must be text.")
        for name in ("frame", "style", "preset", "max_dimension", "history_frames", "history_budget_mb"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                invalid(f"DLSS {name} must be an integer.")
        for name in ("temporal", "reset", "motion_y_up", "protect_outside", "hdr_rolloff", "motion_consistency"):
            if not isinstance(getattr(self, name), bool):
                invalid(f"DLSS {name} must be a boolean.")
        inputs = (self.source, self.worker, self.motion, self.mask, self.depth)
        if self.output.suffix.lower() != ".exr":
            invalid("DLSS output must be a scene-linear EXR file.")
        if self.output.resolve() in {p.resolve() for p in inputs if p is not None}:
            invalid("DLSS output must not overwrite an input or the worker.")
        if self.temporal and self.motion is None:
            invalid("Temporal DLSS requires external current-to-previous motion vectors.")
        if self.mask_source not in {"none", "source_alpha", "input"}:
            invalid("Unknown DLSS mask source.")
        if self.mask_source == "input" and self.mask is None:
            invalid("Connect a control-mask input or select source alpha.")
        if self.motion_units not in {"pixels", "uv"}:
            invalid("Motion units must be pixels or uv.")
        if not self.session or len(self.session) > 256:
            invalid("DLSS session must contain 1–256 characters.")
        if not -(2**31) <= self.frame < 2**31:
            invalid("DLSS frame is outside the supported range.")
        if not 64 <= self.max_dimension <= 8192:
            invalid("DLSS processing limit must be between 64 and 8192 pixels.")
        if not 1 <= self.history_frames <= 4 or not 64 <= self.history_budget_mb <= 1024:
            invalid("DLSS history requires 1–4 previous frames and a 64–1024 MiB CPU budget.")
        ranges = {
            "intensity": (0, 2), "local_tone": (0, 2), "local_structure": (0, 2),
            "skin_structure": (-1, 2), "temporal_strength": (0, 0.95),
            "rejection": (0.001, 10), "scene_cut": (0.01, 10), "timeout": (1, 600),
            "motion_scale_x": (-100000, 100000), "motion_scale_y": (-100000, 100000),
            "rolloff_knee": (0.01, 0.95),
            "history_clamp": (0, 10), "consistency_threshold": (0.01, 100),
            "depth_rejection": (0.001, 10),
        }
        for name, (low, high) in ranges.items():
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not low <= value <= high):
                invalid(f"DLSS {name} must be finite and between {low} and {high}.")
        if self.style not in range(3) or self.preset not in range(4):
            invalid("Invalid DLSS style or preset.")
        for name in ("motion_x", "motion_y", "mask_channel", "depth_channel", "forward_x", "forward_y"):
            if not getattr(self, name):
                invalid(f"DLSS {name} cannot be empty.")

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> DlssRequest:
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise KyvenError(ErrorCode.INVALID_REQUEST, f"Unknown DLSS fields: {sorted(unknown)}")
        values = dict(payload)
        try:
            for name in ("source", "output", "worker", "motion", "mask", "depth"):
                if name in values and values[name] is not None:
                    values[name] = Path(values[name])
            for name in ("frame", "style", "preset", "max_dimension", "history_frames", "history_budget_mb"):
                if name in values:
                    if isinstance(values[name], bool) or int(values[name]) != values[name]:
                        raise ValueError(f"{name} must be an integer")
                    values[name] = int(values[name])
            for name in ("temporal", "reset", "motion_y_up", "protect_outside", "hdr_rolloff", "motion_consistency"):
                if name in values and not isinstance(values[name], bool):
                    raise ValueError(f"{name} must be a boolean")
            request = cls(**values)
            request.validate()
            return request
        except (TypeError, ValueError, AttributeError, OverflowError) as exc:
            raise KyvenError(
                ErrorCode.INVALID_REQUEST, "Invalid DLSS request.", technical_detail=str(exc)
            ) from exc

    def settings(self) -> dict[str, Any]:
        """Settings affecting history, excluding paths, timeline, and scheduling."""
        excluded = {"source", "output", "motion", "mask", "depth", "frame", "reset", "timeout"}
        return {k: str(v) if isinstance(v, Path) else v
                for k, v in asdict(self).items() if k not in excluded}
