"""Finite motion-compensated raw-residual history; never estimates optical flow."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from kyven.cancellation import CancellationToken
from kyven.dlss.temporal import difference, reproject


@dataclass(frozen=True)
class TemporalFrame:
    source: np.ndarray
    residual: np.ndarray
    mask: np.ndarray
    motion: np.ndarray
    forward: np.ndarray | None = None
    depth: np.ndarray | None = None

    @property
    def nbytes(self) -> int:
        return sum(value.nbytes for value in (
            self.source, self.residual, self.mask, self.motion, self.forward, self.depth
        ) if value is not None)


class TemporalBuffer:
    """One shot/session, bounded by both frame count and retained CPU bytes."""

    def __init__(self) -> None:
        self.frames: deque[TemporalFrame] = deque()

    @property
    def nbytes(self) -> int:
        return sum(frame.nbytes for frame in self.frames)

    def clear(self) -> None:
        self.frames.clear()

    def append(self, frame: TemporalFrame, count: int, budget: int) -> None:
        if frame.nbytes > budget:
            raise ValueError("One temporal history frame exceeds its CPU memory budget")
        # Evict before retaining the new frame. References are owned by the serialized service.
        while self.frames and (len(self.frames) >= count or self.nbytes + frame.nbytes > budget):
            self.frames.popleft()
        self.frames.append(frame)

    def accumulate(
        self, residual: np.ndarray, source: np.ndarray, mask: np.ndarray,
        motion: np.ndarray, *, strength: float, rejection: float,
        clamp: float = 2.0, depth: np.ndarray | None = None, depth_rejection: float = 0.1,
        consistency: bool = False, consistency_threshold: float = 1.5,
        cancellation: CancellationToken | None = None,
    ) -> tuple[np.ndarray, dict[str, int | float]]:
        """Align each raw history residual once along the full backward-vector chain.

        Never store this blended output as another sample: history expires at the
        configured horizon instead of recursively carrying old hallucinations forever.
        Forward vectors belong to their own frame (t -> t+1); sample the previous
        frame's forward field at the current backward landing position for FB checks.
        """
        offset = motion.copy()
        step_backward = motion
        chain = np.ones(source.shape[:2], dtype=np.float32)
        maximum = np.zeros_like(chain)
        total = np.zeros_like(chain)
        weighted = np.zeros_like(residual)
        used = 0
        limit = clamp * (0.01 + np.abs(residual)) if clamp > 0 else None
        for age, frame in enumerate(reversed(self.frames), 1):
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            warped_source, valid = reproject(frame.source, offset, cancellation)
            confidence = np.clip(1 - difference(source, warped_source) / rejection, 0, 1)
            del warped_source
            warped_mask, _ = reproject(frame.mask, offset, cancellation)
            confidence *= np.clip(1 - np.abs(mask - warped_mask) * 4, 0, 1)
            confidence *= valid & (mask > 0)
            if consistency:
                if frame.forward is None:
                    confidence.fill(0)
                else:
                    forward, _ = reproject(frame.forward, offset, cancellation)
                    error = np.linalg.norm(step_backward + forward, axis=-1)
                    confidence *= np.clip(1 - error / consistency_threshold, 0, 1)
            if depth is not None and frame.depth is not None:
                warped_depth, _ = reproject(frame.depth, offset, cancellation)
                error = np.abs(depth - warped_depth) / (
                    0.001 + np.maximum(np.abs(depth), np.abs(warped_depth)))
                confidence *= np.clip(1 - error / depth_rejection, 0, 1)
            # A broken intermediate correspondence cannot be bypassed by older samples.
            np.minimum(chain, confidence, out=chain)
            if not np.any(chain > 0):
                break
            warped_residual, _ = reproject(frame.residual, offset, cancellation)
            if limit is not None:
                warped_residual = residual + np.clip(warped_residual - residual, -limit, limit)
            weight = chain / age
            weighted += warped_residual * weight[..., None]
            total += weight
            np.maximum(maximum, chain, out=maximum)
            used += 1
            step_backward, step_valid = reproject(frame.motion, offset, cancellation)
            chain *= step_valid
            offset += step_backward
        mean = np.zeros_like(residual)
        np.divide(weighted, total[..., None], out=mean, where=total[..., None] > 0)
        blend = strength * maximum
        result = residual * (1 - blend[..., None]) + mean * blend[..., None]
        return result, {"temporal_samples_used": used,
                        "temporal_reused_fraction": float(np.mean(maximum > 0)),
                        "temporal_mean_weight": float(blend.mean())}
