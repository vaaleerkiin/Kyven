"""Reversible, hue-preserving HDR conditioning around the bounded NR input.

No frame statistics or exposure adaptation: the same RGB always receives the same
gain/lift. Inverse gains are anchored to the original Source, so half-float
quantization and unchanged inference cannot clip highlights or alter negative RGB.
"""

from __future__ import annotations

import numpy as np


def prepare(rgb: np.ndarray, *, enabled: bool = True, knee: float = 0.5
            ) -> tuple[np.ndarray, np.ndarray]:
    """Return actual half-float model input and its per-pixel inverse gain.

    Positive RGB is scaled uniformly to preserve ratios. A negative minimum is
    lifted before scaling; its exact original value remains in the Source base.
    Above the knee, peak p maps to k + (1-k)*(p-k)/(p-k+1-k).
    """
    working = np.empty_like(rgb, dtype=np.float32)
    inverse_gain = np.ones(rgb.shape[:2], dtype=np.float32)
    for start in range(0, rgb.shape[0], 128):
        end = start + 128
        values = rgb[start:end].astype(np.float64)
        if enabled:
            values -= np.minimum(values.min(axis=-1, keepdims=True), 0)
            peak = values.max(axis=-1)
            shoulder = np.maximum(peak - knee, 0)
            mapped = knee + (1 - knee) * (shoulder / (shoulder + 1 - knee))
            inverse = np.ones_like(peak)
            np.divide(peak, mapped, out=inverse, where=peak > knee)
            values /= inverse[..., None]
            # Store the inverse rather than a tiny gain to avoid denormal precision loss.
            inverse_gain[start:end] = np.minimum(inverse, np.finfo(np.float32).max)
        working[start:end] = values.astype(np.float16).astype(np.float32)
    return working, inverse_gain


def restore_residual(predicted: np.ndarray, working: np.ndarray,
                     inverse_gain: np.ndarray) -> np.ndarray:
    """Undo the conditioning on the neural change; Source supplies the exact base."""
    result = np.empty_like(working)
    for start in range(0, working.shape[0], 128):
        end = start + 128
        difference = predicted[start:end].astype(np.float64) - working[start:end]
        with np.errstate(over="ignore"):
            result[start:end] = difference * inverse_gain[start:end, ..., None]
    return result
