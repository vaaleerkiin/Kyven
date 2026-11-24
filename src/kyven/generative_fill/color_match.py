"""Context calibration and background continuity; NumPy/Pillow only.

Never use pixels of the removed source subject to estimate a correction. The
caller owns the final permission mask and copies Source outside that mask.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def _blur(values: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3 * sigma)))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    output = values.astype(np.float32)
    for axis in (0, 1):
        pads = [(0, 0)] * output.ndim
        pads[axis] = (radius, radius)
        padded = np.pad(output, pads, mode="edge")
        result = np.zeros_like(output)
        for index, weight in enumerate(kernel):
            slices = [slice(None)] * output.ndim
            slices[axis] = slice(index, index + output.shape[axis])
            result += padded[tuple(slices)] * weight
        output = result
    return output


def match_context(predicted: np.ndarray, source: np.ndarray, inference_mask: np.ndarray,
                  *, strength: float = 1.0, background: bool = False
                  ) -> tuple[np.ndarray, dict]:
    """Calibrate paired known pixels; optionally continue background at edges.

    Background matching is for removals only. It carries a background color
    correction through locally flat surfaces, preserving generated texture and geometry. An
    addition/replacement uses context calibration so its new subject keeps its
    intended colors. Uncorrelated/insufficient context is left untouched.
    """
    details = {"applied": False, "background": False}
    if strength <= 0:
        return predicted, details
    edit = np.asarray(inference_mask) > 0
    # Exclude VAE edge contamination and the old subject's antialiasing fringe.
    expanded = np.asarray(Image.fromarray(edit.astype(np.uint8) * 255).filter(
        ImageFilter.MaxFilter(17))) > 0
    known = ~expanded
    if np.count_nonzero(known) < 64 or not np.any(edit):
        return predicted, details
    generated = predicted.astype(np.float32)
    original = source.astype(np.float32)
    x, y = generated[known], original[known]
    stride = max(1, len(x) // 50000)
    x, y = x[::stride], y[::stride]
    gain = np.ones(3, dtype=np.float32)
    offset = np.median(y - x, axis=0)
    for channel in range(3):
        a, b = x[:, channel], y[:, channel]
        weights = np.ones(len(a), dtype=np.float32)
        for _ in range(4):
            total = weights.sum()
            ma, mb = (weights * a).sum() / total, (weights * b).sum() / total
            variance = (weights * (a - ma) ** 2).sum() / total
            g = (np.clip((weights * (a - ma) * (b - mb)).sum() / total / variance,
                         0.8, 1.25) if variance > 16 else 1.0)
            o = np.clip(mb - g * ma, -32, 32)
            residual = b - (a * g + o)
            weights = np.minimum(1.0, 3.0 / np.maximum(np.abs(residual), 0.001))
        gain[channel], offset[channel] = g, o
    residual = y - (x * gain + offset)
    error = np.median(np.abs(residual), axis=0)
    if float(np.max(error)) > 8:
        details["reason"] = "uncorrelated_context"
        return predicted, details
    calibrated = generated * gain + offset
    details.update(applied=True, gain=gain.tolist(), offset=offset.tolist(),
                   context_error=error.tolist())
    if background:
        # A removal can reveal flat pavement/wall whose generated exposure is
        # different even after VAE calibration. Match only rows with consistent,
        # low-texture context on BOTH sides. Never infer across a scene edge.
        correction = np.zeros_like(calibrated)
        eligible = np.zeros(edit.shape[0], dtype=bool)
        matched_rows = 0
        for row in range(edit.shape[0]):
            changes = np.diff(np.pad(edit[row].astype(np.int8), (1, 1)))
            for left, right in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
                if right - left < 8:
                    continue
                lx = np.flatnonzero(known[row, max(0, left - 64):left]) + max(0, left - 64)
                rx = np.flatnonzero(known[row, right:min(edit.shape[1], right + 64)]) + right
                lx, rx = lx[-24:], rx[:24]
                if min(len(lx), len(rx)) < 6:
                    continue
                before_left, before_right = original[row, lx], original[row, rx]
                target_left, target_right = np.median(before_left, axis=0), np.median(before_right, axis=0)
                if np.max(abs(target_left - target_right)) > 12:
                    continue
                if max(np.median(abs(before_left - target_left)),
                       np.median(abs(before_right - target_right))) > 5:
                    continue
                depth = min(16, max(2, (right - left) // 4))
                inside_left = calibrated[row, left:left + depth]
                inside_right = calibrated[row, right - depth:right]
                a, b = np.median(inside_left, axis=0), np.median(inside_right, axis=0)
                if max(np.median(abs(inside_left - a)), np.median(abs(inside_right - b))) > 6:
                    continue
                dl, dr = np.clip(target_left - a, -24, 24), np.clip(target_right - b, -24, 24)
                t = np.linspace(0, 1, right - left, dtype=np.float32)[:, None]
                correction[row, left:right] = dl * (1 - t) + dr * t
                eligible[row] = True
                matched_rows += 1
        if matched_rows:
            # Isolated eligible scanlines through reflections/door frames are
            # not evidence of a flat surface. They would create colored bands.
            transitions = np.diff(np.pad(eligible.astype(np.int8), (1, 1)))
            for start, end in zip(np.flatnonzero(transitions == 1),
                                  np.flatnonzero(transitions == -1)):
                if end - start < 24:
                    correction[start:end] = 0
            # Fade eligibility changes smoothly; don't carry correction across
            # unrelated regions of the image or use removed source pixels.
            calibrated += _blur(correction, 2.0)
            details.update(background=True, background_rows=matched_rows)
    result = np.clip(np.rint(generated + (calibrated - generated) * strength), 0, 255)
    return result.astype(np.uint8), details
