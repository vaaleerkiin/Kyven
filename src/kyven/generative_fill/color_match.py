"""Context calibration and background continuity; NumPy/Pillow only.

Never use pixels of the removed source subject to estimate a correction. The
caller owns the final permission mask and copies Source outside that mask.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def _box_mean(values: np.ndarray, radius: int) -> np.ndarray:
    """Bounded-memory separable mean, independent of OpenCV/SciPy runtimes."""
    result = np.asarray(values, dtype=np.float64)
    size = radius * 2 + 1
    for axis in (0, 1):
        pads = [(0, 0)] * result.ndim
        pads[axis] = (radius, radius)
        padded = np.pad(result, pads, mode="edge")
        sums = np.cumsum(padded, axis=axis)
        shape = list(sums.shape)
        shape[axis] = 1
        sums = np.concatenate((np.zeros(shape), sums), axis=axis)
        left = [slice(None)] * result.ndim
        right = list(left)
        left[axis] = slice(None, -size)
        right[axis] = slice(size, None)
        result = (sums[tuple(right)] - sums[tuple(left)]) / size
    return result.astype(np.float32)


def _match_local_background(calibrated, source, permission, radius):
    """Correct only low-frequency edge color supported by clean, flat context.

    No source pixels under the artist's alpha enter the estimate. Texture and
    positions are retained; a broad taper carries an RGB offset inward. Context
    and addition modes do not use this removal-only step.
    """
    selected = np.asarray(permission) > 0
    offsets = np.zeros_like(calibrated)
    weights = np.zeros(selected.shape, np.float32)
    confidence = np.zeros_like(weights)
    # Estimate each side independently. A wall on one side and pavement on
    # another must not require equal colors or be averaged into one RGB target.
    for vertical in (False, True):
        edit = selected.T if vertical else selected
        original = source.swapaxes(0, 1) if vertical else source
        generated = calibrated.swapaxes(0, 1) if vertical else calibrated
        accumulated = offsets.swapaxes(0, 1) if vertical else offsets
        summed = weights.T if vertical else weights
        strongest = confidence.T if vertical else confidence
        for row in range(edit.shape[0]):
            transitions = np.diff(np.pad(edit[row].astype(np.int8), (1, 1)))
            for left, right in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)):
                if right-left < 16:
                    continue
                for boundary, direction in ((left, 1), (right-1, -1)):
                    known_x = boundary-direction*np.arange(5, 25)
                    known_x = known_x[(known_x >= 0) & (known_x < edit.shape[1])]
                    known_x = known_x[~edit[row, known_x]]
                    inside_x = boundary+direction*np.arange(5, min(25, right-left))
                    if min(len(known_x), len(inside_x)) < 8:
                        continue
                    clean, patch = original[row, known_x], generated[row, inside_x]
                    target, current = np.median(clean, axis=0), np.median(patch, axis=0)
                    if (np.max(np.median(abs(clean-target), axis=0)) > 5 or
                            np.max(np.median(abs(patch-current), axis=0)) > 6):
                        continue
                    delta = target-current
                    # Large unmatched structural edges are not evidence of a cast.
                    if np.max(abs(delta)) > 48:
                        continue
                    delta = np.clip(delta, -32, 32)
                    depth = min(radius, right-left)
                    index = boundary+direction*np.arange(depth)
                    taper = np.clip((radius-np.arange(depth))/(radius*0.7), 0, 1)
                    weight = taper**4  # nearest valid boundary dominates corners
                    accumulated[row, index] += delta*weight[:, None]
                    summed[row, index] += weight
                    strongest[row, index] = np.maximum(strongest[row, index], taper)
    # Smooth only the offset field; never blur generated pixels or move edges.
    correction = _box_mean(offsets, 6) / np.maximum(_box_mean(weights, 6)[..., None], 1e-6)
    low = _box_mean(calibrated, 8)
    variation = np.sqrt(np.maximum(0, _box_mean(calibrated**2, 8)-low**2)).max(axis=2)
    flat = np.clip((20-variation)/10, 0, 1)
    confidence = _box_mean(confidence, 6) * flat * selected
    correction *= confidence[..., None]
    taper = confidence
    return calibrated + correction, {
        "local_background": bool(np.any(taper > 0.01)),
        "local_pixels": int(np.count_nonzero(taper > 0.01)),
        "local_radius": radius,
        "max_local_offset": float(np.max(np.abs(correction))),
    }


def _tone_curves(x, y, gain, offset):
    """Fit paired monotone curves; accept only a held-out error improvement.

    A gain/offset cannot undo gamma drift. Quantile groups are paired pixels,
    not unrelated source/edit histograms, so a new object's palette is retained.
    Unsupported intensities keep the conservative affine calibration.
    """
    levels = np.arange(256, dtype=np.float32)
    curves = levels[:, None] * gain + offset
    if len(x) < 512:
        return curves, False
    changed = False
    for channel in range(3):
        a, b = x[::2, channel], y[::2, channel]
        order = np.argsort(a)
        chunks = np.array_split(order, min(32, len(a) // 32))
        xp = np.array([np.median(a[group]) for group in chunks])
        yp = np.array([np.median(b[group]) for group in chunks])
        unique, inverse = np.unique(xp, return_inverse=True)
        if len(unique) < 8 or np.ptp(unique) < 60:
            continue
        yp = np.array([np.median(yp[inverse == index]) for index in range(len(unique))])
        yp = np.maximum.accumulate(yp)
        weight = np.clip(np.minimum(levels - unique[0], unique[-1] - levels) / 8, 0, 1)
        candidate = curves[:, channel] * (1-weight) + np.interp(levels, unique, yp) * weight
        candidate = np.maximum.accumulate(candidate)
        holdout_a, holdout_b = x[1::2, channel], y[1::2, channel]
        before = np.median(np.abs(holdout_b-(holdout_a*gain[channel]+offset[channel])))
        after = np.median(np.abs(holdout_b-np.interp(holdout_a, levels, candidate)))
        if before > 1 and after < before * 0.7 and before-after > 0.75:
            curves[:, channel] = candidate
            changed = True
    return curves, changed


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
                  *, strength: float = 1.0, background: bool = False,
                  permission_mask: np.ndarray | None = None, local_radius: int = 64,
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
        details["reason"] = "insufficient_clean_context"
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
    curves, nonlinear = _tone_curves(x, y, gain, offset)
    fitted = np.column_stack([np.interp(x[:, c], np.arange(256), curves[:, c]) for c in range(3)])
    residual = y - fitted
    error = np.median(np.abs(residual), axis=0)
    if float(np.max(error)) > 8:
        details["reason"] = "uncorrelated_context"
        return predicted, details
    calibrated = np.stack([np.interp(generated[..., c], np.arange(256), curves[:, c]) for c in range(3)], -1).astype(np.float32)
    details.update(applied=True, gain=gain.tolist(), offset=offset.tolist(),
                   context_error=error.tolist(), method="tone_curve" if nonlinear else "affine")
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
    if background and permission_mask is not None and local_radius > 0:
        calibrated, local = _match_local_background(
            calibrated, original, permission_mask, local_radius)
        details.update(local)
        if local["local_background"]:
            details["background"] = True
    result = np.clip(np.rint(generated + (calibrated - generated) * strength), 0, 255)
    return result.astype(np.uint8), details
