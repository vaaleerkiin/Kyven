"""CPU Reinhard Lab matching, fitted only on shared clean context.

The equations are the standard Lab mean/std transfer also used by ComfyUI's
DLSS5 Color Matcher. No ComfyUI, Torch or scikit-image dependency is required.
Unlike full-image matching, removed subject pixels never define the reference.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

_RGB_XYZ = np.array(((.4124564, .3575761, .1804375),
                     (.2126729, .7151522, .0721750),
                     (.0193339, .1191920, .9503041)), dtype=np.float32)
_WHITE = np.array((.95047, 1, 1.08883), dtype=np.float32)
_XYZ_RGB = np.linalg.inv(_RGB_XYZ)
_DELTA = 6 / 29


def rgb_to_lab(pixels: np.ndarray) -> np.ndarray:
    """sRGB bytes to D65 Lab/100, matching Nuke's CIE-Lab channel convention."""
    rgb = np.asarray(pixels, dtype=np.float32) / 255
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055)**2.4)
    xyz = (linear @ _RGB_XYZ.T) / _WHITE
    f = np.where(xyz > _DELTA**3, np.cbrt(xyz), xyz / (3*_DELTA**2) + 4/29)
    return np.stack((1.16*f[..., 1]-.16,
                     5*(f[..., 0]-f[..., 1]), 2*(f[..., 1]-f[..., 2])), axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """D65 Lab/100 to clipped sRGB bytes."""
    values = np.asarray(lab, dtype=np.float32)
    fy = (values[..., 0]+.16)/1.16
    f = np.stack((fy+values[..., 1]/5, fy, fy-values[..., 2]/2), axis=-1)
    xyz = np.where(f > _DELTA, f**3, 3*_DELTA**2*(f-4/29)) * _WHITE
    linear = xyz @ _XYZ_RGB.T
    rgb = np.where(linear <= .0031308, linear*12.92,
                   1.055*np.maximum(linear, 0)**(1/2.4)-.055)
    return np.uint8(np.clip(rgb*255+.5, 0, 255))


def match_lab_context(generated: np.ndarray, source: np.ndarray,
                      inference_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fit once per generated patch; sliders blend cached results in Nuke."""
    support = np.uint8(np.asarray(inference_mask) > 0)*255
    known = np.asarray(Image.fromarray(support).filter(ImageFilter.MaxFilter(17))) == 0
    step = max(1, int(np.ceil(np.sqrt(known.size/65536))))
    selected = known[::step, ::step]
    count = int(np.count_nonzero(selected))
    details = {"method": "reinhard_lab", "applied": False, "samples": count}
    if count < 64:
        return generated.copy(), {**details, "reason": "insufficient_clean_context"}
    target = rgb_to_lab(np.asarray(generated)[::step, ::step][selected])
    reference = rgb_to_lab(np.asarray(source)[::step, ::step][selected])
    source_mean, reference_mean = target.mean(0), reference.mean(0)
    source_std, reference_std = target.std(0), reference.std(0)
    gain = np.ones(3, dtype=np.float32)
    np.divide(reference_std, source_std, out=gain, where=source_std > 1e-5)
    gain = np.clip(gain, .25, 4)
    offset = reference_mean-source_mean*gain
    result = np.empty_like(generated)
    for y in range(0, len(generated), 128):
        result[y:y+128] = lab_to_rgb(rgb_to_lab(generated[y:y+128])*gain+offset)
    details.update(applied=True, gain=gain.tolist(), offset=offset.tolist())
    return result, details
