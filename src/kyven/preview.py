"""Fast CPU-only previews that never load a model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from kyven.errors import ErrorCode, KyvenError
from kyven.inpaint.masks import prepare_inpaint_masks
from kyven.refine.models import RefineRequest
from kyven.refine.tile_preview import draw_tile_preview
from kyven.refine.trimap import generate_trimap, normalize_trimap
from kyven.segment.models import BoxPrompt, ExecutionProfile
from kyven.segment.output import confidence_trimap, read_logits_npz, write_mask_png_atomic
from kyven.segment.postprocess import fill_enclosed_holes
from kyven.segment.roi import resolve_region


def _absolute_file(payload: dict[str, Any], name: str, *, must_exist: bool) -> Path:
    try:
        value = Path(str(payload[name]))
    except (KeyError, TypeError, ValueError) as exc:
        raise KyvenError(
            code=ErrorCode.INVALID_REQUEST,
            message=f"{name} is required and must be a valid path.",
            technical_detail=str(exc),
        ) from exc
    if not value.is_absolute():
        raise KyvenError(
            code=ErrorCode.INVALID_REQUEST,
            message=f"{name} must be an absolute path.",
        )
    if must_exist and not value.is_file():
        raise KyvenError(
            code=ErrorCode.INVALID_REQUEST,
            message=f"{name} does not exist: {value}",
        )
    return value


def _roi(payload: dict[str, Any]) -> BoxPrompt | None:
    value = payload.get("roi")
    if value is None:
        return None
    try:
        return BoxPrompt(
            float(value["x0"]),
            float(value["y0"]),
            float(value["x1"]),
            float(value["y1"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise KyvenError(
            code=ErrorCode.INVALID_REQUEST,
            message="Preview ROI is invalid.",
            technical_detail=str(exc),
        ) from exc


def prepare_trimap_preview(payload: dict[str, Any]) -> dict[str, Any]:
    """Write an exact trimap preview without running ViTMatte."""

    mask_path = _absolute_file(payload, "mask", must_exist=True)
    output_path = _absolute_file(payload, "output", must_exist=False)
    try:
        foreground_radius = int(payload.get("foreground_radius", 10))
        background_radius = int(payload.get("background_radius", 15))
    except (TypeError, ValueError) as exc:
        raise KyvenError(
            code=ErrorCode.INVALID_REQUEST,
            message="Trimap radii must be integers.",
            technical_detail=str(exc),
        ) from exc
    if foreground_radius < 0 or background_radius < 0:
        raise KyvenError(
            code=ErrorCode.INVALID_REQUEST,
            message="Trimap radii must be zero or greater.",
        )
    try:
        with Image.open(mask_path) as image_file:
            mask = image_file.convert("L")
            width, height = mask.size
            roi = _roi(payload)
            region = resolve_region(roi, width, height) if roi is not None else None
            pixels = np.asarray(mask)
    except (OSError, ValueError) as exc:
        raise KyvenError(
            code=ErrorCode.INVALID_REQUEST,
            message=f"Could not read trimap input: {mask_path}",
            technical_detail=str(exc),
        ) from exc

    trimap = (
        generate_trimap(pixels, foreground_radius, background_radius)
        if bool(payload.get("generate_trimap", True))
        else normalize_trimap(pixels)
    )
    if region is not None:
        full = np.zeros((height, width), dtype=np.uint8)
        full[region.y0 : region.y1, region.x0 : region.x1] = trimap[
            region.y0 : region.y1, region.x0 : region.x1]
        trimap = full
    try:
        write_mask_png_atomic(output_path, trimap)
    except OSError as exc:
        raise KyvenError(
            code=ErrorCode.INVALID_REQUEST,
            message=f"Could not write trimap preview: {output_path}",
            technical_detail=str(exc),
        ) from exc
    return {
        "output": str(output_path),
        "generated": bool(payload.get("generate_trimap", True)),
        "foreground_radius": foreground_radius,
        "background_radius": background_radius,
    }


def prepare_refine_tiles_preview(payload: dict[str, Any]) -> dict[str, Any]:
    """Plan and draw tiles without activating any model provider."""

    mask = _absolute_file(payload, 'mask', must_exist=True)
    output = _absolute_file(payload, 'output', must_exist=False)
    try:
        profile = ExecutionProfile(str(payload.get('profile', 'balanced')))
        default_tile = {'low_memory': 512, 'balanced': 1024, 'quality': 0}[profile.value]
        request = RefineRequest(
            source=mask, mask=mask, output=output, profile=profile,
            roi=_roi(payload), crop_mode=str(payload.get('crop_mode', 'full')),
            context_padding=int(payload.get('context_padding', 128)),
            generate_trimap=bool(payload.get('generate_trimap', True)),
            foreground_radius=int(payload.get('foreground_radius', 10)),
            background_radius=int(payload.get('background_radius', 15)),
            tile_size=int(payload.get('tile_size', default_tile)),
            tile_overlap=int(payload.get('tile_overlap', 64)),
            skip_known_tiles=bool(payload.get('skip_known_tiles', True)),
        )
        if 'image_width' in payload and 'image_height' in payload:
            with Image.open(mask) as image:
                if image.size != (int(payload['image_width']), int(payload['image_height'])):
                    raise ValueError('Source and mask/trimap dimensions must match.')
        return draw_tile_preview(request, labels=bool(payload.get('preview_labels', False)))
    except (OSError, ValueError, TypeError) as exc:
        raise KyvenError(ErrorCode.INVALID_REQUEST, 'Could not prepare the refinement tile preview.',
                         technical_detail=str(exc)) from exc


def postprocess_mask_preview(payload: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a displayed matte from the raw SAM mask without running SAM."""

    source_path = _absolute_file(payload, "source", must_exist=True)
    output_path = _absolute_file(payload, "output", must_exist=False)
    try:
        max_hole_area = int(payload.get("max_hole_area", 2_048))
    except (TypeError, ValueError) as exc:
        raise KyvenError(
            code=ErrorCode.INVALID_REQUEST,
            message="Maximum hole area must be an integer.",
            technical_detail=str(exc),
        ) from exc
    if max_hole_area < 0:
        raise KyvenError(
            code=ErrorCode.INVALID_REQUEST,
            message="Maximum hole area must be zero or greater.",
        )
    try:
        with Image.open(source_path) as image_file:
            raw_mask = np.asarray(image_file.convert("L"))
    except OSError as exc:
        raise KyvenError(
            code=ErrorCode.INVALID_REQUEST,
            message=f"Could not read raw SAM mask: {source_path}",
            technical_detail=str(exc),
        ) from exc

    enabled = bool(payload.get("fill_holes", False))
    if enabled:
        result = fill_enclosed_holes(raw_mask, max_hole_area)
        mask = result.mask
        filled_holes = result.filled_holes
        filled_pixels = result.filled_pixels
    else:
        mask = raw_mask
        filled_holes = 0
        filled_pixels = 0
    try:
        write_mask_png_atomic(output_path, mask)
    except OSError as exc:
        raise KyvenError(
            code=ErrorCode.INVALID_REQUEST,
            message=f"Could not write mask preview: {output_path}",
            technical_detail=str(exc),
        ) from exc
    return {
        "output": str(output_path),
        "fill_holes": enabled,
        "max_hole_area": max_hole_area,
        "filled_holes": filled_holes,
        "filled_pixels": filled_pixels,
    }


def confidence_trimap_preview(payload: dict[str, Any]) -> dict[str, Any]:
    """Rebuild an exact SAM-confidence trimap from cached logits without inference."""

    logits_path = _absolute_file(payload, "logits", must_exist=True)
    output_path = _absolute_file(payload, "output", must_exist=False)
    try:
        width = float(payload.get("confidence_width", 1.0))
    except (TypeError, ValueError) as exc:
        raise KyvenError(ErrorCode.INVALID_REQUEST, "Confidence Width must be numeric.") from exc
    trimap = confidence_trimap(read_logits_npz(logits_path), width)
    write_mask_png_atomic(output_path, trimap)
    return {"output": str(output_path), "confidence_width": width}


def prepare_inpaint_mask_preview(payload: dict[str, Any]) -> dict[str, Any]:
    """Write the exact binary mask used by the Inpaint model."""

    mask_path = _absolute_file(payload, "mask", must_exist=True)
    output_path = _absolute_file(payload, "output", must_exist=False)
    try:
        threshold = float(payload.get("mask_threshold", 0.5))
        model_grow = int(payload.get("mask_grow", 12))
    except (TypeError, ValueError) as exc:
        raise KyvenError(
            ErrorCode.INVALID_REQUEST,
            "Inpaint mask preview controls are invalid.",
            technical_detail=str(exc),
        ) from exc
    if (
        not 0 <= threshold <= 1
        or not -128 <= model_grow <= 128
    ):
        raise KyvenError(ErrorCode.INVALID_REQUEST, "Inpaint mask preview controls are out of range.")
    channel = str(payload.get("mask_channel", "luminance"))
    try:
        with Image.open(mask_path) as image_file:
            if channel == "alpha" and "A" in image_file.getbands():
                pixels = np.asarray(image_file.getchannel("A"), dtype=np.uint8)
            else:
                pixels = np.asarray(image_file.convert("L"), dtype=np.uint8)
    except OSError as exc:
        raise KyvenError(
            ErrorCode.INVALID_REQUEST,
            f"Could not read Inpaint mask preview input: {mask_path}",
            technical_detail=str(exc),
        ) from exc
    model_mask, _composite_mask = prepare_inpaint_masks(
        pixels,
        preprocess=bool(payload.get("preprocess_mask", True)),
        invert=bool(payload.get("invert_mask", False)),
        threshold=threshold,
        model_grow=model_grow,
    )
    write_mask_png_atomic(output_path, model_mask)
    return {
        "output": str(output_path),
        "preprocess_mask": bool(payload.get("preprocess_mask", True)),
        "mask_threshold": threshold,
        "mask_grow": model_grow,
        "model_nonzero_pixels": int(np.count_nonzero(model_mask)),
        "nonzero_pixels": int(np.count_nonzero(model_mask)),
    }
