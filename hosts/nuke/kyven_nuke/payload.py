"""Pure helpers for translating Nuke coordinates into Kyven requests."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

MODEL_IDS = (
    "sam2.1-tiny",
    "sam2.1-small",
    "sam2.1-base-plus",
    "sam2.1-large",
)

MODEL_LABELS = (
    "SAM 2.1 Tiny (4 GB)",
    "SAM 2.1 Small (6 GB, recommended for 8 GB)",
    "SAM 2.1 Base+ (8 GB, may require fallback)",
    "SAM 2.1 Large (12 GB+)",
)

REFINE_MODEL_IDS = ("vitmatte-small-composition-1k",)
REFINE_MODEL_LABELS = ("ViTMatte Small (4 GB+, recommended for 8 GB)",)
INPAINT_MODEL_IDS = ("lama-2025jan-onnx",)
INPAINT_MODEL_LABELS = ("LaMa (CPU / low VRAM)",)


def point(x: float, nuke_y: float, image_height: int, label: str) -> dict[str, Any]:
    """Convert Nuke's bottom-left Y coordinate to image top-left coordinates."""

    return {
        "x": float(x),
        "y": float(image_height) - float(nuke_y),
        "label": label,
    }


def roi_box(
    box: tuple[float, float, float, float],
    image_height: int,
    image_width: int | None = None,
) -> dict[str, float]:
    """Normalize, clamp, and convert a Nuke BBox into a top-left Processing ROI."""

    left, right = sorted((float(box[0]), float(box[2])))
    bottom, top = sorted((float(box[1]), float(box[3])))
    if image_width is not None:
        finite = all(math.isfinite(value) for value in (left, bottom, right, top))
        left = max(0.0, min(float(image_width), left)) if finite else 0.0
        right = max(0.0, min(float(image_width), right)) if finite else 0.0
        bottom = max(0.0, min(float(image_height), bottom)) if finite else 0.0
        top = max(0.0, min(float(image_height), top)) if finite else 0.0
        if right <= left or top <= bottom:
            left, bottom = 0.0, 0.0
            right, top = float(image_width), float(image_height)
    return {
        "x0": left,
        "y0": float(image_height) - top,
        "x1": right,
        "y1": float(image_height) - bottom,
    }


def segment_payload(
    *,
    source: str,
    output: str,
    raw_output: str | None = None,
    model_index: int,
    profile: str,
    image_width: int,
    image_height: int,
    positive_points: Sequence[tuple[float, float]],
    negative_points: Sequence[tuple[float, float]],
    box_enabled: bool,
    box: tuple[float, float, float, float],
    fill_holes: bool = True,
    max_hole_area: int = 2_048,
) -> dict[str, Any]:
    points = [point(*xy, image_height, "positive") for xy in positive_points]
    points.extend(point(*xy, image_height, "negative") for xy in negative_points)
    roi_payload = None
    if box_enabled:
        roi_payload = roi_box(box, image_height, image_width)
    return {
        "source": source,
        "output": output,
        "raw_output": raw_output,
        "model_id": MODEL_IDS[model_index],
        "profile": profile,
        "points": points,
        "box": None,
        "roi": roi_payload,
        "multimask_output": True,
        "fill_holes": bool(fill_holes),
        "max_hole_area": int(max_hole_area),
    }


def segment_video_payload(
    *,
    frames_dir: str,
    output_pattern: str,
    raw_output_pattern: str | None = None,
    model_index: int,
    profile: str,
    image_width: int,
    image_height: int,
    positive_points: Sequence[tuple[float, float]],
    negative_points: Sequence[tuple[float, float]],
    box_enabled: bool,
    box: tuple[float, float, float, float],
    first_frame: int,
    last_frame: int,
    key_frame: int,
    direction: str,
    fill_holes: bool = True,
    max_hole_area: int = 2_048,
    animated_rois: Sequence[tuple[int, tuple[float, float, float, float]]] = (),
) -> dict[str, Any]:
    image_payload = segment_payload(
        source="unused",
        output="unused",
        raw_output=None,
        model_index=model_index,
        profile=profile,
        image_width=image_width,
        image_height=image_height,
        positive_points=positive_points,
        negative_points=negative_points,
        box_enabled=box_enabled,
        box=box,
        fill_holes=fill_holes,
        max_hole_area=max_hole_area,
    )
    return {
        "frames_dir": frames_dir,
        "output_pattern": output_pattern,
        "raw_output_pattern": raw_output_pattern,
        "model_id": image_payload["model_id"],
        "profile": profile,
        "points": image_payload["points"],
        "box": None,
        "roi": None if animated_rois else image_payload["roi"],
        "rois": [
            {"frame": int(frame), **roi_box(frame_box, image_height, image_width)}
            for frame, frame_box in animated_rois
        ],
        "first_frame": int(first_frame),
        "last_frame": int(last_frame),
        "key_frame": int(key_frame),
        "direction": direction,
        "offload_video_to_cpu": True,
        "offload_state_to_cpu": True,
        "fill_holes": image_payload["fill_holes"],
        "max_hole_area": image_payload["max_hole_area"],
    }


def refine_payload(
    *,
    source: str,
    mask: str,
    output: str,
    trimap_output: str | None,
    model_index: int,
    profile: str,
    image_width: int,
    image_height: int,
    roi_enabled: bool,
    roi: tuple[float, float, float, float],
    generate_trimap: bool,
    foreground_radius: int,
    background_radius: int,
    tile_size: int = 0,
    tile_overlap: int = 64,
) -> dict[str, Any]:
    roi_payload = None
    if roi_enabled:
        roi_payload = roi_box(roi, image_height, image_width)
    return {
        "source": source,
        "mask": mask,
        "output": output,
        "trimap_output": trimap_output,
        "model_id": REFINE_MODEL_IDS[model_index],
        "profile": profile,
        "roi": roi_payload,
        "generate_trimap": bool(generate_trimap),
        "foreground_radius": int(foreground_radius),
        "background_radius": int(background_radius),
        "tile_size": int(tile_size),
        "tile_overlap": int(tile_overlap),
    }


def inpaint_payload(
    *, source: str, mask: str, output: str, mask_output: str, model_index: int, profile: str,
    image_width: int, image_height: int, crop_mode: str,
    roi: tuple[float, float, float, float], context_padding: int,
    mask_grow: int, mask_feather: float, mask_threshold: float, invert_mask: bool,
    processing_size: int,
) -> dict[str, Any]:
    return {
        "source": source,
        "mask": mask,
        "output": output,
        "mask_output": mask_output,
        "model_id": INPAINT_MODEL_IDS[model_index],
        "profile": profile,
        "crop_mode": crop_mode,
        "roi": roi_box(roi, image_height, image_width) if crop_mode == "manual" else None,
        "context_padding": int(context_padding),
        "mask_grow": int(mask_grow),
        "mask_feather": float(mask_feather),
        "mask_threshold": float(mask_threshold),
        "invert_mask": bool(invert_mask),
        "processing_size": int(processing_size),
    }
