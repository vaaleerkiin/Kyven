"""Pure helpers for translating Nuke coordinates into Kyven requests."""

from __future__ import annotations

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


def point(x: float, nuke_y: float, image_height: int, label: str) -> dict[str, Any]:
    """Convert Nuke's bottom-left Y coordinate to image top-left coordinates."""

    return {
        "x": float(x),
        "y": float(image_height) - float(nuke_y),
        "label": label,
    }


def segment_payload(
    *,
    source: str,
    output: str,
    model_index: int,
    profile: str,
    image_height: int,
    positive_enabled: bool,
    positive_xy: tuple[float, float],
    negative_enabled: bool,
    negative_xy: tuple[float, float],
    box_enabled: bool,
    box: tuple[float, float, float, float],
) -> dict[str, Any]:
    points = []
    if positive_enabled:
        points.append(point(*positive_xy, image_height, "positive"))
    if negative_enabled:
        points.append(point(*negative_xy, image_height, "negative"))
    box_payload = None
    if box_enabled:
        x0, y0, x1, y1 = box
        box_payload = {
            "x0": float(x0),
            "y0": float(image_height) - float(y1),
            "x1": float(x1),
            "y1": float(image_height) - float(y0),
        }
    return {
        "source": source,
        "output": output,
        "model_id": MODEL_IDS[model_index],
        "profile": profile,
        "points": points,
        "box": box_payload,
        "multimask_output": True,
    }
