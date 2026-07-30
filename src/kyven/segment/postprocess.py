"""Dependency-free binary mask post-processing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class HoleFillResult:
    mask: np.ndarray
    filled_pixels: int
    filled_holes: int


def fill_enclosed_holes(mask: np.ndarray, max_area: int = 2_048) -> HoleFillResult:
    """Fill enclosed background components without changing the exterior mask edge.

    A max_area of zero fills every enclosed component. The run-length connected-component
    implementation avoids optional SciPy/OpenCV dependencies and remains efficient on large frames.
    """

    pixels = np.asarray(mask)
    foreground = pixels >= 0.5 if np.issubdtype(pixels.dtype, np.floating) else pixels > 0
    if foreground.ndim != 2:
        raise ValueError(f"Expected a two-dimensional mask, received {foreground.shape}.")
    if max_area < 0:
        raise ValueError("max_area must be greater than or equal to zero.")
    height, width = foreground.shape
    if height == 0 or width == 0:
        return HoleFillResult(foreground.copy(), 0, 0)

    parents: list[int] = []
    areas: list[int] = []
    touches_border: list[bool] = []
    runs: list[tuple[int, int, int, int]] = []

    def find(component: int) -> int:
        root = component
        while parents[root] != root:
            root = parents[root]
        while parents[component] != component:
            parent = parents[component]
            parents[component] = root
            component = parent
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if areas[left_root] < areas[right_root]:
            left_root, right_root = right_root, left_root
        parents[right_root] = left_root
        areas[left_root] += areas[right_root]
        touches_border[left_root] = touches_border[left_root] or touches_border[right_root]

    previous: list[tuple[int, int, int]] = []
    for y in range(height):
        background = ~foreground[y]
        padded = np.empty(width + 2, dtype=np.bool_)
        padded[0] = False
        padded[-1] = False
        padded[1:-1] = background
        transitions = np.flatnonzero(padded[1:] != padded[:-1])
        current: list[tuple[int, int, int]] = []
        for start, end in zip(transitions[0::2], transitions[1::2], strict=True):
            component = len(parents)
            parents.append(component)
            areas.append(int(end - start))
            touches_border.append(y in {0, height - 1} or start == 0 or end == width)
            current.append((int(start), int(end), component))
            runs.append((y, int(start), int(end), component))

        previous_index = 0
        for start, end, component in current:
            while previous_index < len(previous) and previous[previous_index][1] <= start:
                previous_index += 1
            candidate = previous_index
            while candidate < len(previous) and previous[candidate][0] < end:
                union(component, previous[candidate][2])
                candidate += 1
        previous = current

    output = foreground.copy()
    fillable: set[int] = set()
    for component in range(len(parents)):
        root = find(component)
        if not touches_border[root] and (max_area == 0 or areas[root] <= max_area):
            fillable.add(root)

    filled_pixels = 0
    for y, start, end, component in runs:
        root = find(component)
        if root in fillable:
            output[y, start:end] = True
            filled_pixels += end - start
    return HoleFillResult(output, filled_pixels, len(fillable))
