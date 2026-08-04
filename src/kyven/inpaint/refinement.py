"""Multi-scale feature refinement for the scripted Big-LaMa generator.

This is a dependency-light adaptation of LaMa's Apache-2.0 refinement procedure.  It optimizes
the generator's bottleneck features so that each higher-resolution prediction agrees with the
completed lower-resolution image while preserving known source pixels.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def pyramid_sizes(width: int, height: int, max_scales: int, min_side: int = 256) -> tuple[tuple[int, int], ...]:
    """Return low-to-high pyramid sizes, always ending at the requested native size."""

    def modulo_eight(value: int) -> int:
        return (int(value) + 7) // 8 * 8

    sizes = [(modulo_eight(width), modulo_eight(height))]
    while len(sizes) < max_scales and min(sizes[-1]) >= min_side * 2:
        current_width, current_height = sizes[-1]
        sizes.append((modulo_eight(current_width // 2), modulo_eight(current_height // 2)))
    return tuple(reversed(sizes))


def _forward_layers(layers: Sequence[Any], value: Any) -> Any:
    for layer in layers:
        if not hasattr(layer, "forward") and getattr(layer, "original_name", "") == "ReLU":
            value = value.relu()
        else:
            value = layer(*value) if isinstance(value, tuple) else layer(value)
    return value


def refine_big_lama(
    model: Any,
    image: Any,
    mask: Any,
    *,
    steps: int,
    strength: float,
    max_scales: int,
    cancellation: Any,
) -> Any:
    """Run LaMa's coarse-to-fine bottleneck optimization on one padded RGB crop."""

    import torch
    from torch.nn import functional

    cancellation.raise_if_cancelled()
    layers = tuple(model.generator.model.children())
    if len(layers) < 24:
        raise RuntimeError("The installed Big-LaMa checkpoint does not expose refinement layers.")
    front = layers[:5]
    rear = layers[5:]
    full_height, full_width = image.shape[-2:]
    sizes = pyramid_sizes(full_width, full_height, max_scales)
    reference = None

    for scale_index, (width, height) in enumerate(sizes):
        cancellation.raise_if_cancelled()
        scaled_image = functional.interpolate(
            image, size=(height, width), mode="bilinear", align_corners=False
        )
        scaled_mask = functional.interpolate(mask, size=(height, width), mode="nearest")
        generator_input = torch.cat((scaled_image * (1.0 - scaled_mask), scaled_mask), dim=1)
        with torch.no_grad():
            features = _forward_layers(front, generator_input)
        if not isinstance(features, tuple) or len(features) != 2:
            raise RuntimeError("The installed Big-LaMa bottleneck has an unsupported layout.")

        local = features[0].detach().requires_grad_(reference is not None)
        global_features = features[1].detach().requires_grad_(reference is not None)
        iterations = steps if reference is not None else 1
        optimizer = (
            torch.optim.Adam((local, global_features), lr=0.002 * float(strength))
            if reference is not None
            else None
        )
        prediction = None
        for iteration in range(iterations):
            cancellation.raise_if_cancelled()
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            prediction = _forward_layers(rear, (local, global_features))
            if reference is None:
                break

            down_prediction = functional.interpolate(
                prediction, size=reference.shape[-2:], mode="bilinear", align_corners=False
            )
            down_mask = functional.interpolate(
                scaled_mask, size=reference.shape[-2:], mode="bilinear", align_corners=False
            )
            down_mask = 1.0 - functional.max_pool2d(1.0 - down_mask, 15, stride=1, padding=7)
            known = scaled_mask.expand_as(prediction) < 0.5
            compare = down_mask.expand_as(down_prediction) >= 0.5
            source_loss = torch.mean(torch.abs(prediction[known] - scaled_image[known]))
            if torch.any(compare):
                scale_loss = torch.mean(torch.abs(down_prediction[compare] - reference[compare]))
                loss = source_loss + scale_loss
            else:
                loss = source_loss
            if iteration + 1 < iterations:
                loss.backward()
                optimizer.step()
            progress = (scale_index + (iteration + 1) / iterations) / len(sizes)
            cancellation.report_progress(
                0.45 + 0.42 * progress,
                f"Big-LaMa refinement scale {scale_index + 1}/{len(sizes)}, step {iteration + 1}/{iterations}",
            )

        assert prediction is not None
        reference = (scaled_mask * prediction + (1.0 - scaled_mask) * scaled_image).detach()

    return reference
