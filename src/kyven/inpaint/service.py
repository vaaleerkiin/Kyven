"""ROI-aware inpainting with source-safe masked compositing."""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from kyven.cancellation import CancellationToken
from kyven.errors import ErrorCode, KyvenError
from kyven.inpaint.masks import mask_filter_size, prepare_inpaint_masks
from kyven.inpaint.models import InpaintRequest, InpaintResult
from kyven.segment.output import write_mask_png_atomic
from kyven.segment.providers.registry import ProviderRegistry
from kyven.segment.roi import ResolvedRegion, resolve_region


def _write_rgb_atomic(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=path.suffix, dir=path.parent)
        os.close(descriptor)
        temporary = Path(name)
        image = Image.fromarray(np.asarray(pixels, dtype=np.uint8), mode="RGB")
        image.save(temporary)
        os.replace(temporary, path)
    except Exception as exc:
        if temporary:
            temporary.unlink(missing_ok=True)
        raise KyvenError(ErrorCode.OUTPUT_FAILED, f"Could not write inpaint output: {path}", technical_detail=str(exc)) from exc


def _auto_region(mask: np.ndarray, padding: int) -> ResolvedRegion | None:
    ys, xs = np.nonzero(mask >= 128)
    if not len(xs):
        return None
    height, width = mask.shape
    return ResolvedRegion(
        max(0, int(xs.min()) - padding),
        max(0, int(ys.min()) - padding),
        min(width, int(xs.max()) + 1 + padding),
        min(height, int(ys.max()) + 1 + padding),
        width,
        height,
    )


def _match_patch_color(
    predicted: np.ndarray,
    original: np.ndarray,
    inference_mask: np.ndarray,
    strength: float,
) -> tuple[np.ndarray, list[float]]:
    """Remove a local RGB offset measured in the clean ring around the generated area."""

    if strength <= 0:
        return predicted, [0.0, 0.0, 0.0]
    mask_image = Image.fromarray(inference_mask, mode="L")
    ring_radius = 16
    expanded = np.asarray(
        mask_image.filter(ImageFilter.MaxFilter(mask_filter_size(ring_radius))),
        dtype=np.uint8,
    )
    ring = (expanded >= 128) & (inference_mask < 128)
    if int(np.count_nonzero(ring)) < 32:
        return predicted, [0.0, 0.0, 0.0]
    differences = original[ring].astype(np.float32) - predicted[ring].astype(np.float32)
    offset = np.clip(np.median(differences, axis=0), -32.0, 32.0) * float(strength)
    matched = np.clip(predicted.astype(np.float32) + offset, 0, 255).astype(np.uint8)
    return matched, [round(float(value), 3) for value in offset]


class InpaintService:
    def __init__(self, registry: ProviderRegistry) -> None:
        self._registry = registry

    def run(self, request: InpaintRequest, cancellation: CancellationToken | None = None) -> InpaintResult:
        request.validate()
        token = cancellation or CancellationToken()
        token.report_progress(0.03, "Reading Source and mask")
        with Image.open(request.source) as source_file, Image.open(request.mask) as mask_file:
            source = source_file.convert("RGB")
            if request.mask_channel == "alpha" and "A" in mask_file.getbands():
                mask = mask_file.getchannel("A")
            else:
                mask = mask_file.convert("L")
        if source.size != mask.size:
            raise KyvenError(ErrorCode.INVALID_REQUEST, "Source and inpaint mask dimensions must match.")
        source_pixels = np.asarray(source, dtype=np.uint8)
        mask_pixels = np.asarray(mask, dtype=np.uint8)
        if request.model_mask is not None:
            with Image.open(request.model_mask) as model_mask_file:
                model_mask = model_mask_file.convert("L")
            if model_mask.size != source.size:
                raise KyvenError(
                    ErrorCode.INVALID_REQUEST,
                    "Source and exported model mask dimensions must match.",
                )
            inference_full = np.where(
                np.asarray(model_mask, dtype=np.uint8) >= 128,
                255,
                0,
            ).astype(np.uint8)
            merge_full = (
                255 - mask_pixels
                if request.preprocess_mask and request.invert_mask
                else mask_pixels.copy()
            )
        else:
            inference_full, merge_full = prepare_inpaint_masks(
                mask_pixels,
                preprocess=request.preprocess_mask,
                invert=request.invert_mask,
                threshold=request.mask_threshold,
                model_grow=request.mask_grow,
            )
        if not np.any(inference_full):
            _write_rgb_atomic(request.output, source_pixels)
            if request.patch_output is not None:
                _write_rgb_atomic(request.patch_output, source_pixels)
            if request.mask_output is not None:
                write_mask_png_atomic(request.mask_output, np.zeros_like(mask_pixels))
            token.report_progress(1.0, "Mask is empty; Source copied unchanged")
            return InpaintResult(request.output, request.mask_output, request.patch_output, "empty-mask", {"empty_mask": True})

        if request.crop_mode == "manual":
            region = resolve_region(request.roi, source.width, source.height)  # type: ignore[arg-type]
        elif request.crop_mode == "full":
            region = ResolvedRegion(0, 0, source.width, source.height, source.width, source.height)
        else:
            mask_coverage = np.maximum(
                inference_full,
                np.where(merge_full > 0, 255, 0).astype(np.uint8),
            )
            region = _auto_region(
                mask_coverage,
                request.context_padding,
            )
            assert region is not None
        crop_box = (region.x0, region.y0, region.x1, region.y1)
        source_crop = source.crop(crop_box)
        if request.model_mask is not None:
            inference_mask = inference_full[region.y0 : region.y1, region.x0 : region.x1]
            merge_crop_pixels = merge_full[region.y0 : region.y1, region.x0 : region.x1]
        else:
            raw_crop = np.asarray(Image.fromarray(mask_pixels, mode="L").crop(crop_box))
            inference_mask, merge_crop_pixels = prepare_inpaint_masks(
                raw_crop,
                preprocess=request.preprocess_mask,
                invert=request.invert_mask,
                threshold=request.mask_threshold,
                model_grow=request.mask_grow,
            )
        if not np.any(inference_mask):
            _write_rgb_atomic(request.output, source_pixels)
            if request.patch_output is not None:
                _write_rgb_atomic(request.patch_output, source_pixels)
            if request.mask_output is not None:
                write_mask_png_atomic(request.mask_output, np.zeros_like(mask_pixels))
            return InpaintResult(request.output, request.mask_output, request.patch_output, "empty-roi-mask", {"empty_mask": True})

        provider = self._registry.activate(request.provider_id)
        capabilities = provider.capabilities
        token.report_progress(0.12, "Preparing inpaint crop")
        with tempfile.TemporaryDirectory(prefix="kyven-inpaint-") as directory:
            root = Path(directory)
            crop_source_path = root / "source.png"
            crop_mask_path = root / "mask.png"
            source_crop.save(crop_source_path)
            Image.fromarray(inference_mask, mode="L").save(crop_mask_path)
            prediction = provider.predict(
                replace(request, source=crop_source_path, mask=crop_mask_path, roi=None, crop_mode="full"),
                token,
            )
        predicted = np.asarray(prediction.rgb, dtype=np.uint8)
        expected = (region.height, region.width, 3)
        if predicted.shape != expected:
            raise KyvenError(ErrorCode.INFERENCE_FAILED, "The inpaint provider returned an invalid image size.", technical_detail=f"Expected {expected}, received {predicted.shape}.")

        original_crop = source_pixels[region.y0 : region.y1, region.x0 : region.x1]
        predicted, color_offset = _match_patch_color(
            predicted,
            original_crop,
            inference_mask,
            request.edge_color_match,
        )
        merge_mask = Image.fromarray(merge_crop_pixels, mode="L")
        alpha = np.asarray(merge_mask, dtype=np.float32)[..., None] / 255.0
        full_merge_mask = np.zeros_like(mask_pixels)
        full_merge_mask[region.y0 : region.y1, region.x0 : region.x1] = np.asarray(merge_mask)
        if request.mask_output is not None:
            write_mask_png_atomic(request.mask_output, full_merge_mask)
        full_patch = source_pixels.copy()
        full_patch[region.y0 : region.y1, region.x0 : region.x1] = predicted
        if request.patch_output is not None:
            _write_rgb_atomic(request.patch_output, full_patch)
        merged_crop = np.clip(original_crop * (1.0 - alpha) + predicted * alpha, 0, 255).astype(np.uint8)
        output = source_pixels.copy()
        output[region.y0 : region.y1, region.x0 : region.x1] = merged_crop
        token.report_progress(0.95, "Writing inpaint result")
        _write_rgb_atomic(request.output, output)
        token.report_progress(1.0, "Inpaint complete")
        metadata = dict(prediction.metadata)
        metadata.update({
            "processing_roi": region.metadata(),
            "crop_mode": request.crop_mode,
            "edge_color_offset": color_offset,
        })
        return InpaintResult(
            request.output,
            request.mask_output,
            request.patch_output,
            request.cache_key(capabilities.provider_version, capabilities.model_checksum),
            metadata,
        )
