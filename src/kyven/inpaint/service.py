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
from kyven.inpaint.models import InpaintRequest, InpaintResult
from kyven.segment.output import write_mask_png_atomic
from kyven.segment.providers.registry import ProviderRegistry
from kyven.segment.roi import ResolvedRegion, resolve_region


def _mask_filter_size(radius: int) -> int:
    return max(3, radius * 2 + 1) | 1


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


class InpaintService:
    def __init__(self, registry: ProviderRegistry) -> None:
        self._registry = registry

    def run(self, request: InpaintRequest, cancellation: CancellationToken | None = None) -> InpaintResult:
        request.validate()
        token = cancellation or CancellationToken()
        token.report_progress(0.03, "Reading Source and mask")
        with Image.open(request.source) as source_file, Image.open(request.mask) as mask_file:
            source = source_file.convert("RGB")
            mask = mask_file.convert("L")
        if source.size != mask.size:
            raise KyvenError(ErrorCode.INVALID_REQUEST, "Source and inpaint mask dimensions must match.")
        source_pixels = np.asarray(source, dtype=np.uint8)
        mask_pixels = np.asarray(mask, dtype=np.uint8)
        if request.invert_mask:
            mask_pixels = 255 - mask_pixels
        threshold = round(request.mask_threshold * 255.0)
        binary_mask = np.where(mask_pixels >= threshold, 255, 0).astype(np.uint8)
        if not np.any(binary_mask):
            _write_rgb_atomic(request.output, source_pixels)
            if request.mask_output is not None:
                write_mask_png_atomic(request.mask_output, binary_mask)
            token.report_progress(1.0, "Mask is empty; Source copied unchanged")
            return InpaintResult(request.output, request.mask_output, "empty-mask", {"empty_mask": True})

        if request.crop_mode == "manual":
            region = resolve_region(request.roi, source.width, source.height)  # type: ignore[arg-type]
        elif request.crop_mode == "full":
            region = ResolvedRegion(0, 0, source.width, source.height, source.width, source.height)
        else:
            safe_padding = (
                request.context_padding
                + abs(request.mask_grow)
                + round(request.mask_feather * 3)
            )
            region = _auto_region(binary_mask, safe_padding)
            assert region is not None
        crop_box = (region.x0, region.y0, region.x1, region.y1)
        source_crop = source.crop(crop_box)
        mask_crop = Image.fromarray(binary_mask, mode="L").crop(crop_box)
        if request.mask_grow > 0:
            mask_crop = mask_crop.filter(ImageFilter.MaxFilter(_mask_filter_size(request.mask_grow)))
        elif request.mask_grow < 0:
            mask_crop = mask_crop.filter(ImageFilter.MinFilter(_mask_filter_size(abs(request.mask_grow))))
        inference_mask = np.asarray(mask_crop, dtype=np.uint8)
        if not np.any(inference_mask):
            _write_rgb_atomic(request.output, source_pixels)
            if request.mask_output is not None:
                write_mask_png_atomic(request.mask_output, np.zeros_like(binary_mask))
            return InpaintResult(request.output, request.mask_output, "empty-roi-mask", {"empty_mask": True})

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

        merge_mask = Image.fromarray(inference_mask, mode="L")
        if request.mask_feather:
            merge_mask = merge_mask.filter(ImageFilter.GaussianBlur(float(request.mask_feather)))
        alpha = np.asarray(merge_mask, dtype=np.float32)[..., None] / 255.0
        full_merge_mask = np.zeros_like(binary_mask)
        full_merge_mask[region.y0 : region.y1, region.x0 : region.x1] = np.asarray(merge_mask)
        if request.mask_output is not None:
            write_mask_png_atomic(request.mask_output, full_merge_mask)
        original_crop = source_pixels[region.y0 : region.y1, region.x0 : region.x1]
        merged_crop = np.clip(original_crop * (1.0 - alpha) + predicted * alpha, 0, 255).astype(np.uint8)
        output = source_pixels.copy()
        output[region.y0 : region.y1, region.x0 : region.x1] = merged_crop
        token.report_progress(0.95, "Writing inpaint result")
        _write_rgb_atomic(request.output, output)
        token.report_progress(1.0, "Inpaint complete")
        metadata = dict(prediction.metadata)
        metadata.update({"processing_roi": region.metadata(), "crop_mode": request.crop_mode})
        return InpaintResult(
            request.output,
            request.mask_output,
            request.cache_key(capabilities.provider_version, capabilities.model_checksum),
            metadata,
        )
