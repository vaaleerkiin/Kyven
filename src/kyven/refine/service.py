"""Refinement service with trimap generation, ROI reconstruction, and atomic output."""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from kyven.cancellation import CancellationToken
from kyven.errors import ErrorCode, KyvenError
from kyven.refine.models import RefineRequest, RefineResult
from kyven.refine.trimap import generate_trimap, normalize_trimap
from kyven.segment.output import write_mask_png_atomic
from kyven.segment.providers.registry import ProviderRegistry
from kyven.segment.roi import resolve_region


class RefineService:
    def __init__(self, registry: ProviderRegistry) -> None:
        self._registry = registry

    def run(
        self,
        request: RefineRequest,
        cancellation: CancellationToken | None = None,
    ) -> RefineResult:
        request.validate()
        token = cancellation or CancellationToken()
        token.raise_if_cancelled()
        token.report_progress(0.03, "Reading Source and mask")
        with Image.open(request.source) as source_file, Image.open(request.mask) as mask_file:
            source = source_file.convert("RGB")
            mask = mask_file.convert("L")
        if source.size != mask.size:
            raise KyvenError(
                code=ErrorCode.INVALID_REQUEST,
                message="Source and mask/trimap dimensions must match.",
            )
        region = (
            resolve_region(request.roi, source.width, source.height)
            if request.roi is not None
            else None
        )
        crop_box = (
            (region.x0, region.y0, region.x1, region.y1)
            if region is not None
            else (0, 0, source.width, source.height)
        )
        source_crop = source.crop(crop_box)
        mask_crop = mask.crop(crop_box)
        mask_pixels = np.asarray(mask_crop)
        trimap_pixels = (
            generate_trimap(mask_pixels, request.foreground_radius, request.background_radius)
            if request.generate_trimap
            else normalize_trimap(mask_pixels)
        )
        token.report_progress(0.10, "Prepared ViTMatte trimap")
        trimap_output = request.trimap_output
        if trimap_output is not None:
            if region is None:
                full_trimap = trimap_pixels
            else:
                full_trimap = np.zeros((source.height, source.width), dtype=np.uint8)
                full_trimap[region.y0 : region.y1, region.x0 : region.x1] = trimap_pixels
            write_mask_png_atomic(trimap_output, full_trimap)
        provider = self._registry.activate(request.provider_id)
        capabilities = provider.capabilities
        token.report_progress(0.15, "Loading ViTMatte")
        with tempfile.TemporaryDirectory(prefix="kyven-refine-") as directory:
            root = Path(directory)
            source_path = root / "source.png"
            trimap_path = root / "trimap.png"
            source_crop.save(source_path)
            Image.fromarray(trimap_pixels, mode="L").save(trimap_path)
            prepared = replace(request, source=source_path, mask=trimap_path, roi=None)
            prediction = provider.predict(prepared, token)
        token.raise_if_cancelled()
        token.report_progress(0.93, "Writing refined alpha")
        refined = np.asarray(prediction.alpha, dtype=np.float32)
        expected = (source_crop.height, source_crop.width)
        if refined.shape != expected:
            raise KyvenError(
                code=ErrorCode.INFERENCE_FAILED,
                message="The refinement provider returned an invalid alpha size.",
                technical_detail=f"Expected {expected}, received {refined.shape}.",
            )
        if region is not None:
            full = np.asarray(mask, dtype=np.float32) / 255.0
            full[region.y0 : region.y1, region.x0 : region.x1] = refined
            refined = full
        write_mask_png_atomic(request.output, refined)
        token.report_progress(1.0, "Refinement complete")
        metadata = dict(prediction.metadata)
        metadata["trimap"] = {
            "generated": request.generate_trimap,
            "foreground_radius": request.foreground_radius,
            "background_radius": request.background_radius,
        }
        if region is not None:
            metadata["processing_roi"] = region.metadata()
        return RefineResult(
            output=request.output,
            trimap_output=trimap_output,
            cache_key=request.cache_key(
                capabilities.provider_version,
                capabilities.model_checksum,
            ),
            metadata=metadata,
        )
