"""Refinement service with trimap generation, ROI reconstruction, and atomic output."""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from kyven.cache_manifest import read_valid_manifest, write_manifest_atomic
from kyven.cancellation import CancellationToken
from kyven.errors import ErrorCode, KyvenError
from kyven.refine.models import RefineRequest, RefineResult
from kyven.refine.trimap import generate_trimap, normalize_trimap
from kyven.segment.output import write_mask_png_atomic
from kyven.segment.providers.registry import ProviderRegistry
from kyven.segment.roi import ResolvedRegion, resolve_region


def _full_region(width: int, height: int) -> ResolvedRegion:
    return ResolvedRegion(
        x0=0,
        y0=0,
        x1=width,
        y1=height,
        source_width=width,
        source_height=height,
    )


def _auto_region(trimap: np.ndarray, padding: int) -> ResolvedRegion:
    """Bound foreground and unknown trimap pixels, then add model context."""

    height, width = trimap.shape
    ys, xs = np.nonzero(trimap > 0)
    if xs.size == 0:
        return _full_region(width, height)
    return ResolvedRegion(
        x0=max(0, int(xs.min()) - padding),
        y0=max(0, int(ys.min()) - padding),
        x1=min(width, int(xs.max()) + 1 + padding),
        y1=min(height, int(ys.max()) + 1 + padding),
        source_width=width,
        source_height=height,
    )


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
        full_mask_pixels = np.asarray(mask)
        full_trimap = (
            generate_trimap(
                full_mask_pixels,
                request.foreground_radius,
                request.background_radius,
            )
            if request.generate_trimap
            else normalize_trimap(full_mask_pixels)
        )
        effective_crop_mode = (
            "manual"
            if request.crop_mode == "full" and request.roi is not None
            else request.crop_mode
        )
        if effective_crop_mode == "manual":
            region = resolve_region(request.roi, source.width, source.height)
        elif effective_crop_mode == "auto":
            region = _auto_region(full_trimap, request.context_padding)
        else:
            region = _full_region(source.width, source.height)
        crop_box = (region.x0, region.y0, region.x1, region.y1)
        source_crop = source.crop(crop_box)
        trimap_pixels = full_trimap[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]]
        token.report_progress(0.10, "Prepared ViTMatte trimap")
        trimap_output = request.trimap_output
        if trimap_output is not None:
            write_mask_png_atomic(trimap_output, full_trimap)
        provider = self._registry.activate(request.provider_id)
        capabilities = provider.capabilities
        cache_key = request.cache_key(
            capabilities.provider_version,
            capabilities.model_checksum,
        )
        cached = read_valid_manifest(
            request.output,
            cache_key,
            (request.output, request.trimap_output),
        )
        if cached is not None:
            metadata = dict(cached.get("metadata") or {})
            metadata["cache_hit"] = True
            token.report_progress(1.0, "Reused cached refinement")
            return RefineResult(
                output=request.output,
                trimap_output=request.trimap_output,
                cache_key=cache_key,
                metadata=metadata,
            )
        token.report_progress(0.15, "Loading ViTMatte")
        with tempfile.TemporaryDirectory(prefix="kyven-refine-") as directory:
            root = Path(directory)
            source_path = root / "source.png"
            trimap_path = root / "trimap.png"
            source_crop.save(source_path)
            Image.fromarray(trimap_pixels, mode="L").save(trimap_path)
            prepared = replace(
                request,
                source=source_path,
                mask=trimap_path,
                crop_mode="full",
                roi=None,
            )
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
        metadata["crop_mode"] = effective_crop_mode
        metadata["context_padding"] = request.context_padding
        metadata["processing_roi"] = region.metadata()
        metadata["cache_hit"] = False
        write_manifest_atomic(request.output, cache_key, metadata)
        return RefineResult(
            output=request.output,
            trimap_output=trimap_output,
            cache_key=cache_key,
            metadata=metadata,
        )
