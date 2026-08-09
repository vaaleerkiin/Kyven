"""Engine service for segmentation execution and atomic mask output."""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from kyven.cancellation import CancellationToken
from kyven.segment.models import SegmentPrediction, SegmentRequest, SegmentResult
from kyven.segment.output import confidence_trimap, write_logits_npz_atomic, write_mask_png_atomic
from kyven.segment.postprocess import fill_enclosed_holes
from kyven.segment.providers.registry import ProviderRegistry
from kyven.segment.roi import expand_mask, resolve_region, translate_box, translate_points


class SegmentService:
    """Coordinate validation, provider execution, cache identity, and output."""

    def __init__(self, registry: ProviderRegistry) -> None:
        self._registry = registry

    def run(
        self,
        request: SegmentRequest,
        cancellation: CancellationToken | None = None,
    ) -> SegmentResult:
        """Execute a request synchronously; hosts should call this off their UI thread."""

        request.validate()
        token = cancellation or CancellationToken()
        token.raise_if_cancelled()
        token.report_progress(0.05, "Preparing segmentation")
        provider = self._registry.activate(request.provider_id)
        capabilities = provider.capabilities
        prediction = self._predict(provider, request, token)
        token.raise_if_cancelled()
        token.report_progress(0.90, "Post-processing matte")

        mask = np.asarray(prediction.mask)
        logits = None if prediction.logits is None else np.asarray(prediction.logits, dtype=np.float32)
        if logits is not None and logits.shape != mask.shape:
            logits = np.asarray(
                Image.fromarray(logits, mode="F").resize(
                    (mask.shape[1], mask.shape[0]), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )
        if request.logits_output is not None and logits is not None:
            write_logits_npz_atomic(request.logits_output, logits)
        if request.trimap_output is not None and logits is not None:
            write_mask_png_atomic(
                request.trimap_output, confidence_trimap(logits, request.confidence_width)
            )
        if request.raw_output is not None:
            write_mask_png_atomic(request.raw_output, mask)
        metadata = dict(prediction.metadata)
        if request.fill_holes:
            filled = fill_enclosed_holes(mask, request.max_hole_area)
            mask = filled.mask
            metadata["postprocess"] = {
                "fill_holes": True,
                "max_hole_area": request.max_hole_area,
                "filled_holes": filled.filled_holes,
                "filled_pixels": filled.filled_pixels,
            }
        write_mask_png_atomic(request.output, mask)
        token.report_progress(1.0, "Segmentation complete")

        cache_key = request.cache_key(
            provider_version=capabilities.provider_version,
            model_checksum=capabilities.model_checksum,
        )
        return SegmentResult(
            output=request.output,
            score=prediction.score,
            cache_key=cache_key,
            metadata=metadata,
        )

    @staticmethod
    def _predict(provider, request: SegmentRequest, token: CancellationToken) -> SegmentPrediction:
        if request.roi is None:
            return provider.predict(request, token)

        with Image.open(request.source) as source_image:
            image = source_image.convert("RGB")
            region = resolve_region(request.roi, image.width, image.height)
            points = translate_points(request.points, region)
            box = translate_box(request.box, region)
            if region.is_full_frame:
                prepared = replace(request, points=points, box=box, roi=None)
                prediction = provider.predict(prepared, token)
            else:
                with tempfile.TemporaryDirectory(prefix="kyven-roi-") as directory:
                    cropped_source = Path(directory) / "source.png"
                    image.crop((region.x0, region.y0, region.x1, region.y1)).save(cropped_source)
                    prepared = replace(
                        request,
                        source=cropped_source,
                        points=points,
                        box=box,
                        roi=None,
                    )
                    prediction = provider.predict(prepared, token)

        metadata = dict(prediction.metadata)
        metadata["processing_roi"] = region.metadata()
        return SegmentPrediction(
            mask=expand_mask(np.asarray(prediction.mask), region),
            score=prediction.score,
            logits=(
                None
                if prediction.logits is None
                else _expand_logits(np.asarray(prediction.logits), region)
            ),
            metadata=metadata,
        )


def _expand_logits(logits: np.ndarray, region) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    if values.shape != (region.height, region.width):
        values = np.asarray(
            Image.fromarray(values, mode="F").resize(
                (region.width, region.height), Image.Resampling.BILINEAR
            ), dtype=np.float32
        )
    full = np.full((region.source_height, region.source_width), -100.0, dtype=np.float32)
    full[region.y0 : region.y1, region.x0 : region.x1] = values
    return full
