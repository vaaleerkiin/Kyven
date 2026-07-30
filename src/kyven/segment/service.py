"""Engine service for segmentation execution and atomic mask output."""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from kyven.cancellation import CancellationToken
from kyven.segment.models import SegmentPrediction, SegmentRequest, SegmentResult
from kyven.segment.output import write_mask_png_atomic
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
        provider = self._registry.activate(request.provider_id)
        capabilities = provider.capabilities
        prediction = self._predict(provider, request, token)
        token.raise_if_cancelled()

        write_mask_png_atomic(request.output, np.asarray(prediction.mask))

        cache_key = request.cache_key(
            provider_version=capabilities.provider_version,
            model_checksum=capabilities.model_checksum,
        )
        return SegmentResult(
            output=request.output,
            score=prediction.score,
            cache_key=cache_key,
            metadata=prediction.metadata,
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
            metadata=metadata,
        )
