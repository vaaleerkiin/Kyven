"""Engine service for segmentation execution and atomic mask output."""

from __future__ import annotations

import numpy as np

from kyven.cancellation import CancellationToken
from kyven.segment.models import SegmentRequest, SegmentResult
from kyven.segment.output import write_mask_png_atomic
from kyven.segment.providers.registry import ProviderRegistry


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
        prediction = provider.predict(request, token)
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
