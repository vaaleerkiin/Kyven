"""Engine service for segmentation execution and atomic mask output."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from kyven.cancellation import CancellationToken
from kyven.errors import ErrorCode, KyvenError
from kyven.segment.models import SegmentRequest, SegmentResult
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
        provider = self._registry.get(request.provider_id)
        capabilities = provider.capabilities
        prediction = provider.predict(request, token)
        token.raise_if_cancelled()

        mask = np.asarray(prediction.mask)
        if mask.ndim != 2:
            raise KyvenError(
                code=ErrorCode.INFERENCE_FAILED,
                message="The provider returned an invalid mask shape.",
                technical_detail=f"Expected 2 dimensions, received {mask.shape}.",
            )
        if np.issubdtype(mask.dtype, np.floating):
            pixels = np.clip(mask, 0.0, 1.0) * 255.0
        else:
            pixels = mask.astype(np.uint8) * 255
        self._write_png_atomic(request.output, pixels.astype(np.uint8))

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
    def _write_png_atomic(output: Path, pixels: np.ndarray) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{output.stem}-",
                suffix=".png",
                dir=output.parent,
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
            Image.fromarray(pixels, mode="L").save(temporary_path, format="PNG")
            os.replace(temporary_path, output)
        except Exception as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise KyvenError(
                code=ErrorCode.OUTPUT_FAILED,
                message=f"Could not write segmentation output: {output}",
                technical_detail=str(exc),
                recoverable=True,
                suggested_action="Check the output path and available disk space.",
            ) from exc

