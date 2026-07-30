from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from kyven.cancellation import CancellationToken
from kyven.errors import ErrorCode, KyvenError
from kyven.segment.models import (
    ExecutionProfile,
    PointPrompt,
    ProviderCapabilities,
    SegmentPrediction,
    SegmentRequest,
)
from kyven.segment.providers.base import SegmentationProvider
from kyven.segment.providers.registry import ProviderRegistry
from kyven.segment.service import SegmentService


class SyntheticProvider(SegmentationProvider):
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_id="synthetic",
            display_name="Synthetic",
            provider_version="1",
            model_family="test",
            model_variant="test",
            model_checksum="fixture",
            license_name="CC0-1.0",
            license_url="https://creativecommons.org/publicdomain/zero/1.0/",
            supports_cpu=True,
            supports_points=True,
            supports_boxes=True,
            minimum_vram_mb=0,
            supported_profiles=(ExecutionProfile.LOW_MEMORY,),
        )

    def predict(self, request: SegmentRequest, cancellation: CancellationToken) -> SegmentPrediction:
        cancellation.raise_if_cancelled()
        return SegmentPrediction(
            mask=np.asarray([[False, True], [True, False]]),
            score=0.9,
            metadata={"provider": "synthetic"},
        )

    def unload(self) -> None:
        return


class SegmentServiceTests(unittest.TestCase):
    def test_service_writes_grayscale_png_and_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.new("RGB", (2, 2), color="white").save(source)
            output = root / "mask.png"
            registry = ProviderRegistry()
            registry.register("synthetic", SyntheticProvider)
            request = SegmentRequest(
                source=source,
                output=output,
                points=(PointPrompt(1, 1),),
                provider_id="synthetic",
            )
            result = SegmentService(registry).run(request)
            self.assertTrue(output.is_file())
            self.assertEqual(Image.open(output).getpixel((1, 0)), 255)
            self.assertEqual(result.score, 0.9)
            self.assertEqual(len(result.cache_key), 64)

    def test_cancelled_job_does_not_create_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.new("RGB", (2, 2), color="white").save(source)
            output = root / "mask.png"
            registry = ProviderRegistry()
            registry.register("synthetic", SyntheticProvider)
            request = SegmentRequest(
                source=source,
                output=output,
                points=(PointPrompt(1, 1),),
                provider_id="synthetic",
            )
            token = CancellationToken()
            token.cancel()
            with self.assertRaises(KyvenError) as caught:
                SegmentService(registry).run(request, token)
            self.assertEqual(caught.exception.code, ErrorCode.CANCELLED)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
