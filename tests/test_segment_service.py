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
from kyven.segment.output import read_logits_npz
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


class HoleProvider(SyntheticProvider):
    def predict(self, request: SegmentRequest, cancellation: CancellationToken) -> SegmentPrediction:
        mask = np.ones((5, 5), dtype=np.bool_)
        mask[2, 2] = False
        return SegmentPrediction(mask=mask, score=0.8)


class LogitProvider(SyntheticProvider):
    def predict(self, request, cancellation):
        logits = np.asarray([[-2.0, -0.25, 0.25, 2.0]], dtype=np.float32)
        return SegmentPrediction(mask=logits > 0, logits=logits, score=0.95)


class SegmentServiceTests(unittest.TestCase):
    def test_service_preserves_logits_and_writes_confidence_trimap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.new("RGB", (4, 1), "white").save(source)
            registry = ProviderRegistry()
            registry.register("logits", LogitProvider)
            logits_output = root / "logits.npz"
            trimap_output = root / "trimap.png"
            SegmentService(registry).run(
                SegmentRequest(
                    source=source,
                    output=root / "mask.png",
                    points=(PointPrompt(1, 0),),
                    provider_id="logits",
                    logits_output=logits_output,
                    trimap_output=trimap_output,
                    confidence_width=1.0,
                    fill_holes=False,
                )
            )
            np.testing.assert_allclose(read_logits_npz(logits_output), [[-2, -0.25, 0.25, 2]])
            with Image.open(trimap_output) as trimap:
                self.assertEqual(np.asarray(trimap).tolist(), [[0, 128, 128, 255]])

    def test_service_fills_enclosed_holes_and_reports_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.new("RGB", (5, 5), color="white").save(source)
            output = root / "mask.png"
            registry = ProviderRegistry()
            registry.register("holes", HoleProvider)

            result = SegmentService(registry).run(
                SegmentRequest(
                    source=source,
                    output=output,
                    points=(PointPrompt(2, 2),),
                    provider_id="holes",
                    max_hole_area=10,
                )
            )

            with Image.open(output) as image:
                self.assertEqual(image.getpixel((2, 2)), 255)
            self.assertEqual(result.metadata["postprocess"]["filled_holes"], 1)

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
