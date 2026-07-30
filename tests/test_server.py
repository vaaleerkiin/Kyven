from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from kyven.cancellation import CancellationToken
from kyven.client import KyvenClient, KyvenClientError
from kyven.models.catalog import ModelCatalog
from kyven.refine.models import RefinementCapabilities, RefinePrediction, RefineRequest
from kyven.refine.providers.base import RefinementProvider
from kyven.refine.service import RefineService
from kyven.segment.models import (
    ExecutionProfile,
    ProviderCapabilities,
    SegmentPrediction,
    SegmentRequest,
)
from kyven.segment.output import write_mask_png_atomic
from kyven.segment.providers.base import SegmentationProvider
from kyven.segment.providers.registry import ProviderRegistry
from kyven.segment.service import SegmentService
from kyven.segment.video import VideoSegmentResult, VideoSegmentService
from kyven.server.app import KyvenServer, ServerConfig
from kyven.server.jobs import JobManager


class ServerSyntheticProvider(SegmentationProvider):
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_id="sam2.1-small",
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
            supported_profiles=(ExecutionProfile.BALANCED,),
        )

    def predict(self, request: SegmentRequest, cancellation: CancellationToken) -> SegmentPrediction:
        return SegmentPrediction(np.ones((2, 2), dtype=np.bool_), 0.75)

    def unload(self) -> None:
        return

    def propagate_video(self, request, cancellation):
        outputs = []
        for index in range(request.last_frame - request.first_frame + 1):
            cancellation.raise_if_cancelled()
            output = request.output_for_index(index)
            write_mask_png_atomic(output, np.ones((2, 2), dtype=np.bool_))
            outputs.append(output)
        return VideoSegmentResult(
            outputs=tuple(outputs),
            first_frame=request.first_frame,
            last_frame=request.last_frame,
            key_frame=request.key_frame,
            direction=request.direction,
            metadata={"provider": "synthetic"},
        )


class ServerRefineProvider(RefinementProvider):
    @property
    def capabilities(self) -> RefinementCapabilities:
        return RefinementCapabilities(
            provider_id="vitmatte-small-composition-1k",
            display_name="Synthetic Refine",
            provider_version="1",
            model_checksum="fixture",
            license_name="CC0-1.0",
            license_url="https://creativecommons.org/publicdomain/zero/1.0/",
            supports_cpu=True,
            supports_tiling=True,
            minimum_vram_mb=0,
        )

    def predict(self, request: RefineRequest, cancellation: CancellationToken) -> RefinePrediction:
        with Image.open(request.source) as image:
            height, width = image.height, image.width
        return RefinePrediction(np.full((height, width), 0.5, dtype=np.float32))

    def unload(self) -> None:
        return


class ServerTests(unittest.TestCase):
    def test_authenticated_http_job_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            output = root / "matte.png"
            Image.new("RGB", (4, 4), "white").save(source)
            registry = ProviderRegistry()
            registry.register("sam2.1-small", ServerSyntheticProvider)
            registry.register("vitmatte-small-composition-1k", ServerRefineProvider)
            token = "x" * 32
            server = KyvenServer(
                ServerConfig(token=token, models_dir=root, port=0, available_vram_mb=8192),
                JobManager(
                    SegmentService(registry),
                    VideoSegmentService(registry),
                    RefineService(registry),
                ),
                registry,
                ModelCatalog.builtin(),
            )
            server.start()
            try:
                client = KyvenClient(f"http://127.0.0.1:{server.port}", token)
                self.assertEqual(client.health()["status"], "ok")
                self.assertEqual(client.health()["api_version"], 6)
                self.assertEqual(len(client.models()), 5)
                job_id = client.submit_segment(
                    {
                        "source": str(source.resolve()),
                        "output": str(output.resolve()),
                        "model_id": "sam2.1-small",
                        "points": [{"x": 2, "y": 2, "label": "positive"}],
                        "roi": {"x0": 1, "y0": 1, "x1": 3, "y1": 3},
                    }
                )
                result = client.wait(job_id, timeout_seconds=5)
                self.assertEqual(result["status"], "succeeded")
                self.assertTrue(output.is_file())
                with Image.open(output) as image_mask:
                    self.assertEqual(image_mask.size, (4, 4))
                    self.assertEqual(image_mask.getpixel((0, 0)), 0)
                    self.assertEqual(image_mask.getpixel((1, 1)), 255)
                frames = root / "frames"
                frames.mkdir()
                Image.new("RGB", (4, 4), "white").save(frames / "00001.jpg")
                Image.new("RGB", (4, 4), "white").save(frames / "00002.jpg")
                video_job_id = client.submit_video(
                    {
                        "frames_dir": str(frames.resolve()),
                        "output_pattern": str((root / "video_matte.%04d.png").resolve()),
                        "first_frame": 1,
                        "last_frame": 2,
                        "key_frame": 1,
                        "direction": "forward",
                        "model_id": "sam2.1-small",
                        "points": [{"x": 2, "y": 2, "label": "positive"}],
                        "roi": {"x0": 1, "y0": 1, "x1": 3, "y1": 3},
                    }
                )
                video_result = client.wait(video_job_id, timeout_seconds=5)
                self.assertEqual(video_result["status"], "succeeded")
                self.assertEqual(video_result["progress"], 1.0)
                self.assertEqual(video_result["progress_message"], "Propagation complete")
                self.assertEqual(video_result["result"]["output_count"], 2)
                self.assertTrue((root / "video_matte.0002.png").is_file())
                with Image.open(root / "video_matte.0002.png") as video_mask:
                    self.assertEqual(video_mask.size, (4, 4))
                    self.assertEqual(video_mask.getpixel((0, 0)), 0)
                    self.assertEqual(video_mask.getpixel((1, 1)), 255)
                refine_output = root / "refined.png"
                refine_job_id = client.submit_refine(
                    {
                        "source": str(source.resolve()),
                        "mask": str(output.resolve()),
                        "output": str(refine_output.resolve()),
                        "model_id": "vitmatte-small-composition-1k",
                        "generate_trimap": True,
                        "foreground_radius": 1,
                        "background_radius": 1,
                        "tile_size": 0,
                    }
                )
                refine_result = client.wait(refine_job_id, timeout_seconds=5)
                self.assertEqual(refine_result["status"], "succeeded")
                self.assertTrue(refine_output.is_file())
                with self.assertRaises(KyvenClientError):
                    KyvenClient(f"http://127.0.0.1:{server.port}", "y" * 32).health()
            finally:
                server.shutdown()


if __name__ == "__main__":
    unittest.main()
