from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from kyven.cancellation import CancellationToken
from kyven.client import KyvenClient, KyvenClientError
from kyven.models.catalog import ModelCatalog
from kyven.segment.models import (
    ExecutionProfile,
    ProviderCapabilities,
    SegmentPrediction,
    SegmentRequest,
)
from kyven.segment.providers.base import SegmentationProvider
from kyven.segment.providers.registry import ProviderRegistry
from kyven.segment.service import SegmentService
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


class ServerTests(unittest.TestCase):
    def test_authenticated_http_job_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            output = root / "matte.png"
            Image.new("RGB", (2, 2), "white").save(source)
            registry = ProviderRegistry()
            registry.register("sam2.1-small", ServerSyntheticProvider)
            token = "x" * 32
            server = KyvenServer(
                ServerConfig(token=token, models_dir=root, port=0, available_vram_mb=8192),
                JobManager(SegmentService(registry)),
                registry,
                ModelCatalog.builtin(),
            )
            server.start()
            try:
                client = KyvenClient(f"http://127.0.0.1:{server.port}", token)
                self.assertEqual(client.health()["status"], "ok")
                self.assertEqual(len(client.models()), 4)
                job_id = client.submit_segment(
                    {
                        "source": str(source.resolve()),
                        "output": str(output.resolve()),
                        "model_id": "sam2.1-small",
                        "points": [{"x": 1, "y": 1, "label": "positive"}],
                    }
                )
                result = client.wait(job_id, timeout_seconds=5)
                self.assertEqual(result["status"], "succeeded")
                self.assertTrue(output.is_file())
                with self.assertRaises(KyvenClientError):
                    KyvenClient(f"http://127.0.0.1:{server.port}", "y" * 32).health()
            finally:
                server.shutdown()


if __name__ == "__main__":
    unittest.main()
