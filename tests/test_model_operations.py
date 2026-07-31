from __future__ import annotations

import hashlib
import tempfile
import time
import unittest
from pathlib import Path

from kyven.models.catalog import ModelCatalog, ModelSpec
from kyven.models.operations import ModelOperationManager
from kyven.segment.providers.registry import ProviderRegistry


class ModelOperationTests(unittest.TestCase):
    def test_download_verify_and_remove_local_catalog_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.weights"
            payload = b"trusted model data" * 128
            source.write_bytes(payload)
            spec = ModelSpec(
                model_id="test-model",
                task="refine",
                display_name="Test Model",
                provider="test",
                config="",
                filename="test-model.weights",
                source=source.as_uri(),
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
                parameters_millions=1.0,
                recommended_vram_mb=0,
                supports_cpu=True,
                license="Apache-2.0",
                license_url="",
                commercial_use=True,
                redistribution=True,
            )
            models_dir = root / "models"
            operations = ModelOperationManager(
                ModelCatalog((spec,)),
                models_dir,
                ProviderRegistry(),
            )
            try:
                install = self._wait(operations, operations.submit("download", spec.model_id))
                self.assertEqual(install["status"], "succeeded")
                self.assertEqual(install["progress"], 1.0)
                self.assertEqual((models_dir / spec.filename).read_bytes(), payload)

                remove = self._wait(operations, operations.submit("remove", spec.model_id))
                self.assertEqual(remove["status"], "succeeded")
                self.assertFalse((models_dir / spec.filename).exists())
            finally:
                operations.shutdown()

    @staticmethod
    def _wait(manager: ModelOperationManager, operation_id: str) -> dict:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            operation = manager.get(operation_id)
            if operation["status"] in {"succeeded", "failed", "cancelled"}:
                return operation
            time.sleep(0.01)
        raise AssertionError("Model operation did not finish")


if __name__ == "__main__":
    unittest.main()
